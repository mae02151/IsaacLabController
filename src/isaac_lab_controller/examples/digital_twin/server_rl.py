"""
강화학습(RL) 제어 웹 서버

IsaacLab의 RL 학습(Train)과 추론(Inference)을 웹에서 제어할 수 있는 FastAPI 서버입니다.
학습/추론을 서브프로세스로 실행하고, 출력 로그를 WebSocket으로 실시간 스트리밍합니다.
프레임 캡처 모듈(frame_capture.py)을 통해 시뮬레이션 화면도 웹으로 스트리밍합니다.

사용법:
    python server_rl.py

    # 브라우저에서 http://localhost:8001 접속
"""

import asyncio
import json
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# =====================================================
# 설정
# =====================================================

ISAACLAB_DIR = os.path.expanduser("~/IsaacLab")
TASK_NAME = "Isaac-Reach-OpenManipulatorX-v0"
DEFAULT_TRAIN_NUM_ENVS = 256
INFERENCE_NUM_ENVS = 1
SERVER_PORT = 8001

# 프레임 캡처 설정
FRAME_PATH = "/tmp/isaaclab_rl_frame.jpg"
FRAME_CAPTURE_INTERVAL = 10  # N 스텝마다 캡처
CAMERA_PARAMS_PATH = "/tmp/isaaclab_rl_camera.json"

# 런처 및 스크립트 경로
CURRENT_DIR = Path(__file__).parent
LAUNCHER_PATH = CURRENT_DIR / "rl_launcher.py"
TRAIN_SCRIPT = os.path.join(ISAACLAB_DIR, "scripts/reinforcement_learning/rsl_rl/train.py")
PLAY_SCRIPT = os.path.join(ISAACLAB_DIR, "scripts/reinforcement_learning/rsl_rl/play.py")


# =====================================================
# Pydantic 모델
# =====================================================

class TrainStartRequest(BaseModel):
    """학습 시작 요청"""
    num_envs: int = DEFAULT_TRAIN_NUM_ENVS


class CameraRequest(BaseModel):
    """카메라 파라미터 요청"""
    azimuth: float = 45.0
    elevation: float = 30.0
    distance: float = 5.0


# =====================================================
# RL 프로세스 관리자
# =====================================================

class RLManager:
    """
    RL 학습/추론 서브프로세스 관리자

    한 번에 하나의 프로세스만 실행 가능합니다.
    stdout/stderr를 비동기로 읽어 로그 버퍼에 저장하고,
    WebSocket 클라이언트에 실시간 브로드캐스트합니다.
    rl_launcher.py를 통해 frame_capture 모듈을 주입하여
    시뮬레이션 프레임도 캡처합니다.
    """

    # RSL-RL 로그 패턴: "Learning iteration 42/300"
    _ITERATION_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
    # ETA 패턴: "ETA: 01:23:45"
    _ETA_RE = re.compile(r"ETA:\s+(\d+:\d+:\d+)")
    # Mean reward 패턴: "Mean reward: 12.50"
    _REWARD_RE = re.compile(r"Mean reward:\s+([-\d.]+)")

    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.mode: Optional[str] = None  # "train" 또는 "inference"
        self.running = False
        self.num_envs: int = 0
        self.logs: deque = deque(maxlen=10000)
        self.ws_clients: set = set()
        self._read_task: Optional[asyncio.Task] = None

        # 학습 진행률
        self.current_iteration: int = 0
        self.total_iterations: int = 0
        self.eta: str = ""
        self.mean_reward: Optional[float] = None

    async def start_train(self, num_envs: int = DEFAULT_TRAIN_NUM_ENVS) -> dict:
        """학습 시작 (rl_launcher.py 경유, 프레임 캡처 포함)"""
        if self.running:
            return {"success": False, "error": "이미 프로세스가 실행 중입니다."}

        cmd = [
            sys.executable,
            str(LAUNCHER_PATH),
            TRAIN_SCRIPT,
            f"--task={TASK_NAME}",
            f"--num_envs={num_envs}",
        ]
        self.num_envs = num_envs
        return await self._start_process(cmd, "train")

    async def start_inference(self) -> dict:
        """추론 시작 (rl_launcher.py 경유, 프레임 캡처 포함)"""
        if self.running:
            return {"success": False, "error": "이미 프로세스가 실행 중입니다."}

        cmd = [
            sys.executable,
            str(LAUNCHER_PATH),
            PLAY_SCRIPT,
            f"--task={TASK_NAME}",
            f"--num_envs={INFERENCE_NUM_ENVS}",
            "--hardware",
        ]
        self.num_envs = INFERENCE_NUM_ENVS
        return await self._start_process(cmd, "inference")

    async def _start_process(self, cmd: list, mode: str) -> dict:
        """서브프로세스 시작 (내부용)"""
        try:
            # 이전 프레임 파일 삭제
            if os.path.exists(FRAME_PATH):
                os.remove(FRAME_PATH)

            self.logs.clear()
            self.mode = mode
            self.running = True
            self.current_iteration = 0
            self.total_iterations = 0
            self.eta = ""
            self.mean_reward = None

            log_msg = f"[시스템] {mode} 시작: {' '.join(cmd[2:])}"
            self.logs.append(log_msg)
            await self._broadcast(log_msg)

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=ISAACLAB_DIR,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "RL_FRAME_PATH": FRAME_PATH,
                    "RL_FRAME_INTERVAL": str(FRAME_CAPTURE_INTERVAL),
                    "RL_CAMERA_PARAMS_PATH": CAMERA_PARAMS_PATH,
                },
            )

            # 출력 읽기 태스크 시작
            self._read_task = asyncio.create_task(self._read_output())

            return {"success": True, "mode": mode, "pid": self.process.pid}
        except Exception as e:
            self.running = False
            self.mode = None
            error_msg = f"[시스템] 프로세스 시작 실패: {e}"
            self.logs.append(error_msg)
            await self._broadcast(error_msg)
            return {"success": False, "error": str(e)}

    async def _read_output(self):
        """서브프로세스 stdout을 비동기로 읽기"""
        try:
            while self.process and self.process.returncode is None:
                line = await self.process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    self.logs.append(text)
                    self._parse_progress(text)
                    await self._broadcast(text)
        except Exception as e:
            error_msg = f"[시스템] 출력 읽기 오류: {e}"
            self.logs.append(error_msg)
            await self._broadcast(error_msg)
        finally:
            # 프로세스 종료 처리
            if self.process:
                await self.process.wait()
                exit_code = self.process.returncode
                end_msg = f"[시스템] 프로세스 종료 (exit code: {exit_code})"
                self.logs.append(end_msg)
                await self._broadcast(end_msg)
            self.running = False

    async def stop(self) -> dict:
        """실행 중인 프로세스 중지"""
        if not self.running or not self.process:
            return {"success": False, "error": "실행 중인 프로세스가 없습니다."}

        try:
            stop_msg = f"[시스템] {self.mode} 프로세스 종료 중..."
            self.logs.append(stop_msg)
            await self._broadcast(stop_msg)

            # SIGTERM으로 종료 시도
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                # 10초 후에도 종료 안 되면 SIGKILL
                self.process.kill()
                await self.process.wait()

            self.running = False
            mode = self.mode
            self.mode = None

            done_msg = f"[시스템] {mode} 프로세스 종료 완료"
            self.logs.append(done_msg)
            await self._broadcast(done_msg)

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_progress(self, text: str):
        """로그 라인에서 학습 진행률 정보 파싱"""
        m = self._ITERATION_RE.search(text)
        if m:
            self.current_iteration = int(m.group(1))
            self.total_iterations = int(m.group(2))
            return

        m = self._ETA_RE.search(text)
        if m:
            self.eta = m.group(1)
            return

        m = self._REWARD_RE.search(text)
        if m:
            self.mean_reward = float(m.group(1))

    def get_status(self) -> dict:
        """현재 상태 조회"""
        status = {
            "running": self.running,
            "mode": self.mode,
            "num_envs": self.num_envs,
            "log_lines": len(self.logs),
        }
        if self.mode == "train" and self.total_iterations > 0:
            status["progress"] = {
                "current": self.current_iteration,
                "total": self.total_iterations,
                "eta": self.eta,
                "mean_reward": self.mean_reward,
            }
        return status

    def get_logs(self, last_n: int = 500) -> list:
        """최근 로그 조회"""
        return list(self.logs)[-last_n:]

    async def _broadcast(self, message: str):
        """모든 WebSocket 클라이언트에 메시지 전송"""
        dead = set()
        for ws in self.ws_clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        self.ws_clients -= dead


# =====================================================
# FastAPI 앱 생성
# =====================================================

def create_rl_app() -> FastAPI:
    """RL 제어용 FastAPI 앱 생성"""
    app = FastAPI(
        title="IsaacLab RL Controller",
        description="강화학습 학습/추론 제어 API (프레임 스트리밍 포함)",
        version="0.1.0",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # RL 관리자 인스턴스
    rl_manager = RLManager()
    app.state.rl_manager = rl_manager

    # ----- API 라우트 -----

    @app.post("/api/rl/train/start")
    async def train_start(body: TrainStartRequest):
        """학습 시작"""
        return await rl_manager.start_train(body.num_envs)

    @app.post("/api/rl/train/stop")
    async def train_stop():
        """학습 중지"""
        return await rl_manager.stop()

    @app.post("/api/rl/inference/start")
    async def inference_start():
        """추론 시작"""
        return await rl_manager.start_inference()

    @app.post("/api/rl/inference/stop")
    async def inference_stop():
        """추론 중지"""
        return await rl_manager.stop()

    @app.get("/api/rl/status")
    async def get_status():
        """현재 상태 조회"""
        return {"success": True, "data": rl_manager.get_status()}

    @app.get("/api/rl/logs")
    async def get_logs(last_n: int = 500):
        """최근 로그 조회"""
        return {"success": True, "data": rl_manager.get_logs(last_n)}

    @app.get("/api/rl/frame")
    async def get_frame():
        """
        현재 시뮬레이션 프레임 조회

        frame_capture 모듈이 서브프로세스에서 캡처한 JPEG 프레임을 반환합니다.
        프로세스가 실행 중이 아니거나 아직 프레임이 없으면 204 반환.
        """
        if not os.path.exists(FRAME_PATH):
            return Response(status_code=204)

        try:
            with open(FRAME_PATH, "rb") as f:
                frame_data = f.read()
            if not frame_data:
                return Response(status_code=204)
            return Response(
                content=frame_data,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store"},
            )
        except Exception:
            return Response(status_code=204)

    @app.post("/api/rl/camera")
    async def set_camera(body: CameraRequest):
        """카메라 파라미터 설정 (서브프로세스와 파일로 공유)"""
        params = {
            "azimuth": body.azimuth,
            "elevation": body.elevation,
            "distance": body.distance,
        }
        try:
            tmp_path = CAMERA_PARAMS_PATH + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(params, f)
            os.replace(tmp_path, CAMERA_PARAMS_PATH)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @app.get("/api/health")
    async def health_check():
        """헬스체크"""
        return {"status": "ok", "isaaclab_dir": ISAACLAB_DIR}

    # ----- WebSocket 로그 스트리밍 -----

    @app.websocket("/ws/logs")
    async def websocket_logs(websocket: WebSocket):
        """실시간 로그 스트리밍 WebSocket"""
        await websocket.accept()
        rl_manager.ws_clients.add(websocket)

        try:
            # 기존 로그 히스토리 전송
            for line in rl_manager.get_logs(500):
                await websocket.send_text(line)

            # 연결 유지 (클라이언트 메시지 대기)
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=30)
                except asyncio.TimeoutError:
                    # 타임아웃 시 핑 전송
                    await websocket.send_text("")
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            rl_manager.ws_clients.discard(websocket)

    # ----- 정적 파일 및 프론트엔드 -----

    frontend_path = Path(__file__).parent / "frontend_rl"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def serve_root():
        """루트 경로에서 index.html 제공"""
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Frontend not found", "path": str(index_path)}

    return app


# =====================================================
# 메인 실행
# =====================================================

def main():
    print("=" * 60)
    print("[RL SERVER] IsaacLab RL 제어 서버 시작")
    print("=" * 60)
    print(f"[RL SERVER] IsaacLab 경로: {ISAACLAB_DIR}")
    print(f"[RL SERVER] 태스크: {TASK_NAME}")
    print(f"[RL SERVER] Python: {sys.executable}")
    print(f"[RL SERVER] 런처: {LAUNCHER_PATH}")
    print(f"[RL SERVER] 프레임 경로: {FRAME_PATH}")
    print("")
    print("=" * 60)
    print(f"[RL SERVER] 웹 서버: http://localhost:{SERVER_PORT}")
    print("[RL SERVER] Ctrl+C로 종료")
    print("=" * 60)
    print("")

    app = create_rl_app()

    try:
        uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="warning")
    except KeyboardInterrupt:
        print("\n[RL SERVER] 서버 종료")


if __name__ == "__main__":
    main()
