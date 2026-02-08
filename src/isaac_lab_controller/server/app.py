"""
ControlServer - FastAPI 기반 제어 서버

IsaacLab 시뮬레이션을 웹에서 제어할 수 있는 HTTP/WebSocket 서버입니다.
"""

import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from isaac_lab_controller.adapters.base import SceneAdapter
from isaac_lab_controller.server.routes import camera, objects, materials, robot
from isaac_lab_controller.server.websocket_handler import WebSocketHandler
from isaac_lab_controller.utils.config import load_config, ControllerConfig

logger = logging.getLogger(__name__)


def create_app(scene_adapter: SceneAdapter) -> FastAPI:
    """
    FastAPI 앱 생성 (독립 실행용)
    
    server_standalone.py에서 사용됩니다.
    
    Args:
        scene_adapter: 시뮬레이션 환경 어댑터 (또는 프록시 어댑터)
    
    Returns:
        FastAPI: 설정된 FastAPI 앱
    """
    app = FastAPI(
        title="Isaac Lab Controller",
        description="IsaacLab 시뮬레이션 프론트엔드 제어 API",
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
    
    # 어댑터를 앱 상태에 저장 (라우트에서 접근용)
    app.state.scene = scene_adapter
    app.state.camera = scene_adapter.get_camera_adapter()
    app.state.objects = scene_adapter.get_object_adapter()
    app.state.materials = scene_adapter.get_material_adapter()
    app.state.robot = scene_adapter.get_robot_adapter() if hasattr(scene_adapter, 'get_robot_adapter') else None

    # API 라우트 등록
    app.include_router(camera.router, prefix="/api/camera", tags=["Camera"])
    app.include_router(objects.router, prefix="/api/objects", tags=["Objects"])
    app.include_router(materials.router, prefix="/api/materials", tags=["Materials"])
    app.include_router(robot.router, prefix="/api/robot", tags=["Robot"])

    # 정적 파일 서빙 (Frontend)
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    # 헬스체크
    @app.get("/api/health")
    async def health_check():
        try:
            is_running = scene_adapter.is_running() if hasattr(scene_adapter, 'is_running') else True
        except:
            is_running = False
        return {"status": "ok", "simulation_running": is_running}
    
    # WebSocket 핸들러
    ws_handler = WebSocketHandler(scene_adapter)
    
    @app.websocket("/ws/stream")
    async def websocket_stream(websocket: WebSocket):
        await ws_handler.stream_handler(websocket)
    
    @app.websocket("/ws/control")
    async def websocket_control(websocket: WebSocket):
        await ws_handler.control_handler(websocket)
    
    # 루트 경로에서 index.html 제공
    @app.get("/")
    async def serve_root():
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Frontend not found"}
    
    return app


class ControlServer:
    """
    IsaacLab 프론트엔드 제어 서버
    
    FastAPI를 기반으로 HTTP API와 WebSocket 스트리밍을 제공합니다.
    
    Example:
        >>> from isaac_lab_controller import ControlServer
        >>> 
        >>> adapter = MySceneAdapter(sim, scene)
        >>> server = ControlServer(adapter, config_path="config.yaml")
        >>> 
        >>> # 백그라운드에서 서버 시작
        >>> server.start_background()
        >>> 
        >>> # 시뮬레이션 루프
        >>> while sim.is_running():
        ...     sim.step()
        ...     server.update()  # 프레임 업데이트
    
    Args:
        scene_adapter: 시뮬레이션 환경 어댑터
        config_path: 설정 파일 경로 (선택)
        host: 서버 호스트 주소
        port: 서버 포트
    """
    
    def __init__(
        self,
        scene_adapter: SceneAdapter,
        config_path: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.scene = scene_adapter
        self.config = load_config(config_path) if config_path else ControllerConfig()
        self.host = host
        self.port = port
        
        # FastAPI 앱 생성
        self.app = self._create_app()
        
        # WebSocket 핸들러
        self.ws_handler = WebSocketHandler(scene_adapter)
        
        # 서버 스레드
        self._server_thread: Optional[threading.Thread] = None
        self._is_running = False
        
    def _create_app(self) -> FastAPI:
        """FastAPI 앱 생성 및 설정"""
        app = FastAPI(
            title="Isaac Lab Controller",
            description="IsaacLab 시뮬레이션 프론트엔드 제어 API",
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
        
        # 어댑터를 앱 상태에 저장 (라우트에서 접근용)
        app.state.scene = self.scene
        app.state.camera = self.scene.get_camera_adapter()
        app.state.objects = self.scene.get_object_adapter()
        app.state.materials = self.scene.get_material_adapter()
        app.state.robot = self.scene.get_robot_adapter() if hasattr(self.scene, 'get_robot_adapter') else None

        # API 라우트 등록
        app.include_router(camera.router, prefix="/api/camera", tags=["Camera"])
        app.include_router(objects.router, prefix="/api/objects", tags=["Objects"])
        app.include_router(materials.router, prefix="/api/materials", tags=["Materials"])
        app.include_router(robot.router, prefix="/api/robot", tags=["Robot"])

        # 정적 파일 서빙 (Frontend)
        frontend_path = Path(__file__).parent.parent / "frontend"
        if frontend_path.exists():
            app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

        # 헬스체크
        @app.get("/api/health")
        async def health_check():
            return {"status": "ok", "simulation_running": self.scene.is_running()}
        
        # WebSocket 엔드포인트
        from fastapi import WebSocket
        
        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            await self.ws_handler.stream_handler(websocket)
        
        @app.websocket("/ws/control")
        async def websocket_control(websocket: WebSocket):
            await self.ws_handler.control_handler(websocket)
        
        # 루트 경로에서 index.html 제공
        from fastapi.responses import FileResponse
        
        @app.get("/")
        async def serve_root():
            index_path = frontend_path / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "Frontend not found"}
        
        return app
    
    def start_background(self) -> None:
        """
        백그라운드 스레드에서 서버 시작
        
        시뮬레이션 루프와 동시에 실행할 때 사용합니다.
        """
        if self._is_running:
            logger.warning("서버가 이미 실행 중입니다.")
            return
        
        self._is_running = True
        self._server_thread = threading.Thread(
            target=self._run_server,
            daemon=True,
            name="ControlServer"
        )
        self._server_thread.start()
        logger.info(f"제어 서버 시작: http://{self.host}:{self.port}")
    
    def _run_server(self) -> None:
        """서버 실행 (내부용)"""
        try:
            import uvicorn
        except ImportError as e:
            print(f"[ERROR] uvicorn 임포트 실패: {e}")
            print("[ERROR] 'pip install uvicorn' 로 설치하세요.")
            self._is_running = False
            return

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        server = uvicorn.Server(config)

        # 새 이벤트 루프 생성 (스레드에서)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            print(f"[ControlServer] uvicorn 서버 시작 중... http://{self.host}:{self.port}")
            loop.run_until_complete(server.serve())
        except Exception as e:
            print(f"[ERROR] 서버 실행 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_running = False
            loop.close()
    
    def run(self) -> None:
        """
        서버 동기 실행 (블로킹)
        
        주의: 이 메서드는 블로킹됩니다. 시뮬레이션과 함께 실행하려면
        start_background()를 사용하세요.
        """
        import uvicorn
        
        logger.info(f"제어 서버 시작: http://{self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
    
    def update(self) -> None:
        """
        프레임 업데이트 (시뮬레이션 루프에서 호출)
        
        WebSocket으로 연결된 클라이언트에 새 프레임을 전송합니다.
        """
        if self._is_running:
            self.ws_handler.broadcast_frame()
    
    def stop(self) -> None:
        """서버 중지"""
        self._is_running = False
        logger.info("제어 서버 중지됨")
