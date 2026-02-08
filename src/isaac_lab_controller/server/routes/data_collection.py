"""
데이터 수집 제어 API 라우트

웹 UI 버튼으로 에피소드 녹화/정지/폐기를 제어합니다.

엔드포인트:
    GET  /status  - 수집 상태 조회
    POST /start   - 에피소드 녹화 시작
    POST /stop    - 에피소드 저장 → 다음 에피소드 자동 시작
    POST /discard - 현재 에피소드 폐기 → 새 에피소드 시작
    POST /finish  - 수집 완료 (HDF5 파일 닫기)
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class StopRequest(BaseModel):
    """에피소드 종료 요청"""
    success: bool = True


@router.get("/status")
async def get_status(request: Request):
    """
    데이터 수집 상태 조회
    """
    collector = getattr(request.app.state, "data_collector", None)
    if collector is None:
        raise HTTPException(status_code=404, detail="DataCollector가 등록되지 않았습니다")

    return {
        "success": True,
        "data": {
            "is_recording": collector.is_recording,
            "episode_index": collector.episode_index,
            "frame_count": collector.frame_count,
            "hdf5_path": getattr(collector, "_hdf5_path", ""),
        },
    }


@router.post("/start")
async def start_episode(request: Request):
    """
    에피소드 녹화 시작

    이미 녹화 중이면 무시합니다.
    """
    collector = getattr(request.app.state, "data_collector", None)
    if collector is None:
        raise HTTPException(status_code=404, detail="DataCollector가 등록되지 않았습니다")

    if collector.is_recording:
        return {
            "success": False,
            "message": "이미 녹화 중입니다",
            "episode_index": collector.episode_index,
        }

    collector.start_episode()
    return {
        "success": True,
        "message": f"에피소드 {collector.episode_index} 녹화 시작",
        "episode_index": collector.episode_index,
    }


@router.post("/stop")
async def stop_episode(request: Request, body: StopRequest = StopRequest()):
    """
    현재 에피소드를 저장하고 다음 에피소드를 자동 시작합니다.
    """
    collector = getattr(request.app.state, "data_collector", None)
    if collector is None:
        raise HTTPException(status_code=404, detail="DataCollector가 등록되지 않았습니다")

    if not collector.is_recording:
        return {
            "success": False,
            "message": "녹화 중이 아닙니다",
        }

    saved_episode = collector.episode_index
    frame_count = collector.frame_count
    saved = collector.end_episode(success=body.success)

    if saved:
        collector.start_episode()
        return {
            "success": True,
            "message": f"에피소드 {saved_episode} 저장 완료 ({frame_count} 프레임)",
            "saved_episode": saved_episode,
            "saved_frames": frame_count,
            "next_episode": collector.episode_index,
        }
    else:
        collector.start_episode()
        return {
            "success": False,
            "message": f"에피소드 {saved_episode} 저장 실패 (데이터 없음)",
            "next_episode": collector.episode_index,
        }


@router.post("/discard")
async def discard_episode(request: Request):
    """
    현재 에피소드를 폐기하고 새 에피소드를 시작합니다.
    """
    collector = getattr(request.app.state, "data_collector", None)
    if collector is None:
        raise HTTPException(status_code=404, detail="DataCollector가 등록되지 않았습니다")

    discarded_episode = collector.episode_index
    frame_count = collector.frame_count
    collector.discard_episode()
    collector.start_episode()

    return {
        "success": True,
        "message": f"에피소드 {discarded_episode} 폐기 ({frame_count} 프레임)",
        "discarded_episode": discarded_episode,
        "next_episode": collector.episode_index,
    }


@router.post("/finish")
async def finish_collection(request: Request):
    """
    데이터 수집을 완전히 종료합니다.

    녹화 중인 에피소드가 있으면 먼저 저장한 후 HDF5 파일을 닫습니다.
    """
    collector = getattr(request.app.state, "data_collector", None)
    if collector is None:
        raise HTTPException(status_code=404, detail="DataCollector가 등록되지 않았습니다")

    # 녹화 중이면 먼저 저장
    if collector.is_recording and collector.frame_count > 0:
        collector.end_episode(success=True)

    collector.close()

    return {
        "success": True,
        "message": "데이터 수집 완료",
        "total_episodes": collector.episode_index,
        "hdf5_path": getattr(collector, "_hdf5_path", ""),
    }
