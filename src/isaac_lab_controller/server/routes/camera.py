"""
카메라 제어 API 라우트
"""

from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class PoseRequest(BaseModel):
    """카메라 위치/방향 설정 요청"""
    position: List[float]  # [x, y, z]
    orientation: List[float]  # [w, x, y, z]
    convention: str = "ros"


class LookAtRequest(BaseModel):
    """LookAt 설정 요청"""
    eye: List[float]  # [x, y, z]
    target: List[float]  # [x, y, z]


class OrbitRequest(BaseModel):
    """궤도 회전 요청"""
    azimuth: float  # 수평 각도 (도)
    elevation: float  # 수직 각도 (도)
    distance: float  # 거리
    target: Optional[List[float]] = None  # 회전 중심


class SelectCameraRequest(BaseModel):
    """카메라 선택 요청"""
    camera_id: str


@router.get("/list")
async def list_cameras(request: Request):
    """
    사용 가능한 모든 카메라 목록 조회
    
    Returns:
        list: [{"id": str, "name": str, "active": bool}, ...]
    """
    scene = request.app.state.scene
    try:
        cameras = scene.list_cameras()
        active_id = scene.get_active_camera_id()
        
        # 활성 상태 표시
        for cam in cameras:
            cam["active"] = (cam["id"] == active_id)
        
        return {"success": True, "data": cameras}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select")
async def select_camera(request: Request, body: SelectCameraRequest):
    """
    활성 카메라 변경
    
    Args:
        camera_id: 활성화할 카메라 ID
    """
    scene = request.app.state.scene
    try:
        success = scene.set_active_camera(body.camera_id)
        
        if success:
            # app.state.camera도 업데이트
            new_camera = scene.get_camera_adapter()
            request.app.state.camera = new_camera
        
        return {"success": success, "active_camera": body.camera_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active")
async def get_active_camera(request: Request):
    """현재 활성 카메라 정보 조회"""
    scene = request.app.state.scene
    try:
        active_id = scene.get_active_camera_id()
        cameras = scene.list_cameras()
        active_cam = next((c for c in cameras if c["id"] == active_id), None)
        return {"success": True, "data": active_cam}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pose")
async def get_pose(request: Request):
    """
    현재 카메라 위치/방향 조회
    
    Returns:
        dict: position, orientation, eye, target
    """
    camera = request.app.state.camera
    try:
        pose = camera.get_pose()
        return {"success": True, "data": pose}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pose")
async def set_pose(request: Request, body: PoseRequest):
    """
    카메라 위치/방향 설정
    
    Args:
        position: [x, y, z]
        orientation: [w, x, y, z]
        convention: "ros", "opengl", "world"
    """
    camera = request.app.state.camera
    try:
        success = camera.set_pose(
            body.position,
            body.orientation,
            body.convention
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/lookat")
async def set_lookat(request: Request, body: LookAtRequest):
    """
    LookAt 방식으로 카메라 설정
    
    Args:
        eye: [x, y, z] 카메라 위치
        target: [x, y, z] 바라볼 위치
    """
    camera = request.app.state.camera
    try:
        success = camera.set_lookat(body.eye, body.target)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/orbit")
async def orbit(request: Request, body: OrbitRequest):
    """
    타겟 주위 궤도 회전
    
    Args:
        azimuth: 수평 각도 (도)
        elevation: 수직 각도 (도)
        distance: 거리
        target: 회전 중심 (선택)
    """
    camera = request.app.state.camera
    try:
        success = camera.orbit(
            body.azimuth,
            body.elevation,
            body.distance,
            body.target
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intrinsics")
async def get_intrinsics(request: Request):
    """카메라 내부 파라미터 조회"""
    camera = request.app.state.camera
    try:
        intrinsics = camera.get_intrinsics()
        return {"success": True, "data": intrinsics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/frame")
async def get_frame(request: Request, format: str = "jpeg"):
    """
    현재 프레임 캡처 (단일 이미지)
    
    스트리밍은 WebSocket /ws/stream 사용
    """
    from fastapi.responses import Response
    
    camera = request.app.state.camera
    try:
        frame_bytes = camera.get_frame(format=format)
        media_type = "image/jpeg" if format == "jpeg" else "image/png"
        return Response(content=frame_bytes, media_type=media_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
