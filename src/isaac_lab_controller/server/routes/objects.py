"""
물체 관리 API 라우트
"""

from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SpawnRequest(BaseModel):
    """물체 생성 요청"""
    obj_type: str  # "box", "sphere", "cylinder" 또는 USD 경로
    position: List[float] = [0.0, 0.0, 0.0]  # [x, y, z]
    rotation: List[float] = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
    name: Optional[str] = None
    scale: Optional[List[float]] = None  # [sx, sy, sz]
    color: Optional[List[float]] = None  # [r, g, b]


class PoseUpdateRequest(BaseModel):
    """물체 위치 업데이트 요청"""
    position: Optional[List[float]] = None
    rotation: Optional[List[float]] = None


@router.get("/")
async def list_objects(request: Request):
    """
    현재 관리 중인 모든 물체 목록 조회
    """
    objects = request.app.state.objects
    try:
        obj_list = objects.list_objects()
        result = []
        for obj in obj_list:
            # 딕셔너리와 객체 모두 처리
            if isinstance(obj, dict):
                result.append({
                    "id": obj.get("id", ""),
                    "name": obj.get("name", ""),
                    "type": obj.get("obj_type", ""),
                    "position": obj.get("position", []),
                    "rotation": obj.get("rotation", []),
                    "prim_path": obj.get("prim_path", ""),
                })
            else:
                result.append({
                    "id": obj.id,
                    "name": obj.name,
                    "type": obj.obj_type,
                    "position": obj.position,
                    "rotation": obj.rotation,
                    "prim_path": obj.prim_path,
                })
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_available_types(request: Request):
    """
    생성 가능한 물체 유형 목록
    """
    objects = request.app.state.objects
    try:
        types = objects.get_available_types()
        return {"success": True, "data": types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spawn")
async def spawn_object(request: Request, body: SpawnRequest):
    """
    새 물체 생성
    
    Args:
        obj_type: 물체 유형
        position: 위치
        rotation: 회전
        name: 이름 (선택)
        scale: 크기 (선택)
        color: 색상 (선택)
    
    Returns:
        생성된 물체 ID
    """
    objects = request.app.state.objects
    try:
        kwargs = {}
        if body.scale:
            kwargs["scale"] = body.scale
        if body.color:
            kwargs["color"] = body.color
        
        object_id = objects.spawn(
            obj_type=body.obj_type,
            position=body.position,
            rotation=body.rotation,
            name=body.name,
            **kwargs
        )
        return {"success": True, "object_id": object_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{object_id}")
async def get_object(request: Request, object_id: str):
    """
    특정 물체 정보 조회
    """
    objects = request.app.state.objects
    try:
        obj = objects.get_object(object_id)
        if obj is None:
            raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        return {
            "success": True,
            "data": {
                "id": obj.id,
                "name": obj.name,
                "type": obj.obj_type,
                "position": obj.position,
                "rotation": obj.rotation,
                "prim_path": obj.prim_path,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{object_id}")
async def delete_object(request: Request, object_id: str):
    """
    물체 삭제
    """
    objects = request.app.state.objects
    try:
        success = objects.delete(object_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{object_id}/pose")
async def update_object_pose(request: Request, object_id: str, body: PoseUpdateRequest):
    """
    물체 위치/회전 업데이트
    """
    objects = request.app.state.objects
    try:
        success = objects.set_pose(
            object_id,
            position=body.position,
            rotation=body.rotation
        )
        if not success:
            raise HTTPException(status_code=404, detail=f"Object not found: {object_id}")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
