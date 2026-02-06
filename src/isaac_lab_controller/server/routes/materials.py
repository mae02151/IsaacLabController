"""
재질 관리 API 라우트
"""

from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class CreateMaterialRequest(BaseModel):
    """재질 생성 요청"""
    name: str
    color: List[float]  # [r, g, b] 또는 [r, g, b, a]
    roughness: float = 0.5
    metallic: float = 0.0


class BindMaterialRequest(BaseModel):
    """재질 바인딩 요청"""
    object_id: str
    material_id: str


class UpdateMaterialRequest(BaseModel):
    """재질 업데이트 요청"""
    color: Optional[List[float]] = None
    roughness: Optional[float] = None
    metallic: Optional[float] = None


@router.get("/")
async def list_materials(request: Request):
    """
    모든 재질 목록 조회
    """
    materials = request.app.state.materials
    try:
        mat_list = materials.list_materials()
        return {
            "success": True,
            "data": [
                {
                    "id": mat.id,
                    "name": mat.name,
                    "color": mat.color,
                    "roughness": mat.roughness,
                    "metallic": mat.metallic,
                    "prim_path": mat.prim_path,
                }
                for mat in mat_list
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets")
async def get_preset_materials(request: Request):
    """
    미리 정의된 재질 목록
    """
    materials = request.app.state.materials
    try:
        presets = materials.get_preset_materials()
        return {"success": True, "data": presets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_material(request: Request, body: CreateMaterialRequest):
    """
    새 재질 생성
    
    Args:
        name: 재질 이름
        color: [r, g, b] 색상 (0~1)
        roughness: 거칠기 (0~1)
        metallic: 금속성 (0~1)
    
    Returns:
        생성된 재질 ID
    """
    materials = request.app.state.materials
    try:
        material_id = materials.create(
            name=body.name,
            color=body.color,
            roughness=body.roughness,
            metallic=body.metallic,
        )
        return {"success": True, "material_id": material_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{material_id}")
async def get_material(request: Request, material_id: str):
    """
    특정 재질 정보 조회
    """
    materials = request.app.state.materials
    try:
        mat = materials.get_material(material_id)
        if mat is None:
            raise HTTPException(status_code=404, detail=f"Material not found: {material_id}")
        return {
            "success": True,
            "data": {
                "id": mat.id,
                "name": mat.name,
                "color": mat.color,
                "roughness": mat.roughness,
                "metallic": mat.metallic,
                "prim_path": mat.prim_path,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{material_id}")
async def update_material(request: Request, material_id: str, body: UpdateMaterialRequest):
    """
    재질 속성 업데이트
    """
    materials = request.app.state.materials
    try:
        success = materials.update(
            material_id,
            color=body.color,
            roughness=body.roughness,
            metallic=body.metallic,
        )
        if not success:
            raise HTTPException(status_code=404, detail=f"Material not found: {material_id}")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bind")
async def bind_material(request: Request, body: BindMaterialRequest):
    """
    물체에 재질 적용
    """
    materials = request.app.state.materials
    try:
        success = materials.bind(body.object_id, body.material_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to bind material")
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unbind/{object_id}")
async def unbind_material(request: Request, object_id: str):
    """
    물체에서 재질 제거
    """
    materials = request.app.state.materials
    try:
        success = materials.unbind(object_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
