"""
MaterialAdapter - 재질 관리 인터페이스

재질 생성, 바인딩, 조회를 담당합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class MaterialInfo:
    """재질 정보 데이터 클래스"""
    id: str
    name: str
    color: List[float]  # [r, g, b, a]
    roughness: float
    metallic: float
    prim_path: str


class MaterialAdapter(ABC):
    """
    재질 관리를 위한 추상 인터페이스
    
    시뮬레이션 내에서 재질을 생성하고 물체에 적용합니다.
    
    Example:
        >>> class MyMaterialAdapter(MaterialAdapter):
        ...     def create(self, name, color, roughness, metallic):
        ...         mat_path = f"/World/Materials/{name}"
        ...         spawn_preview_surface(mat_path, cfg)
        ...         return mat_path
    """
    
    @abstractmethod
    def create(
        self, 
        name: str, 
        color: List[float],
        roughness: float = 0.5,
        metallic: float = 0.0,
        **kwargs
    ) -> str:
        """
        새 재질 생성
        
        Args:
            name: 재질 이름
            color: [r, g, b] 또는 [r, g, b, a] 색상 (0~1)
            roughness: 거칠기 (0=매끄러움, 1=거침)
            metallic: 금속성 (0=비금속, 1=금속)
            **kwargs: 추가 파라미터
        
        Returns:
            str: 생성된 재질 ID
        """
        pass
    
    @abstractmethod
    def bind(self, object_id: str, material_id: str) -> bool:
        """
        물체에 재질 적용
        
        Args:
            object_id: 물체 ID
            material_id: 재질 ID
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abstractmethod
    def unbind(self, object_id: str) -> bool:
        """
        물체에서 재질 제거
        
        Args:
            object_id: 물체 ID
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abstractmethod
    def get_material(self, material_id: str) -> Optional[MaterialInfo]:
        """
        특정 재질 정보 조회
        
        Args:
            material_id: 조회할 재질 ID
        
        Returns:
            MaterialInfo 또는 None
        """
        pass
    
    @abstractmethod
    def list_materials(self) -> List[MaterialInfo]:
        """
        모든 재질 목록 조회
        
        Returns:
            List[MaterialInfo]: 재질 정보 리스트
        """
        pass
    
    @abstractmethod
    def update(
        self,
        material_id: str,
        color: Optional[List[float]] = None,
        roughness: Optional[float] = None,
        metallic: Optional[float] = None
    ) -> bool:
        """
        재질 속성 업데이트
        
        Args:
            material_id: 재질 ID
            color: 새 색상 (None이면 변경 안함)
            roughness: 새 거칠기 (None이면 변경 안함)
            metallic: 새 금속성 (None이면 변경 안함)
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    def get_preset_materials(self) -> List[Dict[str, Any]]:
        """
        미리 정의된 재질 목록 (기본 구현 제공)
        
        Returns:
            List[dict]: 프리셋 재질 목록
        """
        return [
            {"name": "red_plastic", "color": [1.0, 0.2, 0.2], "roughness": 0.3, "metallic": 0.0},
            {"name": "blue_plastic", "color": [0.2, 0.2, 1.0], "roughness": 0.3, "metallic": 0.0},
            {"name": "green_plastic", "color": [0.2, 1.0, 0.2], "roughness": 0.3, "metallic": 0.0},
            {"name": "metal_silver", "color": [0.8, 0.8, 0.8], "roughness": 0.2, "metallic": 0.9},
            {"name": "metal_gold", "color": [1.0, 0.84, 0.0], "roughness": 0.2, "metallic": 0.9},
            {"name": "wood", "color": [0.55, 0.27, 0.07], "roughness": 0.7, "metallic": 0.0},
            {"name": "glass", "color": [0.9, 0.9, 0.95], "roughness": 0.1, "metallic": 0.0},
        ]
