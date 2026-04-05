"""
ObjectAdapter - 물체 관리 인터페이스

시뮬레이션 내 물체 생성, 삭제, 조회를 담당합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """물체 정보 데이터 클래스"""
    id: str
    name: str
    obj_type: str
    position: List[float]
    rotation: List[float]
    prim_path: str


class ObjectAdapter(ABC):
    """
    물체 관리를 위한 추상 인터페이스
    
    시뮬레이션 내에서 물체를 동적으로 생성/삭제하고 관리합니다.
    
    Example:
        >>> class MyObjectAdapter(ObjectAdapter):
        ...     def spawn(self, obj_type, position, rotation, **kwargs):
        ...         prim_path = f"/World/Objects/obj_{self._counter}"
        ...         sim_utils.spawn_cuboid(prim_path, cfg)
        ...         return prim_path
    """
    
    @abstractmethod
    def spawn(
        self, 
        obj_type: str, 
        position: List[float], 
        rotation: List[float],
        name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        물체 생성
        
        Args:
            obj_type: 물체 유형 ("box", "sphere", "cylinder", 또는 USD 경로)
            position: [x, y, z] 위치
            rotation: [w, x, y, z] 쿼터니언 회전
            name: 물체 이름 (선택사항)
            **kwargs: 추가 파라미터 (크기, 색상 등)
        
        Returns:
            str: 생성된 물체의 고유 ID
        """
        pass
    
    @abstractmethod
    def delete(self, object_id: str) -> bool:
        """
        물체 삭제
        
        Args:
            object_id: 삭제할 물체 ID
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abstractmethod
    def get_object(self, object_id: str) -> Optional[ObjectInfo]:
        """
        특정 물체 정보 조회
        
        Args:
            object_id: 조회할 물체 ID
        
        Returns:
            ObjectInfo 또는 None
        """
        pass
    
    @abstractmethod
    def list_objects(self) -> List[ObjectInfo]:
        """
        모든 관리 중인 물체 목록 조회
        
        Returns:
            List[ObjectInfo]: 물체 정보 리스트
        """
        pass
    
    @abstractmethod
    def get_available_types(self) -> List[Dict[str, Any]]:
        """
        생성 가능한 물체 유형 목록
        
        Returns:
            List[dict]: [
                {"type": "box", "name": "박스", "description": "..."},
                {"type": "sphere", "name": "구", "description": "..."},
                ...
            ]
        """
        pass
    
    @abstractmethod
    def set_pose(
        self, 
        object_id: str, 
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None
    ) -> bool:
        """
        물체 위치/회전 변경
        
        Args:
            object_id: 물체 ID
            position: 새 위치 (None이면 변경 안함)
            rotation: 새 회전 (None이면 변경 안함)
        
        Returns:
            bool: 성공 여부
        """
        pass
