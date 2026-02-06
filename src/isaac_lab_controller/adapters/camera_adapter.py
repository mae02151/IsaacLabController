"""
CameraAdapter - 카메라 제어 인터페이스

카메라 위치, 방향, 프레임 캡처 등을 제어합니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np


class CameraAdapter(ABC):
    """
    카메라 제어를 위한 추상 인터페이스
    
    카메라 위치/방향 설정, 프레임 캡처 등의 기능을 정의합니다.
    
    Example:
        >>> class MyCameraAdapter(CameraAdapter):
        ...     def __init__(self, camera_sensor):
        ...         self.camera = camera_sensor
        ...
        ...     def set_pose(self, position, orientation):
        ...         self.camera.set_world_poses(
        ...             torch.tensor([position]),
        ...             torch.tensor([orientation])
        ...         )
        ...         return True
    """
    
    @abstractmethod
    def set_pose(
        self, 
        position: List[float], 
        orientation: List[float],
        convention: str = "ros"
    ) -> bool:
        """
        카메라 위치와 방향 설정
        
        Args:
            position: [x, y, z] 위치 (미터 단위)
            orientation: [w, x, y, z] 쿼터니언 회전
            convention: 좌표계 규약 ("ros", "opengl", "world")
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abstractmethod
    def set_lookat(
        self, 
        eye: List[float], 
        target: List[float]
    ) -> bool:
        """
        LookAt 방식으로 카메라 설정
        
        Args:
            eye: [x, y, z] 카메라 위치
            target: [x, y, z] 바라볼 타겟 위치
        
        Returns:
            bool: 성공 여부
        """
        pass
    
    @abstractmethod
    def get_frame(self, format: str = "jpeg") -> bytes:
        """
        현재 카메라 프레임 캡처
        
        Args:
            format: 이미지 포맷 ("jpeg", "png")
        
        Returns:
            bytes: 인코딩된 이미지 바이트 데이터
        """
        pass
    
    @abstractmethod
    def get_pose(self) -> Dict[str, Any]:
        """
        현재 카메라 위치/방향 조회
        
        Returns:
            dict: {
                "position": [x, y, z],
                "orientation": [w, x, y, z],
                "eye": [x, y, z],
                "target": [x, y, z]
            }
        """
        pass
    
    @abstractmethod
    def get_intrinsics(self) -> Dict[str, Any]:
        """
        카메라 내부 파라미터 조회
        
        Returns:
            dict: {
                "width": int,
                "height": int,
                "focal_length": float,
                "horizontal_aperture": float
            }
        """
        pass
    
    def orbit(
        self, 
        azimuth: float, 
        elevation: float, 
        distance: float,
        target: Optional[List[float]] = None
    ) -> bool:
        """
        타겟 주위 궤도 회전 (기본 구현 제공)
        
        Args:
            azimuth: 수평 각도 (도)
            elevation: 수직 각도 (도)
            distance: 타겟까지 거리
            target: 회전 중심 (None이면 현재 타겟 사용)
        
        Returns:
            bool: 성공 여부
        """
        if target is None:
            current = self.get_pose()
            target = current.get("target", [0, 0, 0])
        
        # 구면 좌표계 -> 직교 좌표계 변환
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        x = target[0] + distance * np.cos(el_rad) * np.cos(az_rad)
        y = target[1] + distance * np.cos(el_rad) * np.sin(az_rad)
        z = target[2] + distance * np.sin(el_rad)
        
        return self.set_lookat(eye=[x, y, z], target=target)
