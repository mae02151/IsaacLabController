"""
SceneAdapter - 시뮬레이션 환경의 기본 인터페이스

모든 IsaacLab 환경은 이 클래스를 상속받아 구현해야 합니다.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from isaac_lab_controller.adapters.camera_adapter import CameraAdapter
    from isaac_lab_controller.adapters.object_adapter import ObjectAdapter
    from isaac_lab_controller.adapters.material_adapter import MaterialAdapter
    from isaac_lab_controller.adapters.robot_adapter import RobotAdapter


class SceneAdapter(ABC):
    """
    모든 IsaacLab 환경이 구현해야 하는 기본 인터페이스
    
    이 클래스를 상속받아 특정 시뮬레이션 환경에 맞는 어댑터를 구현합니다.
    
    Example:
        >>> class MySceneAdapter(SceneAdapter):
        ...     def __init__(self, sim, scene):
        ...         self.sim = sim
        ...         self.scene = scene
        ...         self._camera = MyCameraAdapter(scene["camera"])
        ...
        ...     def step(self):
        ...         self.sim.step()
        ...
        ...     def get_camera_adapter(self):
        ...         return self._camera
    """
    
    @abstractmethod
    def step(self) -> None:
        """
        시뮬레이션 1 스텝 진행
        
        시뮬레이션 루프의 한 프레임을 실행합니다.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        환경 리셋
        
        시뮬레이션 환경을 초기 상태로 되돌립니다.
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        시뮬레이션 실행 상태 확인
        
        Returns:
            bool: 시뮬레이션이 실행 중이면 True
        """
        pass
    
    @abstractmethod
    def get_camera_adapter(self) -> "CameraAdapter":
        """
        현재 활성화된 카메라 어댑터 반환
        
        Returns:
            CameraAdapter: 카메라 제어를 위한 어댑터
        """
        pass
    
    @abstractmethod
    def get_object_adapter(self) -> "ObjectAdapter":
        """
        물체 어댑터 반환
        
        Returns:
            ObjectAdapter: 물체 관리를 위한 어댑터
        """
        pass
    
    @abstractmethod
    def get_material_adapter(self) -> "MaterialAdapter":
        """
        재질 어댑터 반환
        
        Returns:
            MaterialAdapter: 재질 관리를 위한 어댑터
        """
        pass
    
    # 다중 카메라 관리 메서드 (기본 구현 제공)
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        사용 가능한 모든 카메라 목록 반환
        
        Returns:
            list: [{"id": str, "name": str, "active": bool}, ...]
        """
        # 기본 구현: 단일 카메라
        return [{"id": "default", "name": "Main Camera", "active": True}]
    
    def get_active_camera_id(self) -> str:
        """
        현재 활성화된 카메라 ID 반환
        
        Returns:
            str: 활성 카메라 ID
        """
        return "default"
    
    def set_active_camera(self, camera_id: str) -> bool:
        """
        활성 카메라 변경
        
        Args:
            camera_id: 활성화할 카메라 ID
        
        Returns:
            bool: 성공 여부
        """
        # 기본 구현: 단일 카메라만 지원
        return camera_id == "default"
    
    def get_camera_by_id(self, camera_id: str) -> Optional["CameraAdapter"]:
        """
        특정 ID의 카메라 어댑터 반환

        Args:
            camera_id: 카메라 ID

        Returns:
            CameraAdapter or None: 해당 카메라 어댑터
        """
        if camera_id == "default":
            return self.get_camera_adapter()
        return None

    def get_robot_adapter(self) -> Optional["RobotAdapter"]:
        """
        로봇 어댑터 반환

        Returns:
            RobotAdapter or None: 로봇 제어를 위한 어댑터 (없으면 None)
        """
        return None

