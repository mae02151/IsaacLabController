"""
추상 어댑터 모듈

모든 IsaacLab 환경에서 구현해야 하는 인터페이스를 정의합니다.
"""

from isaac_lab_controller.adapters.base import SceneAdapter
from isaac_lab_controller.adapters.camera_adapter import CameraAdapter
from isaac_lab_controller.adapters.object_adapter import ObjectAdapter
from isaac_lab_controller.adapters.material_adapter import MaterialAdapter
from isaac_lab_controller.adapters.robot_adapter import RobotAdapter

__all__ = [
    "SceneAdapter",
    "CameraAdapter",
    "ObjectAdapter",
    "MaterialAdapter",
    "RobotAdapter",
]
