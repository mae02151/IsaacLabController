"""
Digital Twin 예제

기존 make_synthetic_data.py를 IsaacLabController와 통합하는 예제입니다.
"""

from isaac_lab_controller.examples.digital_twin.adapters import (
    DigitalTwinSceneAdapter,
    DigitalTwinCameraAdapter,
    DigitalTwinObjectAdapter,
    DigitalTwinMaterialAdapter,
)

__all__ = [
    "DigitalTwinSceneAdapter",
    "DigitalTwinCameraAdapter",
    "DigitalTwinObjectAdapter",
    "DigitalTwinMaterialAdapter",
]
