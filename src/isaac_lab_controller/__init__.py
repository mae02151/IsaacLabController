"""
IsaacLabController - 재사용 가능한 IsaacLab 프론트엔드 제어 시스템

이 패키지는 웹 브라우저에서 IsaacLab 시뮬레이션을 실시간으로 제어할 수 있게 해줍니다.

사용법:
    from isaac_lab_controller import ControlServer
    from isaac_lab_controller.adapters import SceneAdapter

    class MySceneAdapter(SceneAdapter):
        ...

    server = ControlServer(MySceneAdapter(sim, scene))
    server.run()
"""

from isaac_lab_controller.adapters import (
    SceneAdapter,
    CameraAdapter,
    ObjectAdapter,
    MaterialAdapter,
)
from isaac_lab_controller.server import ControlServer

__version__ = "0.1.0"
__all__ = [
    "ControlServer",
    "SceneAdapter",
    "CameraAdapter",
    "ObjectAdapter",
    "MaterialAdapter",
]
