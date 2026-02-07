"""
독립 실행 FastAPI 서버 (서버 분리 모드)

IsaacLab과 별도 프로세스로 실행되며, ZMQ를 통해 통신합니다.
코드 수정 후 이 스크립트만 재시작하면 됩니다.

사용법:
    # 터미널 1: 시뮬레이션 (한 번만 실행, 계속 켜둠)
    c:\\Users\\dongu\\IsaacLab\\isaaclab.bat -p run_sim_only.py
    
    # 터미널 2: 서버 (수정 후 재시작)
    python server_standalone.py
    
    # 브라우저에서 http://localhost:8000 접속
"""

import sys
import os
import uvicorn
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# 패키지 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from isaac_lab_controller.ipc import ServerBridge
from isaac_lab_controller.server.app import create_app


# =====================================================
# 프록시 어댑터 클래스들
# ZMQ를 통해 시뮬레이션의 실제 어댑터 메서드를 호출
# =====================================================

@dataclass
class ObjectInfo:
    """물체 정보"""
    id: str
    name: str
    obj_type: str
    position: List[float]
    rotation: List[float]
    prim_path: str = ""


@dataclass
class MaterialInfo:
    """재질 정보"""
    id: str
    name: str
    color: List[float]
    roughness: float = 0.5
    metallic: float = 0.0
    prim_path: str = ""


class ProxyCameraAdapter:
    """카메라 어댑터 프록시 - ZMQ로 시뮬레이션과 통신"""
    
    def __init__(self, bridge: ServerBridge):
        self.bridge = bridge
    
    def get_frame(self, format: str = "jpeg") -> bytes:
        response = self.bridge.call("camera", "get_frame", kwargs={"format": format})
        if response.success and response.data:
            return response.data
        return b""
    
    def set_lookat(self, eye: List[float], target: List[float]) -> bool:
        response = self.bridge.call("camera", "set_lookat", args=[eye, target])
        return response.success
    
    def set_pose(self, position: List[float], orientation: List[float], convention: str = "ros") -> bool:
        response = self.bridge.call("camera", "set_pose", args=[position, orientation, convention])
        return response.success
    
    def get_pose(self) -> Dict[str, Any]:
        response = self.bridge.call("camera", "get_pose")
        return response.data if response.success else {}
    
    def get_intrinsics(self) -> Dict[str, Any]:
        response = self.bridge.call("camera", "get_intrinsics")
        return response.data if response.success else {}
    
    def orbit(self, azimuth: float, elevation: float, distance: float, 
              target: Optional[List[float]] = None) -> bool:
        """궤도 카메라 제어 - azimuth/elevation/distance로 위치 계산"""
        response = self.bridge.call("camera", "orbit", 
                                   args=[azimuth, elevation, distance],
                                   kwargs={"target": target})
        return response.success


class ProxyObjectAdapter:
    """물체 어댑터 프록시 - ZMQ로 시뮬레이션과 통신"""
    
    def __init__(self, bridge: ServerBridge):
        self.bridge = bridge
    
    def spawn(self, obj_type: str, position: List[float], rotation: List[float], 
              name: Optional[str] = None, **kwargs) -> str:
        response = self.bridge.call("object", "spawn", 
                                   args=[obj_type, position, rotation],
                                   kwargs={"name": name, **kwargs})
        return response.data if response.success else ""
    
    def delete(self, object_id: str) -> bool:
        response = self.bridge.call("object", "delete", args=[object_id])
        return response.success and response.data
    
    def get_object(self, object_id: str) -> Optional[ObjectInfo]:
        response = self.bridge.call("object", "get_object", args=[object_id])
        if response.success and response.data:
            return ObjectInfo(**response.data) if isinstance(response.data, dict) else response.data
        return None
    
    def list_objects(self) -> List[ObjectInfo]:
        response = self.bridge.call("object", "list_objects")
        if response.success and response.data:
            return [ObjectInfo(**o) if isinstance(o, dict) else o for o in response.data]
        return []
    
    def get_available_types(self) -> List[Dict[str, Any]]:
        response = self.bridge.call("object", "get_available_types")
        return response.data if response.success else []
    
    def set_pose(self, object_id: str, position: Optional[List[float]] = None,
                rotation: Optional[List[float]] = None) -> bool:
        response = self.bridge.call("object", "set_pose", 
                                   args=[object_id],
                                   kwargs={"position": position, "rotation": rotation})
        return response.success and response.data
    
    def transform(self, object_id: str, target_type: str = "random_box") -> bool:
        """물체를 다른 물체로 변환"""
        response = self.bridge.call("object", "transform", 
                                   args=[object_id, target_type])
        return response.success


class ProxyMaterialAdapter:
    """재질 어댑터 프록시 - ZMQ로 시뮬레이션과 통신"""
    
    def __init__(self, bridge: ServerBridge):
        self.bridge = bridge
    
    def create(self, name: str, color: List[float], 
               roughness: float = 0.5, metallic: float = 0.0, **kwargs) -> str:
        response = self.bridge.call("material", "create",
                                   args=[name, color],
                                   kwargs={"roughness": roughness, "metallic": metallic, **kwargs})
        return response.data if response.success else ""
    
    def bind(self, object_id: str, material_id: str) -> bool:
        response = self.bridge.call("material", "bind", args=[object_id, material_id])
        return response.success
    
    def unbind(self, object_id: str) -> bool:
        response = self.bridge.call("material", "unbind", args=[object_id])
        return response.success and response.data
    
    def get_material(self, material_id: str) -> Optional[MaterialInfo]:
        response = self.bridge.call("material", "get_material", args=[material_id])
        if response.success and response.data:
            return MaterialInfo(**response.data) if isinstance(response.data, dict) else response.data
        return None
    
    def list_materials(self) -> List[MaterialInfo]:
        response = self.bridge.call("material", "list_materials")
        if response.success and response.data:
            return [MaterialInfo(**m) if isinstance(m, dict) else m for m in response.data]
        return []
    
    def get_preset_materials(self) -> List[Dict[str, Any]]:
        response = self.bridge.call("material", "get_preset_materials")
        return response.data if response.success else []
    
    def update(self, material_id: str, color: Optional[List[float]] = None,
              roughness: Optional[float] = None, metallic: Optional[float] = None) -> bool:
        response = self.bridge.call("material", "update",
                                   args=[material_id],
                                   kwargs={"color": color, "roughness": roughness, "metallic": metallic})
        return response.success and response.data


class ProxySceneAdapter:
    """씬 어댑터 프록시 - ZMQ로 시뮬레이션과 통신"""
    
    def __init__(self, bridge: ServerBridge):
        self.bridge = bridge
        self.camera = ProxyCameraAdapter(bridge)
        self.objects = ProxyObjectAdapter(bridge)
        self.materials = ProxyMaterialAdapter(bridge)
    
    def get_camera_adapter(self):
        return self.camera
    
    def get_object_adapter(self):
        return self.objects
    
    def get_material_adapter(self):
        return self.materials
    
    def is_running(self) -> bool:
        response = self.bridge.call("scene", "is_running")
        return response.success and response.data
    
    # 카메라 관련 메서드 위임
    def list_cameras(self) -> List[Dict[str, Any]]:
        response = self.bridge.call("scene", "list_cameras")
        return response.data if response.success else []
    
    def set_active_camera(self, camera_id: str) -> bool:
        response = self.bridge.call("scene", "set_active_camera", args=[camera_id])
        return response.success and response.data
    
    def get_active_camera_id(self) -> str:
        response = self.bridge.call("scene", "get_active_camera_id")
        return response.data if response.success else ""


def main():
    print("=" * 60)
    print("[SERVER] FastAPI 서버 시작 (서버 분리 모드)")
    print("=" * 60)
    
    # ZMQ 브릿지 연결
    zmq_url = "tcp://localhost:5555"
    print(f"[SERVER] ZMQ 연결 시도: {zmq_url}")
    bridge = ServerBridge(zmq_url=zmq_url, timeout_ms=5000)
    
    # 연결 확인
    if not bridge.is_connected():
        print("[SERVER] ⚠️ 시뮬레이션에 연결할 수 없습니다.")
        print("[SERVER] 먼저 run_sim_only.py를 실행하세요!")
        print("=" * 60)
        # 연결 안되어도 서버 시작 (나중에 연결될 수 있음)
    else:
        print("[SERVER] ✅ 시뮬레이션 연결 성공!")
    
    # 프록시 어댑터 생성
    scene_adapter = ProxySceneAdapter(bridge)
    
    # FastAPI 앱 생성
    app = create_app(scene_adapter)
    
    print("")
    print("=" * 60)
    print("[SERVER] 🌐 웹 서버: http://localhost:8000")
    print("[SERVER] 📝 코드 수정 후 Ctrl+C로 종료하고 다시 실행하세요")
    print("=" * 60)
    print("")
    
    # 서버 실행
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        print("\n[SERVER] 서버 종료")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
