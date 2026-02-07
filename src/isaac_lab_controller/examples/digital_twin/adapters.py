"""
Digital Twin 예제용 어댑터 구현

이 모듈은 IsaacLabController를 기존 IsaacLab 시뮬레이션에
연동하는 구체적인 어댑터 구현을 제공합니다.
"""

from typing import Dict, Any, List, Optional
import uuid
import cv2
import numpy as np
import torch
import threading

from isaac_lab_controller.adapters import (
    SceneAdapter,
    CameraAdapter,
    ObjectAdapter,
    MaterialAdapter,
)
from isaac_lab_controller.adapters.object_adapter import ObjectInfo
from isaac_lab_controller.adapters.material_adapter import MaterialInfo


class DigitalTwinCameraAdapter(CameraAdapter):
    """
    IsaacLab Camera 센서를 감싸는 어댑터
    
    스레드 안전한 프레임 버퍼링과 명령 큐를 사용합니다.
    - 메인 스레드: process_commands() + update_frame() 호출
    - 서버 스레드: set_lookat/orbit 등이 명령 큐에 추가, get_frame()으로 캐시된 프레임 읽기
    
    Args:
        camera: IsaacLab CameraCfg로 생성된 카메라 센서
    """
    
    def __init__(self, camera, device: str = "cuda:0"):
        self.camera = camera
        self.device = device
        self._current_eye = [1.5, 0.0, 0.6]
        self._current_target = [0.0, 0.0, 0.0]
        
        # 스레드 안전한 프레임 버퍼
        self._frame_lock = threading.Lock()
        self._cached_frame: Optional[bytes] = None
        self._frame_ready = False
        
        # 명령 큐 (스레드 안전)
        from queue import Queue
        self._command_queue: Queue = Queue()
    
    def set_pose(
        self, 
        position: List[float], 
        orientation: List[float],
        convention: str = "ros"
    ) -> bool:
        """카메라 위치/방향 설정 (큐에 추가)"""
        self._command_queue.put(('set_pose', position, orientation, convention))
        return True
    
    def set_lookat(self, eye: List[float], target: List[float]) -> bool:
        """LookAt 설정 (큐에 추가)"""
        self._command_queue.put(('set_lookat', eye, target))
        return True
    
    def _execute_set_pose(self, position: List[float], orientation: List[float], convention: str) -> bool:
        """실제 set_pose 실행 (메인 스레드에서 호출)"""
        try:
            pos_tensor = torch.tensor([position], device=self.device, dtype=torch.float32)
            ori_tensor = torch.tensor([orientation], device=self.device, dtype=torch.float32)
            self.camera.set_world_poses(pos_tensor, ori_tensor, convention=convention)
            self._current_eye = position
            return True
        except Exception as e:
            print(f"카메라 위치 설정 오류: {e}")
            return False
    
    def _execute_set_lookat(self, eye: List[float], target: List[float]) -> bool:
        """실제 set_lookat 실행 (메인 스레드에서 호출)"""
        try:
            eyes = torch.tensor([eye], device=self.device, dtype=torch.float32)
            targets = torch.tensor([target], device=self.device, dtype=torch.float32)
            self.camera.set_world_poses_from_view(eyes, targets)
            self._current_eye = eye
            self._current_target = target
            print(f"[DEBUG] 카메라 이동: eye={eye}, target={target}")
            return True
        except Exception as e:
            print(f"LookAt 설정 오류: {e}")
            return False
    
    def process_commands(self) -> None:
        """
        메인 스레드에서 호출: 큐에 쌓인 명령 처리
        
        시뮬레이션 루프에서 매 프레임마다 호출해야 합니다.
        """
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == 'set_lookat':
                    self._execute_set_lookat(cmd[1], cmd[2])
                elif cmd[0] == 'set_pose':
                    self._execute_set_pose(cmd[1], cmd[2], cmd[3])
            except Exception as e:
                print(f"명령 처리 오류: {e}")
    
    def update_frame(self) -> None:
        """
        메인 스레드에서 호출: 카메라 프레임 캡처 및 버퍼 저장
        
        이 메서드는 반드시 시뮬레이션 메인 스레드에서 호출해야 합니다.
        """
        try:
            # 카메라 데이터 가져오기 (CUDA 텐서)
            rgb_tensor = self.camera.data.output["rgb"][0]  # (H, W, 4)
            rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)
            
            # RGBA -> BGR 변환
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2BGR)
            
            # JPEG 인코딩
            _, buffer = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            # 스레드 안전하게 버퍼 저장
            with self._frame_lock:
                self._cached_frame = frame_bytes
                self._frame_ready = True
                
        except Exception as e:
            print(f"프레임 업데이트 오류: {e}")
    
    def get_frame(self, format: str = "jpeg") -> bytes:
        """
        서버 스레드에서 호출: 캐시된 프레임 반환 (스레드 안전)
        """
        with self._frame_lock:
            if self._cached_frame is not None:
                return self._cached_frame
        
        # 프레임이 없으면 빈 이미지 반환
        empty = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', empty)
        return buffer.tobytes()
    
    def get_pose(self) -> Dict[str, Any]:
        return {
            "eye": self._current_eye,
            "target": self._current_target,
            "position": self._current_eye,
            "orientation": [1.0, 0.0, 0.0, 0.0],  # 기본값
        }
    
    def get_intrinsics(self) -> Dict[str, Any]:
        return {
            "width": 640,
            "height": 480,
            "focal_length": 24.0,
            "horizontal_aperture": 20.955,
        }


class DigitalTwinObjectAdapter(ObjectAdapter):
    """
    IsaacLab Scene에서 물체를 관리하는 어댑터
    
    스레드 안전한 명령 큐를 사용합니다.
    - 서버 스레드: spawn/delete 명령을 큐에 추가
    - 메인 스레드: process_commands()로 실제 USD 프리미티브 생성/삭제
    """
    
    def __init__(self, scene, sim_utils_module, device: str = "cuda:0"):
        self.scene = scene
        self.sim_utils = sim_utils_module
        self.device = device
        self._objects: Dict[str, ObjectInfo] = {}
        self._counter = 0
        
        # 명령 큐 (스레드 안전)
        from queue import Queue
        self._command_queue: Queue = Queue()
        self._pending_results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()
    
    def spawn(
        self, 
        obj_type: str, 
        position: List[float], 
        rotation: List[float],
        name: Optional[str] = None,
        **kwargs
    ) -> str:
        """물체 생성 요청 (큐에 추가)"""
        object_id = str(uuid.uuid4())[:8]
        self._command_queue.put(('spawn', object_id, obj_type, position, rotation, name, kwargs))
        print(f"[DEBUG] 물체 생성 큐에 추가: {object_id} ({obj_type})")
        return object_id
    
    def delete(self, object_id: str) -> bool:
        """물체 삭제 요청 (큐에 추가)"""
        if object_id not in self._objects:
            return False
        self._command_queue.put(('delete', object_id))
        print(f"[DEBUG] 물체 삭제 큐에 추가: {object_id}")
        return True
    
    def process_commands(self) -> None:
        """
        메인 스레드에서 호출: 큐에 쌓인 물체 생성/삭제 처리
        
        시뮬레이션 루프에서 매 프레임마다 호출해야 합니다.
        """
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == 'spawn':
                    _, object_id, obj_type, position, rotation, name, kwargs = cmd
                    self._execute_spawn(object_id, obj_type, position, rotation, name, **kwargs)
                elif cmd[0] == 'delete':
                    self._execute_delete(cmd[1])
            except Exception as e:
                print(f"물체 명령 처리 오류: {e}")
    
    def _execute_spawn(
        self, 
        object_id: str,
        obj_type: str, 
        position: List[float], 
        rotation: List[float],
        name: Optional[str] = None,
        **kwargs
    ) -> None:
        """실제 물체 생성 (메인 스레드에서 실행)"""
        try:
            prim_path = f"/World/DynamicObjects/obj_{self._counter}"
            self._counter += 1
            
            # 물체 유형에 따른 설정
            if obj_type == "box":
                cfg = self.sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=self.sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.8, 0.2, 0.2))
                    ),
                )
                self.sim_utils.spawn_cuboid(prim_path, cfg)
                
            elif obj_type == "sphere":
                cfg = self.sim_utils.SphereCfg(
                    radius=0.05,
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=self.sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.2, 0.8, 0.2))
                    ),
                )
                self.sim_utils.spawn_sphere(prim_path, cfg)
                
            elif obj_type == "cylinder":
                cfg = self.sim_utils.CylinderCfg(
                    radius=0.05,
                    height=0.1,
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=self.sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.2, 0.2, 0.8))
                    ),
                )
                self.sim_utils.spawn_cylinder(prim_path, cfg)
            else:
                print(f"지원하지 않는 물체 유형: {obj_type}")
                return
            
            # 물체 정보 저장
            obj_info = ObjectInfo(
                id=object_id,
                name=name or f"{obj_type}_{self._counter}",
                obj_type=obj_type,
                position=position,
                rotation=rotation,
                prim_path=prim_path,
            )
            self._objects[object_id] = obj_info
            print(f"[DEBUG] 물체 생성 완료: {object_id} at {prim_path}")
            
        except Exception as e:
            print(f"물체 생성 오류: {e}")
    
    def _execute_delete(self, object_id: str) -> None:
        """실제 물체 삭제 (메인 스레드에서 실행)"""
        if object_id not in self._objects:
            return
        
        try:
            from pxr import Usd
            import omni.usd
            
            obj = self._objects[object_id]
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obj.prim_path)
            if prim.IsValid():
                stage.RemovePrim(obj.prim_path)
            
            del self._objects[object_id]
            print(f"[DEBUG] 물체 삭제 완료: {object_id}")
        except Exception as e:
            print(f"물체 삭제 오류: {e}")
    
    def get_object(self, object_id: str) -> Optional[ObjectInfo]:
        return self._objects.get(object_id)
    
    def list_objects(self) -> List[ObjectInfo]:
        return list(self._objects.values())
    
    def get_available_types(self) -> List[Dict[str, Any]]:
        return [
            {"type": "box", "name": "박스", "description": "정육면체"},
            {"type": "sphere", "name": "구", "description": "구체"},
            {"type": "cylinder", "name": "원통", "description": "원기둥"},
            {"type": "cone", "name": "원뿔", "description": "원뿔"},
        ]
    
    def set_pose(
        self, 
        object_id: str, 
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None
    ) -> bool:
        if object_id not in self._objects:
            return False
        
        try:
            from pxr import UsdGeom
            import omni.usd
            
            obj = self._objects[object_id]
            stage = omni.usd.get_context().get_stage()
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(obj.prim_path))
            
            if position:
                obj.position = position
                # USD API로 위치 설정
            if rotation:
                obj.rotation = rotation
                
            return True
        except Exception as e:
            print(f"위치 설정 오류: {e}")
            return False


class DigitalTwinMaterialAdapter(MaterialAdapter):
    """
    IsaacLab 재질 관리 어댑터
    
    스레드 안전한 명령 큐를 사용합니다.
    - 서버 스레드: create/bind 명령을 큐에 추가
    - 메인 스레드: process_commands()로 실제 USD 프리미티브 생성
    """
    
    def __init__(self, sim_utils_module):
        self.sim_utils = sim_utils_module
        self._materials: Dict[str, MaterialInfo] = {}
        self._counter = 0
        
        # 명령 큐 (스레드 안전)
        from queue import Queue
        self._command_queue: Queue = Queue()
    
    def create(
        self, 
        name: str, 
        color: List[float],
        roughness: float = 0.5,
        metallic: float = 0.0,
        **kwargs
    ) -> str:
        """재질 생성 요청 (큐에 추가, 즉시 목록에 표시)"""
        material_id = str(uuid.uuid4())[:8]
        prim_path = f"/World/Materials/mat_{self._counter}"
        self._counter += 1
        
        # 색상 정규화
        if len(color) == 3:
            color = color + [1.0]
        
        # 즉시 MaterialInfo 저장 (목록 조회에 바로 보이게)
        mat_info = MaterialInfo(
            id=material_id,
            name=name,
            color=color,
            roughness=roughness,
            metallic=metallic,
            prim_path=prim_path,
        )
        self._materials[material_id] = mat_info
        
        # 실제 USD 생성은 메인 스레드에서 처리
        self._command_queue.put(('create', material_id, name, color, roughness, metallic, prim_path, kwargs))
        print(f"[DEBUG] 재질 생성 큐에 추가: {material_id} ({name})")
        return material_id
    
    def bind(self, object_id: str, material_id: str) -> bool:
        """재질 바인딩 요청 (큐에 추가)"""
        self._command_queue.put(('bind', object_id, material_id))
        print(f"[DEBUG] 재질 바인딩 큐에 추가: {object_id} <- {material_id}")
        return True
    
    def process_commands(self) -> None:
        """
        메인 스레드에서 호출: 큐에 쌓인 재질 생성/바인딩 처리
        
        시뮬레이션 루프에서 매 프레임마다 호출해야 합니다.
        """
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == 'create':
                    _, material_id, name, color, roughness, metallic, prim_path, kwargs = cmd
                    self._execute_create(material_id, name, color, roughness, metallic, prim_path, **kwargs)
                elif cmd[0] == 'bind':
                    self._execute_bind(cmd[1], cmd[2])
            except Exception as e:
                print(f"재질 명령 처리 오류: {e}")
    
    def _execute_create(
        self, 
        material_id: str,
        name: str, 
        color: List[float],
        roughness: float,
        metallic: float,
        prim_path: str,
        **kwargs
    ) -> None:
        """실제 재질 생성 (메인 스레드에서 실행)"""
        try:
            # PreviewSurface 재질 생성
            cfg = self.sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(color[:3]),
                roughness=roughness,
                metallic=metallic,
            )
            self.sim_utils.spawn_preview_surface(prim_path, cfg)
            print(f"[DEBUG] 재질 USD 생성 완료: {material_id} at {prim_path}")
            
        except Exception as e:
            print(f"재질 생성 오류: {e}")
    
    def _execute_bind(self, object_id: str, material_id: str) -> None:
        """실제 재질 바인딩 (메인 스레드에서 실행)"""
        if material_id not in self._materials:
            print(f"재질을 찾을 수 없음: {material_id}")
            return
        
        try:
            from pxr import UsdShade, UsdGeom, Usd
            import omni.usd
            
            mat = self._materials[material_id]
            
            # ObjectAdapter에서 물체의 prim_path 가져오기
            if hasattr(self, '_object_adapter') and self._object_adapter:
                obj = self._object_adapter.get_object(object_id)
                if obj is None:
                    print(f"물체를 찾을 수 없음: {object_id}")
                    return
                object_prim_path = obj.prim_path
            else:
                # ObjectAdapter가 없으면 object_id를 prim_path로 사용
                object_prim_path = object_id
            
            # USD Material 바인딩 (IsaacLab 호환 방식)
            stage = omni.usd.get_context().get_stage()
            object_prim = stage.GetPrimAtPath(object_prim_path)
            material_prim = stage.GetPrimAtPath(mat.prim_path)
            
            if not object_prim.IsValid():
                print(f"유효하지 않은 object prim: {object_prim_path}")
                return
            if not material_prim.IsValid():
                print(f"유효하지 않은 material prim: {mat.prim_path}")
                return
            
            material = UsdShade.Material(material_prim)
            
            # 하위 geometry/mesh prim 찾기 (Usd.PrimRange 사용)
            target_prims = []
            for descendant in Usd.PrimRange(object_prim):
                if descendant.IsA(UsdGeom.Mesh) or descendant.IsA(UsdGeom.Cube) or descendant.IsA(UsdGeom.Sphere):
                    target_prims.append(descendant)
            
            # mesh가 없으면 루트 prim에 바인딩
            if not target_prims:
                target_prims = [object_prim]
            
            # 모든 대상에 재질 바인딩
            for target in target_prims:
                UsdShade.MaterialBindingAPI(target).Bind(material)
                print(f"[DEBUG] 재질 바인딩 완료: {target.GetPath()} <- {mat.prim_path}")
                
        except Exception as e:
            import traceback
            print(f"재질 바인딩 오류: {e}")
            traceback.print_exc()
    
    def unbind(self, object_id: str) -> bool:
        # 재질 해제 구현
        return True
    
    def get_material(self, material_id: str) -> Optional[MaterialInfo]:
        return self._materials.get(material_id)
    
    def list_materials(self) -> List[MaterialInfo]:
        return list(self._materials.values())
    
    def update(
        self,
        material_id: str,
        color: Optional[List[float]] = None,
        roughness: Optional[float] = None,
        metallic: Optional[float] = None
    ) -> bool:
        if material_id not in self._materials:
            return False
        
        mat = self._materials[material_id]
        if color:
            mat.color = color
        if roughness is not None:
            mat.roughness = roughness
        if metallic is not None:
            mat.metallic = metallic
        
        return True


class DigitalTwinSceneAdapter(SceneAdapter):
    """
    Digital Twin 환경을 위한 Scene 어댑터
    
    기존 make_synthetic_data.py 스타일의 시뮬레이션을 감쌉니다.
    다중 카메라 관리를 지원합니다.
    """
    
    def __init__(self, sim, scene, simulation_app, sim_utils_module):
        """
        Args:
            sim: SimulationContext 인스턴스
            scene: InteractiveScene 인스턴스
            simulation_app: IsaacSim 앱 인스턴스
            sim_utils_module: isaaclab.sim 모듈
        """
        self.sim = sim
        self.scene = scene
        self.simulation_app = simulation_app
        self.sim_utils = sim_utils_module
        
        # 카메라 어댑터 관리
        self._cameras: Dict[str, DigitalTwinCameraAdapter] = {}
        self._active_camera_id: str = ""
        
        # scene에서 카메라 자동 탐지 및 어댑터 생성
        self._discover_cameras()
        
        # 기타 어댑터 초기화
        self._objects = DigitalTwinObjectAdapter(scene, sim_utils_module, str(sim.device))
        self._materials = DigitalTwinMaterialAdapter(sim_utils_module)
        
        # MaterialAdapter가 ObjectAdapter를 참조할 수 있도록 연결
        self._materials._object_adapter = self._objects
    
    def _discover_cameras(self) -> None:
        """InteractiveScene에서 카메라 센서 자동 탐지"""
        device = str(self.sim.device)
        
        # scene의 모든 키 순회하며 카메라 찾기
        for key in dir(self.scene):
            if key.startswith("_"):
                continue
            try:
                item = self.scene[key]
                # Camera 센서인지 확인 (data.output에 rgb가 있는지로 체크)
                if hasattr(item, "data") and hasattr(item.data, "output"):
                    if "rgb" in item.data.output:
                        camera_id = key
                        self._cameras[camera_id] = DigitalTwinCameraAdapter(item, device)
                        if not self._active_camera_id:
                            self._active_camera_id = camera_id
                        print(f"[INFO] 카메라 발견: {camera_id}")
            except (KeyError, TypeError, AttributeError):
                continue
        
        # 카메라가 없으면 기본 "camera" 키 시도
        if not self._cameras:
            try:
                camera_sensor = self.scene["camera"]
                self._cameras["camera"] = DigitalTwinCameraAdapter(camera_sensor, device)
                self._active_camera_id = "camera"
                print("[INFO] 기본 카메라 사용: camera")
            except KeyError:
                print("[WARNING] 씬에서 카메라를 찾을 수 없습니다.")
    
    def step(self) -> None:
        self.sim.step()
    
    def reset(self) -> None:
        self.sim.reset()
    
    def is_running(self) -> bool:
        return self.simulation_app.is_running()
    
    def get_camera_adapter(self) -> CameraAdapter:
        """현재 활성 카메라 어댑터 반환"""
        if self._active_camera_id and self._active_camera_id in self._cameras:
            return self._cameras[self._active_camera_id]
        # fallback: 첫 번째 카메라
        if self._cameras:
            return list(self._cameras.values())[0]
        raise RuntimeError("사용 가능한 카메라가 없습니다.")
    
    def get_object_adapter(self) -> ObjectAdapter:
        return self._objects
    
    def get_material_adapter(self) -> MaterialAdapter:
        return self._materials
    
    # 다중 카메라 관리 메서드 오버라이드
    
    def list_cameras(self) -> List[Dict[str, Any]]:
        """사용 가능한 모든 카메라 목록 반환"""
        cameras = []
        for camera_id, adapter in self._cameras.items():
            intrinsics = adapter.get_intrinsics()
            cameras.append({
                "id": camera_id,
                "name": camera_id.replace("_", " ").title(),
                "active": camera_id == self._active_camera_id,
                "resolution": f"{intrinsics.get('width', 640)}x{intrinsics.get('height', 480)}"
            })
        return cameras
    
    def get_active_camera_id(self) -> str:
        """현재 활성 카메라 ID 반환"""
        return self._active_camera_id
    
    def set_active_camera(self, camera_id: str) -> bool:
        """활성 카메라 변경"""
        if camera_id in self._cameras:
            self._active_camera_id = camera_id
            print(f"[INFO] 활성 카메라 변경: {camera_id}")
            return True
        return False
    
    def get_camera_by_id(self, camera_id: str) -> Optional[CameraAdapter]:
        """특정 ID의 카메라 어댑터 반환"""
        return self._cameras.get(camera_id)
    
    def add_camera(self, camera_id: str, camera_sensor) -> bool:
        """새 카메라 추가 (런타임에 동적 추가용)"""
        if camera_id in self._cameras:
            return False
        self._cameras[camera_id] = DigitalTwinCameraAdapter(
            camera_sensor, str(self.sim.device)
        )
        print(f"[INFO] 카메라 추가됨: {camera_id}")
        return True

