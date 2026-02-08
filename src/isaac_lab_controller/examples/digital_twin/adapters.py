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
from isaac_lab_controller.examples.digital_twin.cardboard_material import CardboardMaterial


class DigitalTwinCameraAdapter(CameraAdapter):
    """
    IsaacLab Camera 센서를 감싸는 어댑터
    
    스레드 안전한 프레임 버퍼링과 명령 큐를 사용합니다.
    - 메인 스레드: process_commands() + update_frame() 호출
    - 서버 스레드: set_lookat/orbit 등이 명령 큐에 추가, get_frame()으로 캐시된 프레임 읽기
    
    Args:
        camera: IsaacLab CameraCfg로 생성된 카메라 센서
    """
    
    def __init__(self, camera, device: str = "cuda:0", target_fps: int = 30, jpeg_quality: int = 85):
        self.camera = camera
        self.device = device
        self._current_eye = [0.0, 0.1, 0.8]
        self._current_target = [0.0, 0.1, 0.0]

        # FPS 제한
        self._target_fps = target_fps
        self._frame_interval = 1.0 / target_fps
        self._last_frame_time = 0.0
        self._jpeg_quality = jpeg_quality

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
        """
        실제 set_lookat 실행 (메인 스레드에서 호출)
        
        Warp 호환성 문제를 피하기 위해 USD API를 직접 사용합니다.
        """
        try:
            import math
            import omni.usd
            from pxr import UsdGeom, Gf
            
            # eye -> target 방향 벡터 계산
            dx = target[0] - eye[0]
            dy = target[1] - eye[1]
            dz = target[2] - eye[2]
            
            # 방향 벡터 정규화
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            if length < 1e-6:
                print("LookAt 설정 오류: eye와 target이 너무 가깝습니다")
                return False
            
            forward = Gf.Vec3d(dx/length, dy/length, dz/length)
            up = Gf.Vec3d(0, 0, 1)  # Z-up
            
            # right = forward x up
            right = forward ^ up  # Gf.Vec3d cross product
            right_len = right.GetLength()
            if right_len < 1e-6:
                # forward가 up과 평행한 경우
                up = Gf.Vec3d(0, 1, 0)
                right = forward ^ up
                right_len = right.GetLength()
            right = right / right_len
            
            # 실제 up = right x forward
            actual_up = right ^ forward
            
            # 회전 행렬 -> 쿼터니언 변환
            # 카메라는 -Z를 바라보므로 forward를 뒤집음
            # 행렬의 각 행: right, actual_up, -forward
            r = [right[0], right[1], right[2]]
            u = [actual_up[0], actual_up[1], actual_up[2]]
            f = [-forward[0], -forward[1], -forward[2]]

            # 회전 행렬에서 쿼터니언 직접 계산 (Matrix3d → Rotation 미지원)
            tr = r[0] + u[1] + f[2]
            if tr > 0:
                s = math.sqrt(tr + 1.0) * 2
                qw = 0.25 * s
                qx = (u[2] - f[1]) / s
                qy = (f[0] - r[2]) / s
                qz = (r[1] - u[0]) / s
            elif r[0] > u[1] and r[0] > f[2]:
                s = math.sqrt(1.0 + r[0] - u[1] - f[2]) * 2
                qw = (u[2] - f[1]) / s
                qx = 0.25 * s
                qy = (r[1] + u[0]) / s
                qz = (r[2] + f[0]) / s
            elif u[1] > f[2]:
                s = math.sqrt(1.0 + u[1] - r[0] - f[2]) * 2
                qw = (f[0] - r[2]) / s
                qx = (r[1] + u[0]) / s
                qy = 0.25 * s
                qz = (u[2] + f[1]) / s
            else:
                s = math.sqrt(1.0 + f[2] - r[0] - u[1]) * 2
                qw = (r[1] - u[0]) / s
                qx = (r[2] + f[0]) / s
                qy = (u[2] + f[1]) / s
                qz = 0.25 * s

            quat = Gf.Quatd(qw, qx, qy, qz)
            
            # USD Stage에서 카메라 prim 가져오기
            # IsaacLab의 prim_path는 정규식 패턴(env_.*)을 포함하므로
            # 실제 USD 경로로 변환 (env_0 사용)
            stage = omni.usd.get_context().get_stage()
            resolved_path = self.camera.cfg.prim_path.replace(".*", "0")
            camera_prim = stage.GetPrimAtPath(resolved_path)
            
            if camera_prim.IsValid():
                xform = UsdGeom.Xformable(camera_prim)
                xform.ClearXformOpOrder()

                # 변환 설정
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(eye[0], eye[1], eye[2]))

                orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
                orient_op.Set(quat)

                self._current_eye = eye
                self._current_target = target
                # print(f"[DEBUG] 카메라 이동 (USD API): eye={eye}, target={target}, path={resolved_path}")
            else:
                print(f"[ERROR] 카메라 prim을 찾을 수 없음: {resolved_path}")
                return False

            return True
            
        except Exception as e:
            import traceback
            print(f"LookAt 설정 오류: {e}")
            traceback.print_exc()
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
                elif cmd[0] == 'orbit':
                    self._execute_orbit(cmd[1], cmd[2], cmd[3], cmd[4])
            except Exception as e:
                print(f"명령 처리 오류: {e}")
    
    def update_frame(self) -> None:
        """
        메인 스레드에서 호출: 카메라 프레임 캡처 및 버퍼 저장

        이 메서드는 반드시 시뮬레이션 메인 스레드에서 호출해야 합니다.
        target_fps에 따라 프레임 캡처를 스킵하여 불필요한 GPU→CPU 복사를 줄입니다.
        """
        import time

        now = time.monotonic()
        if (now - self._last_frame_time) < self._frame_interval:
            return
        self._last_frame_time = now

        try:
            # 카메라 데이터 가져오기 (CUDA 텐서)
            rgb_tensor = self.camera.data.output["rgb"][0]  # (H, W, 4)
            rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)

            # RGBA -> BGR 변환
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2BGR)

            # JPEG 인코딩
            _, buffer = cv2.imencode('.jpg', rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
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
    
    def orbit(self, azimuth: float, elevation: float, distance: float,
              target: Optional[List[float]] = None) -> bool:
        """
        궤도 카메라 제어 (큐에 추가)
        
        구형 좌표계(azimuth, elevation, distance)를 카르테시안 좌표로 변환하여
        카메라 위치를 설정합니다.
        
        Args:
            azimuth: 수평 각도 (도, 0-360)
            elevation: 수직 각도 (도, -90 ~ 90)
            distance: 타겟으로부터의 거리
            target: 회전 중심점 [x, y, z], 기본값은 현재 타겟
        """
        self._command_queue.put(('orbit', azimuth, elevation, distance, target))
        return True
    
    def _execute_orbit(self, azimuth: float, elevation: float, distance: float,
                       target: Optional[List[float]] = None) -> bool:
        """실제 orbit 실행 (메인 스레드에서 호출)"""
        import math
        
        # 타겟 설정 (기본값: 현재 타겟 또는 원점)
        if target is None:
            target = self._current_target or [0.0, 0.0, 0.0]
        
        # 각도를 라디안으로 변환
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        
        # 구형 좌표 -> 카르테시안 좌표 변환
        # x = distance * cos(elevation) * cos(azimuth)
        # y = distance * cos(elevation) * sin(azimuth)
        # z = distance * sin(elevation)
        eye_x = target[0] + distance * math.cos(el_rad) * math.cos(az_rad)
        eye_y = target[1] + distance * math.cos(el_rad) * math.sin(az_rad)
        eye_z = target[2] + distance * math.sin(el_rad)
        
        eye = [eye_x, eye_y, eye_z]
        
        # set_lookat 실행
        return self._execute_set_lookat(eye, target)


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
                elif cmd[0] == 'transform':
                    self._execute_transform(cmd[1], cmd[2])
                elif cmd[0] == 'set_pose':
                    self._execute_set_pose(cmd[1], cmd[2], cmd[3])
            except Exception as e:
                print(f"물체 명령 처리 오류: {e}")
    
    # YCB USD 에셋 매핑 (물체 유형 → Nucleus 경로)
    _YCB_ASSETS = {
        "cracker_box": "Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugar_box": "Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "soup_can": "Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustard_bottle": "Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    }

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
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            prim_path = f"/World/DynamicObjects/obj_{self._counter}"
            self._counter += 1

            # YCB USD 에셋 직접 스폰
            if obj_type in self._YCB_ASSETS:
                asset_path = self._YCB_ASSETS[obj_type]
                usd_path = f"{ISAAC_NUCLEUS_DIR}/{asset_path}"
                cfg = self.sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(0.5, 0.5, 0.5),
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                )
                self.sim_utils.spawn_from_usd(prim_path, cfg, translation=tuple(position), orientation=tuple(rotation))

            elif obj_type == "box":
                cfg = self.sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.8, 0.2, 0.2))
                    ),
                )
                self.sim_utils.spawn_cuboid(prim_path, cfg, translation=tuple(position), orientation=tuple(rotation))

            elif obj_type == "sphere":
                cfg = self.sim_utils.SphereCfg(
                    radius=0.03,
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.2, 0.8, 0.2))
                    ),
                )
                self.sim_utils.spawn_sphere(prim_path, cfg, translation=tuple(position), orientation=tuple(rotation))

            elif obj_type == "cylinder":
                cfg = self.sim_utils.CylinderCfg(
                    radius=0.03,
                    height=0.06,
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.2, 0.2, 0.8))
                    ),
                )
                self.sim_utils.spawn_cylinder(prim_path, cfg, translation=tuple(position), orientation=tuple(rotation))
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
            {"type": "cracker_box", "name": "크래커 박스", "description": "YCB 크래커 박스"},
            {"type": "sugar_box", "name": "설탕 박스", "description": "YCB 설탕 박스"},
            {"type": "soup_can", "name": "토마토 수프 캔", "description": "YCB 수프 캔"},
            {"type": "mustard_bottle", "name": "머스타드 병", "description": "YCB 머스타드 병"},
            {"type": "box", "name": "기본 큐보이드", "description": "빨간 큐보이드"},
            {"type": "sphere", "name": "구", "description": "녹색 구체"},
            {"type": "cylinder", "name": "원통", "description": "파란 원기둥"},
        ]
    
    def set_pose(
        self,
        object_id: str,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None
    ) -> bool:
        """물체 위치/회전 변경 (큐에 추가)"""
        if object_id not in self._objects:
            return False
        self._command_queue.put(('set_pose', object_id, position, rotation))
        return True

    def _execute_set_pose(
        self,
        object_id: str,
        position: Optional[List[float]],
        rotation: Optional[List[float]],
    ) -> None:
        """실제 위치/회전 설정 (메인 스레드에서 실행, GPU tensor API 사용)"""
        if object_id not in self._objects:
            return

        try:
            obj = self._objects[object_id]

            pos = list(position) if position else list(obj.position)

            rot = list(rotation) if rotation else list(obj.rotation or [1, 0, 0, 0])
            # rot 형식: [w, x, y, z] → PhysX 형식: [x, y, z, w]
            qw, qx, qy, qz = rot[0], rot[1], rot[2], rot[3]

            # GPU tensor API로 위치 설정 (PhysX GPU Direct API 호환)
            import omni.physics.tensors.impl.api as physx

            sim_view = physx.create_simulation_view(self.device)
            sim_view.set_subspace_roots("/World/DynamicObjects")
            rb_view = sim_view.create_rigid_body_view(obj.prim_path)

            # PhysX 텐서 형식: [x, y, z, qx, qy, qz, qw]
            transforms = torch.tensor(
                [[pos[0], pos[1], pos[2], qx, qy, qz, qw]],
                dtype=torch.float32,
                device=self.device,
            )
            rb_view.set_transforms(transforms)

            # 속도 초기화 (이동 후 정지)
            zeros = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
            rb_view.set_velocities(zeros)

            # 저장된 정보 업데이트
            if position:
                obj.position = pos
            if rotation:
                obj.rotation = rot

            print(f"[DEBUG] 물체 위치 변경 (tensor API): {object_id} -> pos={pos}")
        except Exception as e:
            # GPU tensor API 실패 시 USD API 폴백 (CPU 모드 등)
            try:
                from pxr import UsdGeom, Gf
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(obj.prim_path)
                if not prim.IsValid():
                    print(f"[ERROR] set_pose: prim을 찾을 수 없음: {obj.prim_path}")
                    return

                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()

                if position:
                    xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
                    obj.position = pos
                if rotation:
                    xform.AddOrientOp().Set(Gf.Quatd(rot[0], rot[1], rot[2], rot[3]))
                    obj.rotation = rot

                print(f"[DEBUG] 물체 위치 변경 (USD 폴백): {object_id} -> pos={pos}")
            except Exception as fallback_e:
                print(f"위치 설정 오류: {e} / 폴백 오류: {fallback_e}")
    
    def transform(self, object_id: str, target_type: str = "random_box") -> bool:
        """물체를 다른 물체로 변환 (큐에 추가)"""
        if object_id not in self._objects:
            return False
        self._command_queue.put(('transform', object_id, target_type))
        print(f"[DEBUG] 물체 변환 큐에 추가: {object_id} -> {target_type}")
        return True
    
    def _execute_transform(self, object_id: str, target_type: str) -> None:
        """실제 변환 실행: 기존 물체 삭제 후 같은 위치에 새 물체 생성"""
        if object_id not in self._objects:
            print(f"[DEBUG] 변환할 물체를 찾을 수 없음: {object_id}")
            return
        
        try:
            from pxr import Usd
            import omni.usd
            import random
            
            # 1. 기존 물체 정보 저장
            old_obj = self._objects[object_id]
            old_position = list(old_obj.position)  # 수정 가능하도록 리스트 복사
            old_rotation = old_obj.rotation
            old_prim_path = old_obj.prim_path

            # 2. 기존 물체 삭제
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(old_prim_path)
            if prim.IsValid():
                stage.RemovePrim(old_prim_path)
            del self._objects[object_id]
            
            # 3. 새 물체 생성 (같은 위치)
            new_prim_path = f"/World/DynamicObjects/obj_{self._counter}"
            self._counter += 1
            
            if target_type == "random_box":
                # 랜덤 택배 박스 생성 (USD 에셋 우선, 큐보이드 폴백)
                box_config = CardboardMaterial.generate_config()
                CardboardMaterial.spawn(
                    new_prim_path, self.sim_utils, box_config,
                    translation=tuple(old_position), orientation=tuple(old_rotation),
                )

                # 새 물체 정보 저장
                new_obj = ObjectInfo(
                    id=object_id,  # 같은 ID 유지
                    name=f"택배박스_{self._counter}",
                    obj_type="random_box",
                    position=old_position,
                    rotation=old_rotation,
                    prim_path=new_prim_path,
                )
                self._objects[object_id] = new_obj
                
                print(f"[DEBUG] 물체 변환 완료: {object_id} -> 택배박스 (크기: {box_config['size']})")
            else:
                print(f"지원하지 않는 변환 유형: {target_type}")
                
        except Exception as e:
            import traceback
            print(f"물체 변환 오류: {e}")
            traceback.print_exc()
    
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

