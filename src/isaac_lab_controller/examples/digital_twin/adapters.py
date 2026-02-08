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
    RobotAdapter,
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
    
    def __init__(self, camera, device: str = "cuda:0", target_fps: int = 60, jpeg_quality: int = 75):
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
        import torch

        now = time.monotonic()
        if (now - self._last_frame_time) < self._frame_interval:
            return
        self._last_frame_time = now

        try:
            # CUDA 에러 상태 클리어 (다른 모듈의 CUDA 에러가 남아있을 수 있음)
            if torch.cuda.is_available():
                torch.cuda.current_device()  # CUDA 컨텍스트 보장
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    # CUDA 에러 상태 클리어
                    pass

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
            print(f"[CameraAdapter] 프레임 업데이트 오류: {e}")
    
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

        # 지연 포즈 업데이트 큐 (GPU 파이프라인 호환)
        # spawn 후 다음 프레임에서 GPU tensor API로 위치를 재설정
        self._deferred_poses: List[tuple] = []
    
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
        # 1. 지연된 포즈 업데이트 먼저 적용 (이전 프레임에서 spawn된 물체)
        if self._deferred_poses:
            pending = self._deferred_poses[:]
            self._deferred_poses.clear()
            for item in pending:
                if len(item) == 4:
                    object_id, position, rotation, retry_count = item
                else:
                    object_id, position, rotation = item
                    retry_count = 0
                self._execute_set_pose(object_id, position, rotation, _retry_count=retry_count)

        # 2. 큐에 쌓인 명령 처리
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

    # 스폰 시 Z 오프셋 (위에서 떨어뜨리기, 기존 물체 밀어내기 방지)
    _SPAWN_DROP_HEIGHT = 0.3

    def _execute_spawn(
        self,
        object_id: str,
        obj_type: str,
        position: List[float],
        rotation: List[float],
        name: Optional[str] = None,
        **kwargs
    ) -> None:
        """실제 물체 생성 (메인 스레드에서 실행)

        물체를 목표 위치보다 높은 곳에서 스폰하여 중력으로 자연스럽게 떨어뜨립니다.
        이렇게 하면 기존 물체를 밀어내는 현상을 방지할 수 있습니다.
        """
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            prim_path = f"/World/DynamicObjects/obj_{self._counter}"
            self._counter += 1

            # 스폰 위치: 목표 Z + 드롭 높이
            spawn_pos = (position[0], position[1], position[2] + self._SPAWN_DROP_HEIGHT)

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
                self.sim_utils.spawn_from_usd(prim_path, cfg, translation=spawn_pos, orientation=tuple(rotation))

            elif obj_type == "box":
                cfg = self.sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.05),
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.8, 0.2, 0.2))
                    ),
                )
                self.sim_utils.spawn_cuboid(prim_path, cfg, translation=spawn_pos, orientation=tuple(rotation))

            elif obj_type == "sphere":
                cfg = self.sim_utils.SphereCfg(
                    radius=0.03,
                    rigid_props=self.sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=self.sim_utils.CollisionPropertiesCfg(),
                    visual_material=self.sim_utils.PreviewSurfaceCfg(
                        diffuse_color=kwargs.get("color", (0.2, 0.8, 0.2))
                    ),
                )
                self.sim_utils.spawn_sphere(prim_path, cfg, translation=spawn_pos, orientation=tuple(rotation))

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
                self.sim_utils.spawn_cylinder(prim_path, cfg, translation=spawn_pos, orientation=tuple(rotation))
            else:
                print(f"지원하지 않는 물체 유형: {obj_type}")
                return

            # 물체 정보 저장 (목표 위치 기록, 실제 스폰은 높은 곳)
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

    # 지연 포즈 최대 재시도 횟수 (physics view에 prim이 아직 없을 때)
    _MAX_DEFERRED_RETRIES = 5

    def _execute_set_pose(
        self,
        object_id: str,
        position: Optional[List[float]],
        rotation: Optional[List[float]],
        _retry_count: int = 0,
    ) -> None:
        """실제 위치/회전 설정 (메인 스레드에서 실행, GPU tensor API 사용)

        Isaac Lab 표준 방식: physx.create_simulation_view("torch", stage_id)로
        시뮬레이션 뷰를 생성하고 rigid_body_view를 통해 GPU 파이프라인 호환 변환 수행.
        """
        if object_id not in self._objects:
            return

        obj = self._objects[object_id]
        pos = list(position) if position else list(obj.position)
        rot = list(rotation) if rotation else list(obj.rotation or [1, 0, 0, 0])
        # rot 형식: [w, x, y, z] → PhysX 형식: [x, y, z, w]
        qw, qx, qy, qz = rot[0], rot[1], rot[2], rot[3]

        try:
            # GPU tensor API로 위치 설정 (Isaac Lab 표준 방식)
            import omni.physics.tensors.impl.api as physx
            from isaaclab.sim.utils.stage import get_current_stage_id

            stage_id = get_current_stage_id()
            sim_view = physx.create_simulation_view("torch", stage_id)
            sim_view.set_subspace_roots("/")
            rb_view = sim_view.create_rigid_body_view(obj.prim_path)

            if rb_view.count == 0:
                # Physics가 아직 이 prim을 인식하지 못함 → 다음 프레임에 재시도
                if _retry_count < self._MAX_DEFERRED_RETRIES:
                    self._deferred_poses.append((object_id, position, rotation, _retry_count + 1))
                    print(f"[DEBUG] rigid body 미발견, 재시도 예약 ({_retry_count + 1}/{self._MAX_DEFERRED_RETRIES}): {object_id}")
                else:
                    print(f"[WARN] rigid body 최대 재시도 초과, 포기: {object_id}")
                return

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
            print(f"[WARN] tensor API 위치 설정 실패: {e}")
            # GPU tensor API 실패 시 USD API 폴백 (Isaac Lab standardize_xform_ops 사용)
            try:
                from isaaclab.sim.utils.transforms import standardize_xform_ops
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath(obj.prim_path)
                if not prim.IsValid():
                    print(f"[ERROR] set_pose: prim을 찾을 수 없음: {obj.prim_path}")
                    return

                standardize_xform_ops(
                    prim,
                    translation=tuple(pos),
                    orientation=tuple(rot),
                )

                if position:
                    obj.position = pos
                if rotation:
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
            
            # 3. 새 물체 생성 (높은 곳에서 떨어뜨리기)
            new_prim_path = f"/World/DynamicObjects/obj_{self._counter}"
            self._counter += 1

            # 스폰 위치: 기존 위치보다 높은 곳
            spawn_pos = (old_position[0], old_position[1], old_position[2] + self._SPAWN_DROP_HEIGHT)

            if target_type == "random_box":
                # 랜덤 택배 박스 생성 (USD 에셋 우선, 큐보이드 폴백)
                box_config = CardboardMaterial.generate_config()
                CardboardMaterial.spawn(
                    new_prim_path, self.sim_utils, box_config,
                    translation=spawn_pos, orientation=tuple(old_rotation),
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


class DigitalTwinRobotAdapter(RobotAdapter):
    """
    IsaacLab Articulation을 감싸는 로봇 어댑터

    텔레오퍼레이션 모드:
    - Demo 모드: 사인파 모션 생성 (테스트용)
    - ROS2 모드: 실제 로봇의 joint_states를 ROS2로 수신하여 시뮬레이션에 적용

    스레드 안전한 명령 큐를 사용합니다.
    - 서버 스레드: start_teleop/stop_teleop/set_joint_positions 명령을 큐에 추가
    - 메인 스레드: process_commands()로 실제 관절 상태 적용
    """

    # Open Manipulator X 관절 설정
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
    GRIPPER_JOINT_NAME = "gripper_left_joint"
    JOINT_LIMITS = {
        "joint1": (-3.14159, 3.14159),
        "joint2": (-1.5, 1.5),
        "joint3": (-1.5, 1.4),
        "joint4": (-1.7, 1.97),
        "gripper_left_joint": (-0.01, 0.019),
    }

    def __init__(self, robot, device: str = "cuda:0"):
        """
        Args:
            robot: IsaacLab Articulation 인스턴스
            device: torch 디바이스
        """
        self.robot = robot
        self.device = device

        # 텔레오프 상태
        self._teleop_running = False
        self._teleop_mode = "demo"  # "demo" | "ros2"
        self._ros2_connected = False
        self._ros2_bridge = None
        self._demo_time = 0.0

        # 현재 관절 상태 (스레드 안전)
        self._joint_lock = threading.Lock()
        self._current_joint_positions = [0.0, -0.5, 0.5, 0.0, 0.01]  # 초기 포즈

        # 관절 인덱스 매핑
        self._arm_joint_indices = [robot.joint_names.index(name) for name in self.ARM_JOINT_NAMES]
        self._gripper_joint_idx = robot.joint_names.index(self.GRIPPER_JOINT_NAME)

        # 명령 큐 (스레드 안전)
        from queue import Queue
        self._command_queue: Queue = Queue()

        # 원본 PD 게인 저장 (텔레오프 시 비활성화용)
        self._original_stiffness = self.robot.root_physx_view.get_dof_stiffnesses().clone()
        self._original_damping = self.robot.root_physx_view.get_dof_dampings().clone()

        # 관절 상태 텐서 미리 할당 (매 프레임 재사용하여 CUDA 할당 부하 감소)
        import torch
        num_envs = 1
        self._joint_pos_buffer = torch.zeros((num_envs, robot.num_joints), device=self.device)
        self._joint_vel_buffer = torch.zeros((num_envs, robot.num_joints), device=self.device)

        print(f"[RobotAdapter] 초기화: joints={robot.joint_names}")
        print(f"[RobotAdapter] arm_indices={self._arm_joint_indices}, gripper_idx={self._gripper_joint_idx}")
        print(f"[RobotAdapter] 원본 stiffness={self._original_stiffness}")
        print(f"[RobotAdapter] 원본 damping={self._original_damping}")

    def get_joint_names(self) -> List[str]:
        return self.ARM_JOINT_NAMES + [self.GRIPPER_JOINT_NAME]

    def get_joint_positions(self) -> List[float]:
        with self._joint_lock:
            return list(self._current_joint_positions)

    def set_joint_positions(self, positions: List[float]) -> bool:
        """관절 위치 설정 요청 (큐에 추가)"""
        self._command_queue.put(('set_joints', positions))
        return True

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "name": "Open Manipulator X",
            "num_joints": 5,
            "joint_names": self.get_joint_names(),
            "joint_limits": self.JOINT_LIMITS,
        }

    def start_teleop(self, mode: str = "demo") -> bool:
        """텔레오퍼레이션 시작 요청 (큐에 추가)"""
        self._command_queue.put(('start_teleop', mode))
        return True

    def stop_teleop(self) -> bool:
        """텔레오퍼레이션 중지 요청 (큐에 추가)"""
        self._command_queue.put(('stop_teleop',))
        return True

    def get_teleop_status(self) -> Dict[str, Any]:
        return {
            "running": self._teleop_running,
            "mode": self._teleop_mode,
            "connected": self._ros2_connected if self._teleop_mode == "ros2" else True,
        }

    def process_commands(self) -> None:
        """
        메인 스레드에서 호출: 큐 처리 + 텔레오프 업데이트

        시뮬레이션 루프에서 매 프레임마다 호출해야 합니다.
        """
        import torch

        # 1. 명령 큐 처리
        while not self._command_queue.empty():
            try:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == 'set_joints':
                    self._execute_set_joints(cmd[1])
                elif cmd[0] == 'start_teleop':
                    self._execute_start_teleop(cmd[1])
                elif cmd[0] == 'stop_teleop':
                    self._execute_stop_teleop()
            except Exception as e:
                print(f"[RobotAdapter] 명령 처리 오류: {e}")

        # 2. 텔레오프 업데이트
        if not self._teleop_running:
            return

        try:
            if self._teleop_mode == "demo":
                self._update_demo_teleop()
            elif self._teleop_mode == "ros2":
                self._update_ros2_teleop()
        except Exception as e:
            print(f"[RobotAdapter] 텔레오프 업데이트 오류: {e}")

    def _execute_set_joints(self, positions: List[float]) -> None:
        """실제 관절 위치 적용 (메인 스레드에서 실행)

        PD position target을 사용하여 부드럽게 이동합니다.
        write_joint_state_to_sim()은 관절을 순간이동시켜 물체를 밀어내는
        물리 불안정을 유발하므로 사용하지 않습니다.
        """
        try:
            # 미리 할당된 버퍼 재사용 (CUDA 텐서 할당 최소화)
            self._joint_pos_buffer.zero_()

            # arm 관절 설정
            for i, idx in enumerate(self._arm_joint_indices):
                if i < len(positions):
                    self._joint_pos_buffer[:, idx] = positions[i]

            # gripper 설정
            if len(positions) > 4:
                self._joint_pos_buffer[:, self._gripper_joint_idx] = positions[4]

            # PD position target 설정 (부드러운 이동, 물리 안정성 유지)
            # GPU 파이프라인: 텐서를 GPU에 유지해야 함 (device -1 = CPU 오류 방지)
            indices = torch.tensor([0], dtype=torch.long, device=self.device)
            self.robot.root_physx_view.set_dof_position_targets(
                self._joint_pos_buffer, indices
            )

            with self._joint_lock:
                self._current_joint_positions = list(positions[:5])

        except Exception as e:
            print(f"[RobotAdapter] 관절 설정 오류: {e}")

    def _execute_start_teleop(self, mode: str) -> None:
        """텔레오퍼레이션 시작 (메인 스레드에서 실행)"""
        if mode == "ros2":
            try:
                # ROS2 브릿지 동적 임포트
                import sys
                teleop_path = "/home/jwson/IsaacLab/scripts/teleop/open_manipulator_x"
                if teleop_path not in sys.path:
                    sys.path.insert(0, teleop_path)

                from ros2_bridge import OpenManipulatorRos2Bridge, ROS2_AVAILABLE
                if not ROS2_AVAILABLE:
                    print("[RobotAdapter] ROS2를 사용할 수 없습니다. demo 모드로 전환합니다.")
                    mode = "demo"
                else:
                    from config import OpenManipulatorConfig, Ros2Config
                    robot_cfg = OpenManipulatorConfig()
                    ros2_cfg = Ros2Config()
                    self._ros2_bridge = OpenManipulatorRos2Bridge(
                        robot_cfg=robot_cfg,
                        ros2_cfg=ros2_cfg,
                        device=self.device
                    )
                    self._ros2_connected = False
                    print("[RobotAdapter] ROS2 브릿지 초기화 완료")
            except Exception as e:
                print(f"[RobotAdapter] ROS2 초기화 실패: {e}, demo 모드로 전환")
                mode = "demo"

        self._teleop_mode = mode
        self._teleop_running = True
        self._demo_time = 0.0

        # PD 게인 유지 — set_dof_position_targets()로 부드럽게 추종
        # PD 컨트롤러가 물리적으로 안정한 힘을 계산하므로 물체 밀어내기 방지
        print(f"[RobotAdapter] 텔레오퍼레이션 시작: mode={mode}")

    def _execute_stop_teleop(self) -> None:
        """텔레오퍼레이션 중지 (메인 스레드에서 실행)"""
        self._teleop_running = False
        if self._ros2_bridge is not None:
            self._ros2_bridge = None
        self._ros2_connected = False
        print("[RobotAdapter] 텔레오퍼레이션 중지")

    def _update_demo_teleop(self) -> None:
        """데모 모드: 사인파 모션 생성 및 적용"""
        import torch
        import math

        self._demo_time += 0.01  # ~100Hz 시뮬레이션 dt

        positions = [
            0.5 * math.sin(self._demo_time * 0.5),           # joint1
            -0.5 + 0.3 * math.sin(self._demo_time * 0.7),    # joint2
            0.5 + 0.2 * math.sin(self._demo_time * 0.9),     # joint3
            0.3 * math.sin(self._demo_time * 1.1),           # joint4
            0.01 + 0.009 * math.sin(self._demo_time * 0.3),  # gripper
        ]

        self._execute_set_joints(positions)

    def _update_ros2_teleop(self) -> None:
        """ROS2 모드: 실제 로봇 관절 상태 수신 및 적용"""
        import torch

        if self._ros2_bridge is None:
            return

        # 연결 상태 업데이트
        self._ros2_connected = self._ros2_bridge.is_connected
        if not self._ros2_connected:
            return

        # 관절 상태 텐서 가져오기 [j1, j2, j3, j4, gripper]
        joint_state = self._ros2_bridge.get_joint_state_tensor()
        positions = joint_state.cpu().tolist()

        self._execute_set_joints(positions)


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

        # 로봇 어댑터 (선택적 - scene에 robot이 있으면 생성)
        self._robot: Optional[DigitalTwinRobotAdapter] = None
        self._discover_robot()
    
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

    def _discover_robot(self) -> None:
        """InteractiveScene에서 로봇(Articulation) 자동 탐지"""
        device = str(self.sim.device)
        try:
            robot = self.scene["robot"]
            # Articulation인지 확인 (joint_names 속성으로 체크)
            if hasattr(robot, "joint_names") and hasattr(robot, "write_joint_state_to_sim"):
                self._robot = DigitalTwinRobotAdapter(robot, device)
                print(f"[INFO] 로봇 발견: {robot.joint_names}")
        except (KeyError, TypeError, AttributeError):
            print("[INFO] 씬에 로봇이 없습니다.")

    def get_robot_adapter(self) -> Optional[RobotAdapter]:
        """로봇 어댑터 반환"""
        return self._robot

    def set_robot_adapter(self, robot_adapter: DigitalTwinRobotAdapter) -> None:
        """로봇 어댑터 수동 설정"""
        self._robot = robot_adapter

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

