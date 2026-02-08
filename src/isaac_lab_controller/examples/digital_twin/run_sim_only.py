
"""
IsaacLab 시뮬레이션 전용 실행 스크립트 (서버 분리 모드)

이 스크립트는 IsaacLab 시뮬레이션만 실행하고,
ZMQ를 통해 별도 프로세스의 FastAPI 서버와 통신합니다.

사용법:
    # 터미널 1: 시뮬레이션 (한 번만 실행, 계속 켜둠)
    c:\\Users\\dongu\\IsaacLab\\isaaclab.bat -p run_sim_only.py
    
    # 터미널 2: 서버 (수정 후 재시작)
    python server_standalone.py
"""

import torch
import os
import numpy as np

# IsaacLab 앱 실행 (가장 먼저!)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(launcher_args={
    "headless": False, 
    "enable_cameras": True
})
simulation_app = app_launcher.app

# 이후 import
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# IPC 브릿지
from isaac_lab_controller.ipc import SimulationBridge
from isaac_lab_controller.examples.digital_twin.adapters import DigitalTwinSceneAdapter


def get_lookat_quat(cam_pos, target_pos, up=np.array([0, 0, 1])):
    """카메라 LookAt 쿼터니언 계산"""
    forward = target_pos - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    z_axis = -forward 
    right = np.cross(up, z_axis)
    x_axis = right / np.linalg.norm(right)
    y_axis = np.cross(z_axis, x_axis)
    
    rot_mat = np.column_stack((x_axis, y_axis, z_axis))
    
    tr = np.trace(rot_mat)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        qy = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        qz = (rot_mat[1, 0] - rot_mat[0, 1]) / S
    elif (rot_mat[0, 0] > rot_mat[1, 1]) and (rot_mat[0, 0] > rot_mat[2, 2]):
        S = np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
        qw = (rot_mat[2, 1] - rot_mat[1, 2]) / S
        qx = 0.25 * S
        qy = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        qz = (rot_mat[0, 2] + rot_mat[2, 0]) / S
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        S = np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
        qw = (rot_mat[0, 2] - rot_mat[2, 0]) / S
        qx = (rot_mat[0, 1] + rot_mat[1, 0]) / S
        qy = 0.25 * S
        qz = (rot_mat[1, 2] + rot_mat[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
        qw = (rot_mat[1, 0] - rot_mat[0, 1]) / S
        qx = (rot_mat[0, 2] + rot_mat[2, 0]) / S
        qy = (rot_mat[1, 2] + rot_mat[2, 1]) / S
        qz = 0.25 * S
    return (qw, qx, qy, qz)


def spawn_pallets():
    """물류 센터 팔레트 2개 초기 배치"""
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR as _NUCLEUS_DIR

    pallet_positions = [
        (-0.3, 0.1, 0.0),
        (0.3, 0.1, 0.0),
    ]
    pallet_scale = (1.0, 1.0, 1.0)

    for i, pos in enumerate(pallet_positions):
        prim_path = f"/World/Pallets/pallet_{i}"
        try:
            usd_path = f"{_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT_visual_collision.usd"
            cfg = sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=pallet_scale,
            )
            sim_utils.spawn_from_usd(prim_path, cfg, translation=pos)
            print(f"[SIM] 팔레트 스폰 (USD): {prim_path}")
        except Exception as e:
            print(f"[SIM] USD 팔레트 실패, 큐보이드 사용: {e}")
            cfg = sim_utils.CuboidCfg(
                size=(0.3, 0.2, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.45, 0.25),
                    roughness=0.9,
                ),
            )
            sim_utils.spawn_cuboid(prim_path, cfg, translation=pos)
            print(f"[SIM] 팔레트 스폰 (큐보이드): {prim_path}")


def main():
    print("=" * 60)
    print("[SIM] IsaacLab 시뮬레이션 시작 (서버 분리 모드)")
    print("=" * 60)
    
    # 1. 시뮬레이션 설정
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,  # 100Hz 물리 (테스트용 고속)
        physx=sim_utils.PhysxCfg(
            enable_stabilization=True,
            enable_external_forces_every_iteration=True
        )
    )
    sim = SimulationContext(sim_cfg)

    # 2. 장면 구성
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    
    # 조명 및 바닥
    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # 카메라 설정 (수직 top-down 뷰: right=+X, down=+Y, 양 팔레트 중심)
    cam_pos = np.array([0.0, 0.1, 0.8])
    target_pos = np.array([0.0, 0.1, 0.0])
    cam_rot = get_lookat_quat(cam_pos, target_pos, up=np.array([0, 1, 0]))

    scene_cfg.camera = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=tuple(cam_pos), rot=cam_rot, convention="opengl")
    )

    # 장면 생성
    scene = InteractiveScene(scene_cfg)

    # 팔레트 배치 (sim.reset() 전에 스폰)
    spawn_pallets()

    sim.reset()
    
    print("[SIM] 시뮬레이션 초기화 완료")

    # 3. 어댑터 및 ZMQ 브릿지 설정
    adapter = DigitalTwinSceneAdapter(sim, scene, simulation_app, sim_utils)
    
    # ZMQ 브릿지 시작 (포트 5555)
    bridge = SimulationBridge(port=5555)
    bridge.register_adapter("camera", adapter.get_camera_adapter())
    bridge.register_adapter("object", adapter.get_object_adapter())
    bridge.register_adapter("material", adapter.get_material_adapter())
    bridge.register_adapter("scene", adapter)
    
    print("")
    print("=" * 60)
    print("[SIM] ZMQ 브릿지 대기 중: tcp://*:5555")
    print("[SIM] 이제 다른 터미널에서 server_standalone.py를 실행하세요!")
    print("=" * 60)
    print("")

    # 4. 시뮬레이션 루프
    step_count = 0
    while simulation_app.is_running():
        sim.step()
        
        # 씬 업데이트 (센서 데이터 갱신)
        scene.update(sim.get_physics_dt())
        
        # 어댑터 가져오기
        camera_adapter = adapter.get_camera_adapter()
        object_adapter = adapter.get_object_adapter()
        material_adapter = adapter.get_material_adapter()
        
        # 1. 큐에 쌓인 명령 처리 (매 프레임)
        if hasattr(camera_adapter, 'process_commands'):
            camera_adapter.process_commands()
        if hasattr(object_adapter, 'process_commands'):
            object_adapter.process_commands()
        if hasattr(material_adapter, 'process_commands'):
            material_adapter.process_commands()
        
        # 2. ZMQ 메시지 처리 (non-blocking)
        bridge.process_messages()
        
        # 3. 프레임 캡처 (매 스텝)
        if hasattr(camera_adapter, 'update_frame'):
            camera_adapter.update_frame()
        
        step_count += 1

    print("[SIM] 시뮬레이션 종료")
    bridge.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
