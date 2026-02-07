
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
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg, ArticulationCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 로봇 설정
from isaaclab_assets import OPEN_MANIPULATOR_X_GRIPPER_CFG

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

    # 로봇 설정
    scene_cfg.robot = OPEN_MANIPULATOR_X_GRIPPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # 카메라 설정
    cam_pos = np.array([1.5, 0.0, 0.6])
    target_pos = np.array([0.0, 0.0, 0.0])
    cam_rot = get_lookat_quat(cam_pos, target_pos)

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

    # 로봇 어댑터 등록
    robot_adapter = adapter.get_robot_adapter()
    if robot_adapter:
        bridge.register_adapter("robot", robot_adapter)
        print("[SIM] 로봇 어댑터 등록 완료")
    
    print("")
    print("=" * 60)
    print("[SIM] ZMQ 브릿지 대기 중: tcp://*:5555")
    print("[SIM] 이제 다른 터미널에서 server_standalone.py를 실행하세요!")
    print("=" * 60)
    print("")

    # 4. 시뮬레이션 루프
    camera_adapter = adapter.get_camera_adapter()
    object_adapter = adapter.get_object_adapter()
    material_adapter = adapter.get_material_adapter()

    step_count = 0
    while simulation_app.is_running():
        # 1. 로봇 명령 처리 (sim.step 이전에 실행해야 렌더링에 반영됨)
        if robot_adapter and hasattr(robot_adapter, 'process_commands'):
            robot_adapter.process_commands()

        # 2. 물리 시뮬레이션 + 렌더링
        sim.step()

        # 3. 씬 업데이트 (센서 데이터 갱신)
        scene.update(sim.get_physics_dt())

        # 4. 카메라/물체/재질 명령 처리
        if hasattr(camera_adapter, 'process_commands'):
            camera_adapter.process_commands()
        if hasattr(object_adapter, 'process_commands'):
            object_adapter.process_commands()
        if hasattr(material_adapter, 'process_commands'):
            material_adapter.process_commands()

        # 5. ZMQ 메시지 처리 (non-blocking)
        bridge.process_messages()

        # 6. 프레임 캡처 (매 스텝)
        if hasattr(camera_adapter, 'update_frame'):
            camera_adapter.update_frame()

        step_count += 1

    print("[SIM] 시뮬레이션 종료")
    bridge.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
