"""
Digital Twin 통합 실행 스크립트 (IsaacSim + FastAPI 단일 프로세스)

ZMQ 통신 없이 IsaacSim과 백엔드 서버를 한 프로세스에서 실행합니다.
서버는 백그라운드 스레드에서, 시뮬레이션은 메인 스레드에서 동작합니다.

사용법:
    isaaclab -p run.py

    브라우저에서 http://localhost:8000 접속

비교:
    - run.py: 통합 모드 (단일 프로세스, ZMQ 없음, 빠름)
    - run_sim_only.py + server_standalone.py: 분리 모드 (ZMQ 통신, 서버 재시작 가능)
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

# IsaacLabController
from isaac_lab_controller import ControlServer
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
    print("[INFO] IsaacLabController 통합 모드 시작")
    print("[INFO] ZMQ 없이 단일 프로세스로 실행합니다")
    print("=" * 60)

    # 1. 시뮬레이션 설정
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,  # 100Hz 물리
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

    # 카메라 설정 (수직 top-down 뷰: right=+X, down=+Y)
    cam_pos = np.array([0.5, 0.0, 1.5])
    target_pos = np.array([0.5, 0.0, 0.0])
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

    # 장면 생성 및 시뮬레이션 시작
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print("[INFO] 시뮬레이션 초기화 완료")

    # 3. 어댑터 생성 (sim.reset() 후에 생성)
    adapter = DigitalTwinSceneAdapter(sim, scene, simulation_app, sim_utils)

    # 4. 서버 시작 (백그라운드 스레드)
    # 주의: IsaacSim(Omniverse Kit)이 내부적으로 포트 8000을 사용하므로 다른 포트 사용
    server_port = 8080
    server = ControlServer(adapter, port=server_port)
    server.start_background()

    print("")
    print("=" * 60)
    print(f"[INFO] 웹 서버 시작: http://localhost:{server_port}")
    print("[INFO] 브라우저에서 접속하여 시뮬레이션을 제어하세요!")
    print("=" * 60)
    print("")

    # 5. 시뮬레이션 루프
    step_count = 0
    while simulation_app.is_running():
        sim.step()

        # 씬 업데이트 (센서 데이터 갱신)
        scene.update(sim.get_physics_dt())

        # 메인 스레드에서 명령 처리 (CUDA 스레드 안전성)
        camera_adapter = adapter.get_camera_adapter()
        object_adapter = adapter.get_object_adapter()
        material_adapter = adapter.get_material_adapter()

        # 큐에 쌓인 카메라/물체/재질 명령 처리 (매 프레임)
        camera_adapter.process_commands()
        object_adapter.process_commands()
        material_adapter.process_commands()

        # 프레임 캡처 (매 스텝)
        camera_adapter.update_frame()

        step_count += 1

    print("[INFO] 시뮬레이션 종료")
    server.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
