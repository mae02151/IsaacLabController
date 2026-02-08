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


def spawn_pallets():
    """물류 센터 팔레트 2개 초기 배치"""
    pallet_positions = [
        (-0.15, 0.1, 0.075),
        (0.15, 0.1, 0.075),
    ]
    pallet_scale = (1.0, 1.0, 1.0)

    for i, pos in enumerate(pallet_positions):
        prim_path = f"/World/Pallets/pallet_{i}"
        try:
            usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT_visual_collision.usd"
            cfg = sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=pallet_scale,
            )
            sim_utils.spawn_from_usd(prim_path, cfg, translation=pos)
            print(f"[INFO] 팔레트 스폰 (USD): {prim_path}")
        except Exception as e:
            print(f"[INFO] USD 팔레트 실패, 큐보이드 사용: {e}")
            cfg = sim_utils.CuboidCfg(
                size=(0.3, 0.1, 0.3),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.45, 0.25),
                    roughness=0.9,
                ),
            )
            sim_utils.spawn_cuboid(prim_path, cfg, translation=pos)
            print(f"[INFO] 팔레트 스폰 (큐보이드): {prim_path}")

        # +0.3 쪽 (index 1) 팔레트 색상 변경 (연두색)
        if i == 1:
            try:
                # 0. /World/Looks 스코프 확인 및 생성
                import omni.usd
                from pxr import UsdShade, UsdGeom, Sdf

                stage = omni.usd.get_context().get_stage()
                looks_path = "/World/Looks"
                looks_prim = stage.GetPrimAtPath(looks_path)
                if not looks_prim.IsValid():
                    UsdGeom.Scope.Define(stage, looks_path)

                # 1. 재질 생성
                material_path = "/World/Looks/GreenMaterial"
                material_cfg = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.5, 0.8, 0.2),  # 연두색
                    roughness=0.5,
                    metallic=0.0,
                )
                sim_utils.spawn_preview_surface(material_path, material_cfg)

                # 2. 바인딩 (USD API 사용 - 강제 적용)
                pallet_prim = stage.GetPrimAtPath(prim_path)
                material_prim = stage.GetPrimAtPath(material_path)

                if pallet_prim.IsValid() and material_prim.IsValid():
                    # Material Binding API 적용 (strongerThanDescendants)
                    material = UsdShade.Material(material_prim)
                    UsdShade.MaterialBindingAPI(pallet_prim).Bind(
                        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
                    )
                    print(f"[INFO] 팔레트 색상 변경 완료 (강제): {prim_path} -> 연두색")
                else:
                    print(f"[WARN] 팔레트 색상 변경 실패: Prim을 찾을 수 없음 ({prim_path})")

            except Exception as e:
                print(f"[ERROR] 팔레트 색상 변경 중 오류 발생: {e}")


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
