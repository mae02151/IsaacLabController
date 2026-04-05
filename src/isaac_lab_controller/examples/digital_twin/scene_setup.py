"""
Digital Twin 씬 설정 모듈

카메라 유틸리티, 씬 구성, 팔레트 스폰 등
run.py와 run_sim_only.py에서 공통으로 사용하는 씬 설정 로직을 제공합니다.

NOTE: IsaacLab import는 함수 내부에서 lazy import합니다.
      AppLauncher 실행 후에만 사용 가능하기 때문입니다.
"""

import numpy as np
from typing import Tuple, List, Optional


def get_lookat_quat(
    cam_pos: np.ndarray,
    target_pos: np.ndarray,
    up: np.ndarray = np.array([0, 0, 1]),
) -> Tuple[float, float, float, float]:
    """카메라 LookAt 쿼터니언 계산 (OpenGL 컨벤션, -Z forward)"""
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


def create_simulation_cfg():
    """Digital Twin 기본 시뮬레이션 설정 (100Hz, stabilization 활성)"""
    import isaaclab.sim as sim_utils

    return sim_utils.SimulationCfg(
        dt=0.01,
        physx=sim_utils.PhysxCfg(
            enable_stabilization=True,
            enable_external_forces_every_iteration=True,
        ),
    )


def create_scene_cfg(
    cam_pos: np.ndarray = np.array([0.0, 0.1, 0.8]),
    target_pos: np.ndarray = np.array([0.0, 0.1, 0.0]),
    cam_up: np.ndarray = np.array([0, 1, 0]),
    resolution: Tuple[int, int] = (640, 480),
):
    """
    Digital Twin 기본 씬 구성 (DomeLight + GroundPlane + top-down 카메라)

    Args:
        cam_pos: 카메라 위치
        target_pos: 카메라가 바라볼 위치
        cam_up: 카메라 업 벡터
        resolution: (width, height) 카메라 해상도
    """
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils

    cam_rot = get_lookat_quat(cam_pos, target_pos, up=cam_up)

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)

    scene_cfg.dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    scene_cfg.ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    width, height = resolution
    scene_cfg.camera = CameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=tuple(cam_pos), rot=cam_rot, convention="opengl"
        ),
    )

    return scene_cfg


def spawn_pallets(
    positions: Optional[List[Tuple[float, float, float]]] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    highlight_index: Optional[int] = None,
    highlight_color: Tuple[float, float, float] = (0.5, 0.8, 0.2),
):
    """
    물류 센터 KLT Bin 팔레트 스폰

    Args:
        positions: 팔레트별 (x, y, z) 위치 목록 (기본: 2개)
        scale: 팔레트 스케일
        highlight_index: 색상을 변경할 팔레트 인덱스 (None이면 변경 안 함)
        highlight_color: 하이라이트 색상 (기본: 연두색)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    if positions is None:
        # positions = [(-0.1, 0.1, 0.075), (0.1, 0.1, 0.075)]
        positions = [(-0.15, 0.1, 0.075)]

    for i, pos in enumerate(positions):
        prim_path = f"/World/Pallets/pallet_{i}"
        try:
            usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/KLT_Bin/small_KLT_visual_collision.usd"
            cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale)
            sim_utils.spawn_from_usd(prim_path, cfg, translation=pos)
            # 충돌 근사를 convexDecomposition으로 변경하여 내부 빈 공간 인식
            _set_concave_collision(prim_path)
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

        if highlight_index is not None and i == highlight_index:
            _apply_material(prim_path, highlight_color)


def _set_concave_collision(prim_path: str):
    """팔레트 충돌 메시를 convexDecomposition으로 변경하여 오목한 내부 공간 인식

    기본 convexHull은 내부가 꽉 찬 형태로 처리되어 물체가 안으로 들어가지 못합니다.
    convexDecomposition은 메시를 여러 볼록 조각으로 분해하여 오목한 형상을 지원합니다.
    """
    try:
        import omni.usd
        from pxr import Usd, UsdPhysics, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        changed = False
        # 모든 하위 프림을 재귀적으로 순회 (Usd.PrimRange)
        for descendant in Usd.PrimRange(prim):
            if descendant.GetTypeName() == "Mesh":
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Get(stage, descendant.GetPath())
                if not mesh_collision_api:
                    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(descendant)
                mesh_collision_api.GetApproximationAttr().Set("convexDecomposition")
                changed = True
                print(f"[INFO] 충돌 근사 변경: {descendant.GetPath()} -> convexDecomposition")

        if not changed:
            # Mesh 프림이 없는 경우, 루트 프림 자체에 적용
            if not UsdPhysics.MeshCollisionAPI.Get(stage, prim.GetPath()):
                UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set("convexDecomposition")
            print(f"[INFO] 충돌 근사 변경 (루트): {prim_path} -> convexDecomposition")
    except Exception as e:
        print(f"[WARN] 충돌 근사 변경 실패: {e}")


def _apply_material(
    prim_path: str,
    color: Tuple[float, float, float],
    roughness: float = 0.5,
    metallic: float = 0.0,
):
    """USD 프림에 PreviewSurface 재질 바인딩"""
    try:
        import omni.usd
        from pxr import UsdShade, UsdGeom
        import isaaclab.sim as sim_utils

        stage = omni.usd.get_context().get_stage()

        looks_path = "/World/Looks"
        looks_prim = stage.GetPrimAtPath(looks_path)
        if not looks_prim.IsValid():
            UsdGeom.Scope.Define(stage, looks_path)

        mat_name = prim_path.replace("/", "_").strip("_")
        material_path = f"/World/Looks/{mat_name}_material"

        material_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=color, roughness=roughness, metallic=metallic,
        )
        sim_utils.spawn_preview_surface(material_path, material_cfg)

        pallet_prim = stage.GetPrimAtPath(prim_path)
        material_prim = stage.GetPrimAtPath(material_path)

        if pallet_prim.IsValid() and material_prim.IsValid():
            material = UsdShade.Material(material_prim)
            UsdShade.MaterialBindingAPI(pallet_prim).Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            print(f"[INFO] 재질 적용 완료: {prim_path} -> {color}")
        else:
            print(f"[WARN] 재질 적용 실패: Prim을 찾을 수 없음 ({prim_path})")
    except Exception as e:
        print(f"[ERROR] 재질 적용 중 오류: {e}")
