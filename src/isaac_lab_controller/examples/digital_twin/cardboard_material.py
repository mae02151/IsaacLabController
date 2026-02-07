"""
사실적인 택배 박스(카드보드) 스폰 및 재질 관리

1순위: NVIDIA Nucleus의 텍스처가 입혀진 USD 박스 에셋 사용
2순위: PreviewSurface 큐보이드 (roughness/metallic 최적화) 폴백

사용법:
    from .cardboard_material import CardboardMaterial

    config = CardboardMaterial.generate_config()
    CardboardMaterial.spawn(prim_path, sim_utils, config)
"""

import random
from typing import Dict, Any, Optional, Tuple


class CardboardMaterial:
    """
    사실적인 택배 박스 스폰 매니저

    NVIDIA Nucleus의 텍스처 USD 에셋을 우선 사용하고,
    접근 불가 시 PreviewSurface 큐보이드로 폴백합니다.
    """

    # 갈색 계열 카드보드 색상 팔레트 (폴백용)
    COLORS = [
        (0.72, 0.53, 0.35),
        (0.65, 0.45, 0.28),
        (0.78, 0.60, 0.40),
        (0.58, 0.42, 0.25),
        (0.70, 0.55, 0.38),
        (0.82, 0.65, 0.45),
    ]

    ROUGHNESS_RANGE = (0.85, 0.95)
    METALLIC = 0.0

    # NVIDIA Nucleus 박스 에셋 (텍스처 포함, 확인됨)
    _NUCLEUS_BOX_ASSETS = [
        "Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
    ]

    @classmethod
    def generate_config(cls) -> Dict[str, Any]:
        """랜덤 택배 박스 설정 생성"""
        width = random.uniform(0.15, 0.5)
        depth = random.uniform(0.1, 0.4)
        height = random.uniform(0.1, 0.35)

        volume = width * depth * height
        mass = volume * 100

        return {
            "size": (width, depth, height),
            "color": random.choice(cls.COLORS),
            "roughness": random.uniform(*cls.ROUGHNESS_RANGE),
            "mass": max(0.1, mass),
        }

    @classmethod
    def spawn(
        cls,
        prim_path: str,
        sim_utils,
        config: Dict[str, Any],
        translation: Optional[Tuple[float, float, float]] = None,
        orientation: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        택배 박스 스폰 (USD 에셋 우선, 폴백 큐보이드)

        Args:
            prim_path: USD 경로
            sim_utils: isaaclab.sim 모듈
            config: generate_config()로 생성된 설정
            translation: 스폰 위치 (x, y, z)
            orientation: 스폰 회전 (w, x, y, z)
        """
        # 1순위: NVIDIA Nucleus 텍스처 USD 에셋
        if cls._try_spawn_usd_asset(prim_path, sim_utils, config, translation, orientation):
            return

        # 2순위: PreviewSurface 큐보이드 (roughness 최적화)
        cls._spawn_cuboid_fallback(prim_path, sim_utils, config, translation, orientation)

    @classmethod
    def _try_spawn_usd_asset(cls, prim_path: str, sim_utils, config, translation=None, orientation=None) -> bool:
        """NVIDIA Nucleus USD 박스 에셋으로 스폰 시도"""
        try:
            from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

            asset = random.choice(cls._NUCLEUS_BOX_ASSETS)
            usd_path = f"{ISAAC_NUCLEUS_DIR}/{asset}"

            # 원하는 크기에 맞춰 균일 스케일
            scale_factor = random.uniform(1.5, 3.0)

            cfg = sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(scale_factor, scale_factor, scale_factor),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=config["mass"]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            sim_utils.spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation)

            print(
                f"[CardboardMaterial] USD 에셋 스폰: {prim_path} "
                f"({asset}, scale={scale_factor:.1f}x)"
            )
            return True

        except Exception as e:
            print(f"[CardboardMaterial] USD 에셋 실패, 폴백 사용: {e}")
            return False

    @classmethod
    def _spawn_cuboid_fallback(cls, prim_path: str, sim_utils, config, translation=None, orientation=None) -> None:
        """PreviewSurface 큐보이드 폴백 (spawn 시 roughness/metallic 설정)"""
        cfg = sim_utils.CuboidCfg(
            size=config["size"],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=config["mass"]),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=config["color"],
                roughness=config.get("roughness", 0.92),
                metallic=cls.METALLIC,
            ),
        )
        sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation)

        print(
            f"[CardboardMaterial] 큐보이드 폴백 스폰: {prim_path} "
            f"(roughness={config.get('roughness', 0.92):.2f})"
        )
