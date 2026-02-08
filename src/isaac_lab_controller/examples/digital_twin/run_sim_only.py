"""
IsaacLab 시뮬레이션 전용 실행 스크립트 (서버 분리 모드)

이 스크립트는 IsaacLab 시뮬레이션만 실행하고,
ZMQ를 통해 별도 프로세스의 FastAPI 서버와 통신합니다.

사용법:
    # 터미널 1: 시뮬레이션 (한 번만 실행, 계속 켜둠)
    isaaclab -p run_sim_only.py

    # 터미널 2: 서버 (수정 후 재시작)
    python server_standalone.py
"""

# IsaacLab 앱 실행 (가장 먼저!)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(launcher_args={
    "headless": False,
    "enable_cameras": True,
})
simulation_app = app_launcher.app

# 이후 import
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils

from isaac_lab_controller.ipc import SimulationBridge
from isaac_lab_controller.examples.digital_twin.adapters import DigitalTwinSceneAdapter
from isaac_lab_controller.examples.digital_twin.scene_setup import (
    create_simulation_cfg,
    create_scene_cfg,
    spawn_pallets,
)
from isaac_lab_controller.examples.digital_twin.sim_loop import run_simulation_loop


def main():
    print("=" * 60)
    print("[SIM] IsaacLab 시뮬레이션 시작 (서버 분리 모드)")
    print("=" * 60)

    # 1. 시뮬레이션 + 씬 설정
    sim = SimulationContext(create_simulation_cfg())
    scene = InteractiveScene(create_scene_cfg())

    # 2. 팔레트 배치
    spawn_pallets(positions=[(-0.3, 0.1, 0.0), (0.3, 0.1, 0.0)])

    sim.reset()
    print("[SIM] 시뮬레이션 초기화 완료")

    # 3. 어댑터 + ZMQ 브릿지
    adapter = DigitalTwinSceneAdapter(sim, scene, simulation_app, sim_utils)

    bridge = SimulationBridge(port=5555)
    bridge.register_adapter("camera", adapter.get_camera_adapter())
    bridge.register_adapter("object", adapter.get_object_adapter())
    bridge.register_adapter("material", adapter.get_material_adapter())
    bridge.register_adapter("scene", adapter)

    print(f"\n{'=' * 60}")
    print("[SIM] ZMQ 브릿지 대기 중: tcp://*:5555")
    print("[SIM] 이제 다른 터미널에서 server_standalone.py를 실행하세요!")
    print(f"{'=' * 60}\n")

    # 4. 시뮬레이션 루프 (ZMQ 메시지 처리 추가)
    run_simulation_loop(
        sim, scene, simulation_app, adapter,
        on_step=bridge.process_messages,
    )

    print("[SIM] 시뮬레이션 종료")
    bridge.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
