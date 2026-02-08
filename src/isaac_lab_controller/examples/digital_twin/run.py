"""
Digital Twin 통합 실행 스크립트 (IsaacSim + FastAPI 단일 프로세스)

ZMQ 통신 없이 IsaacSim과 백엔드 서버를 한 프로세스에서 실행합니다.
서버는 백그라운드 스레드에서, 시뮬레이션은 메인 스레드에서 동작합니다.

사용법:
    isaaclab -p run.py

    브라우저에서 http://localhost:8080 접속
"""

# IsaacLab 앱 실행 (가장 먼저!)
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(launcher_args={
    "headless": False,
    "enable_cameras": True,
})
simulation_app = app_launcher.app

# 이후 import (IsaacLab 앱 실행 후에만 가능)
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils

from isaac_lab_controller import ControlServer
from isaac_lab_controller.examples.digital_twin.adapters import DigitalTwinSceneAdapter
from isaac_lab_controller.examples.digital_twin.scene_setup import (
    create_simulation_cfg,
    create_scene_cfg,
    spawn_pallets,
)
from isaac_lab_controller.examples.digital_twin.sim_loop import run_simulation_loop


def main():
    print("=" * 60)
    print("[INFO] IsaacLabController 통합 모드 시작")
    print("[INFO] ZMQ 없이 단일 프로세스로 실행합니다")
    print("=" * 60)

    # 1. 시뮬레이션 + 씬 설정
    sim = SimulationContext(create_simulation_cfg())
    scene = InteractiveScene(create_scene_cfg())

    # 2. 팔레트 배치 (pallet_1에 연두색 적용)
    spawn_pallets(highlight_index=1)

    sim.reset()
    print("[INFO] 시뮬레이션 초기화 완료")

    # 3. 어댑터 + 서버 시작
    adapter = DigitalTwinSceneAdapter(sim, scene, simulation_app, sim_utils)

    server_port = 8080
    server = ControlServer(adapter, port=server_port)
    server.start_background()

    print(f"\n{'=' * 60}")
    print(f"[INFO] 웹 서버 시작: http://localhost:{server_port}")
    print("[INFO] 브라우저에서 접속하여 시뮬레이션을 제어하세요!")
    print(f"{'=' * 60}\n")

    # 4. 시뮬레이션 루프
    run_simulation_loop(sim, scene, simulation_app, adapter)

    print("[INFO] 시뮬레이션 종료")
    server.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
