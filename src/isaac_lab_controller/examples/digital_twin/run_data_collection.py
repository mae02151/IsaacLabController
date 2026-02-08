"""
데이터 수집 통합 실행 스크립트 (IsaacSim + FastAPI + DataCollector)

run.py와 동일한 통합 모드에서 시뮬레이션을 돌리면서
카메라 RGB + 로봇 관절 데이터를 에피소드 단위로 HDF5에 저장합니다.

에피소드 제어 (웹 UI 버튼):
    POST /api/data/start   - 녹화 시작
    POST /api/data/stop    - 에피소드 저장 → 다음 에피소드 자동 시작
    POST /api/data/discard - 에피소드 폐기 → 새 에피소드 시작
    POST /api/data/finish  - 수집 완료 및 종료
    GET  /api/data/status  - 현재 상태 조회

사용법:
    isaaclab -p run_data_collection.py

    브라우저에서 http://localhost:8080 접속 (카메라 스트리밍 + 데이터 수집 제어)
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
from isaaclab_assets import OPEN_MANIPULATOR_X_GRIPPER_CFG

from isaac_lab_controller import ControlServer
from isaac_lab_controller.examples.digital_twin.adapters import DigitalTwinSceneAdapter
from isaac_lab_controller.examples.digital_twin.scene_setup import (
    create_simulation_cfg,
    create_scene_cfg,
    spawn_pallets,
)
from isaac_lab_controller.data_collection import DataCollector


def main():
    print("=" * 60)
    print("[DATA] 데이터 수집 모드 (통합)")
    print("[DATA] 시뮬레이션 + 서버 + 데이터 수집을 동시 실행합니다")
    print("=" * 60)

    # 1. 시뮬레이션 + 씬 설정 (로봇 포함)
    sim = SimulationContext(create_simulation_cfg())
    scene_cfg = create_scene_cfg()
    scene_cfg.robot = OPEN_MANIPULATOR_X_GRIPPER_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    scene = InteractiveScene(scene_cfg)

    # 2. 팔레트 배치
    spawn_pallets(highlight_index=1)

    sim.reset()
    print("[DATA] 시뮬레이션 초기화 완료")

    # 3. 어댑터 + 서버 시작
    adapter = DigitalTwinSceneAdapter(sim, scene, simulation_app, sim_utils)

    server_port = 8080
    server = ControlServer(adapter, port=server_port)

    # 4. 어댑터 가져오기
    camera_adapter = adapter.get_camera_adapter()
    object_adapter = adapter.get_object_adapter()
    material_adapter = adapter.get_material_adapter()
    robot_adapter = adapter.get_robot_adapter()

    # 5. DataCollector 초기화
    collector = DataCollector(
        camera_adapter=camera_adapter,
        robot_adapter=robot_adapter,
        output_dir="./collected_data",
        image_size=(224, 224),
        task_description="pick_up_object",
    )

    # 6. DataCollector를 서버 앱에 등록 (웹 UI 버튼 제어용)
    server.app.state.data_collector = collector

    # 7. 서버 시작 (백그라운드 스레드)
    server.start_background()

    print(f"[DATA] 웹 서버: http://localhost:{server_port}")
    print("")
    print("=" * 60)
    print("[DATA] 데이터 수집 준비 완료!")
    print("[DATA] 웹 UI에서 버튼으로 제어하세요:")
    print(f"  API: http://localhost:{server_port}/api/data/status")
    print("  POST /api/data/start   - 녹화 시작")
    print("  POST /api/data/stop    - 저장 → 다음 에피소드")
    print("  POST /api/data/discard - 폐기 → 새 에피소드")
    print("  POST /api/data/finish  - 수집 완료")
    print("=" * 60)
    print("")

    # 8. 시뮬레이션 루프
    step_count = 0
    while simulation_app.is_running():
        # 1. 로봇 명령 처리
        if robot_adapter and hasattr(robot_adapter, 'process_commands'):
            robot_adapter.process_commands()

        # 2. 물리 시뮬레이션 + 렌더링
        sim.step()

        # 3. 씬 업데이트 (센서 데이터 갱신)
        scene.update(sim.get_physics_dt())

        # 4. 데이터 수집 (sim.step + scene.update 직후)
        collector.record_step(sim_time=sim.current_time)

        # 5. 카메라/물체/재질 명령 처리
        if hasattr(camera_adapter, 'process_commands'):
            camera_adapter.process_commands()
        if hasattr(object_adapter, 'process_commands'):
            object_adapter.process_commands()
        if hasattr(material_adapter, 'process_commands'):
            material_adapter.process_commands()

        # 6. 프레임 캡처 (웹 스트리밍용)
        if hasattr(camera_adapter, 'update_frame'):
            camera_adapter.update_frame()

        step_count += 1

        # 주기적 상태 출력 (매 500 스텝)
        if step_count % 500 == 0:
            rec_mark = "REC" if collector.is_recording else "---"
            print(
                f"[DATA] [{rec_mark}] step={step_count}, "
                f"episode={collector.episode_index}, "
                f"frames={collector.frame_count}"
            )

    print("[DATA] 시뮬레이션 종료")
    collector.close()
    server.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
