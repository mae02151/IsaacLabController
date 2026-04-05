"""
Digital Twin 시뮬레이션 루프

매 프레임: sim.step → scene.update → process_commands → on_step → update_frame
"""

from typing import Optional, Callable


def run_simulation_loop(
    sim,
    scene,
    simulation_app,
    adapter,
    on_step: Optional[Callable] = None,
):
    """
    표준 시뮬레이션 루프 실행

    Args:
        sim: SimulationContext 인스턴스
        scene: InteractiveScene 인스턴스
        simulation_app: IsaacSim 앱 인스턴스
        adapter: DigitalTwinSceneAdapter 인스턴스
        on_step: 매 프레임 추가 처리 콜백 (예: bridge.process_messages)
    """
    camera_adapter = adapter.get_camera_adapter()
    object_adapter = adapter.get_object_adapter()
    material_adapter = adapter.get_material_adapter()

    while simulation_app.is_running():
        sim.step()
        scene.update(sim.get_physics_dt())

        camera_adapter.process_commands()
        object_adapter.process_commands()
        material_adapter.process_commands()

        if on_step is not None:
            on_step()

        camera_adapter.update_frame()
