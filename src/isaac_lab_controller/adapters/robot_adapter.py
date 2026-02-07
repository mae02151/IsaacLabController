"""
RobotAdapter - 로봇 제어 인터페이스

로봇 관절 상태 조회, 설정, 텔레오퍼레이션 제어를 위한 추상 인터페이스입니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class RobotAdapter(ABC):
    """
    로봇 제어를 위한 추상 인터페이스

    관절 상태 조회/설정, 텔레오퍼레이션 모드 제어 기능을 정의합니다.

    Example:
        >>> class MyRobotAdapter(RobotAdapter):
        ...     def __init__(self, robot):
        ...         self.robot = robot
        ...
        ...     def get_joint_names(self):
        ...         return self.robot.joint_names
    """

    @abstractmethod
    def get_joint_names(self) -> List[str]:
        """
        로봇 관절 이름 목록 반환

        Returns:
            List[str]: 관절 이름 리스트
        """
        pass

    @abstractmethod
    def get_joint_positions(self) -> List[float]:
        """
        현재 관절 위치 반환

        Returns:
            List[float]: 각 관절의 현재 위치 (라디안 또는 미터)
        """
        pass

    @abstractmethod
    def set_joint_positions(self, positions: List[float]) -> bool:
        """
        관절 위치 설정

        Args:
            positions: 설정할 관절 위치 리스트

        Returns:
            bool: 성공 여부
        """
        pass

    @abstractmethod
    def get_robot_info(self) -> Dict[str, Any]:
        """
        로봇 정보 반환

        Returns:
            dict: {
                "name": str,
                "num_joints": int,
                "joint_names": List[str],
                "joint_limits": Dict[str, tuple]
            }
        """
        pass

    @abstractmethod
    def start_teleop(self, mode: str = "demo") -> bool:
        """
        텔레오퍼레이션 시작

        Args:
            mode: 텔레오프 모드 ("ros2" | "demo")

        Returns:
            bool: 성공 여부
        """
        pass

    @abstractmethod
    def stop_teleop(self) -> bool:
        """
        텔레오퍼레이션 중지

        Returns:
            bool: 성공 여부
        """
        pass

    @abstractmethod
    def get_teleop_status(self) -> Dict[str, Any]:
        """
        텔레오퍼레이션 상태 조회

        Returns:
            dict: {
                "running": bool,
                "mode": str,
                "connected": bool
            }
        """
        pass

    @abstractmethod
    def process_commands(self) -> None:
        """
        메인 스레드에서 호출: 큐에 쌓인 명령 처리 및 텔레오프 업데이트

        시뮬레이션 루프에서 매 프레임마다 호출해야 합니다.
        """
        pass
