"""
데이터 수집 모듈

시뮬레이션에서 Vision-Action 데이터를 수집하고,
HuggingFace LeRobot 호환 포맷으로 변환하는 기능을 제공합니다.
"""

from isaac_lab_controller.data_collection.data_collector import DataCollector

__all__ = ["DataCollector"]
