"""
DataCollector - Vision-Action 데이터 수집기

시뮬레이션 루프에 훅(hook)처럼 연결하여 카메라 RGB 이미지와
로봇 관절 상태/액션 데이터를 에피소드 단위로 HDF5에 저장합니다.

사용법:
    collector = DataCollector(camera_adapter, robot_adapter, output_dir="./data")
    collector.start_episode()

    while running:
        sim.step()
        scene.update(dt)
        collector.record_step(sim_time=sim.current_time)

    collector.end_episode(success=True)
    collector.close()
"""

import os
import time
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np


class DataCollector:
    """시뮬레이션에서 Vision-Action 데이터를 수집하여 HDF5로 저장하는 클래스.

    카메라 어댑터에서 RGB 이미지를, 로봇 어댑터에서 관절 상태를 읽어
    에피소드 단위로 메모리 버퍼에 저장한 후, 에피소드 종료 시 HDF5로 내보냅니다.

    Args:
        camera_adapter: DigitalTwinCameraAdapter 인스턴스 (camera.data.output["rgb"] 접근)
        robot_adapter: DigitalTwinRobotAdapter 인스턴스 (get_joint_positions() 사용)
        output_dir: HDF5 파일 저장 경로
        image_size: 리사이즈할 이미지 해상도 (width, height)
        task_description: 태스크 설명 문자열 (고정값)
        dataset_filename: HDF5 파일 이름 (확장자 제외)
    """

    def __init__(
        self,
        camera_adapter,
        robot_adapter,
        output_dir: str = "./collected_data",
        image_size: Tuple[int, int] = (224, 224),
        task_description: str = "pick_up_object",
        dataset_filename: str = "dataset",
    ):
        self._camera_adapter = camera_adapter
        self._robot_adapter = robot_adapter  # None 가능 (로봇 없는 씬)
        self._output_dir = output_dir
        self._image_size = image_size  # (width, height)
        self._task_description = task_description
        self._dataset_filename = dataset_filename

        # 에피소드 카운터
        self._episode_index = 0
        self._frame_index = 0
        self._recording = False

        # 에피소드 버퍼
        self._images: List[np.ndarray] = []
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._timestamps: List[float] = []

        # 이전 관절 상태 (액션 계산용)
        self._prev_joint_positions: Optional[np.ndarray] = None

        # HDF5 파일 핸들러
        self._hdf5_file: Optional[h5py.File] = None

        # 출력 디렉토리 생성 (절대경로로 변환)
        self._output_dir = os.path.abspath(self._output_dir)
        os.makedirs(self._output_dir, exist_ok=True)

        # HDF5 파일 경로
        self._hdf5_path = os.path.join(self._output_dir, f"{self._dataset_filename}.hdf5")

        # HDF5 파일 초기화
        self._init_hdf5()

        has_robot = self._robot_adapter is not None
        print("")
        print("=" * 60)
        print("[DataCollector] 초기화 완료")
        print(f"  HDF5 저장 경로: {self._hdf5_path}")
        print(f"  이미지 크기: {self._image_size}")
        print(f"  로봇: {'연결됨' if has_robot else '없음 (이미지만 수집)'}")
        print(f"  태스크: {self._task_description}")
        print("=" * 60)
        print("")

    def _init_hdf5(self):
        """HDF5 파일 생성 및 초기화."""
        self._hdf5_file = h5py.File(self._hdf5_path, "w")
        self._data_group = self._hdf5_file.create_group("data")
        self._data_group.attrs["total"] = 0
        self._data_group.attrs["env_args"] = '{"env_name": "digital_twin", "type": 2}'

    def start_episode(self):
        """새 에피소드 수집 시작. 버퍼를 초기화합니다."""
        self._images.clear()
        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._frame_index = 0
        self._prev_joint_positions = None
        self._recording = True
        print("")
        print(">> ======== REC [EP %d] 녹화 시작 ========" % self._episode_index)

    def record_step(self, sim_time: float = 0.0):
        """현재 프레임의 데이터를 버퍼에 기록합니다.

        sim.step() + scene.update() 직후에 호출해야 합니다.

        Args:
            sim_time: 현재 시뮬레이션 시간 (초)
        """
        if not self._recording:
            return

        try:
            # 1. RGB 이미지 캡처 (CUDA → CPU → numpy)
            rgb_tensor = self._camera_adapter.camera.data.output["rgb"][0]  # (H, W, 4) RGBA
            rgb_np = rgb_tensor.cpu().numpy().astype(np.uint8)
            rgb_np = rgb_np[:, :, :3]  # RGBA → RGB

            # 리사이즈
            w, h = self._image_size
            rgb_resized = cv2.resize(rgb_np, (w, h), interpolation=cv2.INTER_AREA)

            # 2. 로봇 관절 위치 읽기
            if self._robot_adapter is not None:
                joint_positions = self._robot_adapter.get_joint_positions()
                joints = np.array(joint_positions, dtype=np.float32)
            else:
                joints = np.zeros(0, dtype=np.float32)

            # 3. 액션 = 현재 관절 위치 그대로 [j1, j2, j3, j4, gripper]
            action = joints.copy()

            # 4. 상태 = 관절 위치 + 가우시안 노이즈 (관측 노이즈 시뮬레이션)
            #    관절별 노이즈 표준편차: arm 0.01 rad, gripper 0.001 rad
            if len(joints) > 0:
                noise_std = np.array([0.01, 0.01, 0.01, 0.01, 0.001], dtype=np.float32)
                noise = np.random.normal(0.0, noise_std).astype(np.float32)
                state = joints + noise
            else:
                state = joints

            # 4. 버퍼에 추가
            self._images.append(rgb_resized)
            self._states.append(state)
            self._actions.append(action)
            self._timestamps.append(sim_time)

            self._frame_index += 1

        except Exception as e:
            print(f"[DataCollector] record_step 오류: {e}")

    def end_episode(self, success: bool = True) -> bool:
        """현재 에피소드를 HDF5에 저장하고 버퍼를 초기화합니다.

        Args:
            success: 에피소드 성공 여부

        Returns:
            bool: 저장 성공 여부
        """
        self._recording = False

        if len(self._images) == 0:
            print(f"[DataCollector] 에피소드 {self._episode_index}: 데이터 없음, 스킵")
            return False

        try:
            num_frames = len(self._images)
            demo_name = f"demo_{self._episode_index}"

            # HDF5 그룹 생성
            ep_group = self._data_group.create_group(demo_name)
            ep_group.attrs["num_samples"] = num_frames
            ep_group.attrs["success"] = success
            ep_group.attrs["task_description"] = self._task_description

            # obs 그룹
            obs_group = ep_group.create_group("obs")

            # 이미지 저장 (T, H, W, 3) uint8
            images_array = np.stack(self._images, axis=0)
            obs_group.create_dataset("image", data=images_array, compression="gzip", compression_opts=4)

            # 관절 상태 저장 (T, N) float32
            states_array = np.stack(self._states, axis=0)
            obs_group.create_dataset("state", data=states_array, compression="gzip")

            # 액션 저장 (T, M) float32
            actions_array = np.stack(self._actions, axis=0)
            ep_group.create_dataset("actions", data=actions_array, compression="gzip")

            # 메타데이터 저장
            ep_group.create_dataset(
                "episode_index",
                data=np.full(num_frames, self._episode_index, dtype=np.int32),
            )
            ep_group.create_dataset(
                "frame_index",
                data=np.arange(num_frames, dtype=np.int32),
            )
            ep_group.create_dataset(
                "timestamp",
                data=np.array(self._timestamps, dtype=np.float64),
            )

            # 전체 스텝 수 업데이트
            self._data_group.attrs["total"] += num_frames

            # 디스크에 플러시
            self._hdf5_file.flush()

            print(">> ======== STOP [EP %d] 녹화 종료 ========" % self._episode_index)
            print(">>   저장: %d 프레임 -> %s" % (num_frames, self._hdf5_path))
            print(">>   총 누적: %d 에피소드" % (self._episode_index + 1))
            print("")

            self._episode_index += 1
            return True

        except Exception as e:
            print(f"[DataCollector] end_episode 오류: {e}")
            import traceback
            traceback.print_exc()
            return False

    def discard_episode(self):
        """현재 에피소드 데이터를 버리고 새로 시작합니다."""
        num_frames = len(self._images)
        self._recording = False
        self._images.clear()
        self._states.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._frame_index = 0
        self._prev_joint_positions = None
        print("")
        print(">> ======== DISCARD [EP %d] 폐기 (%d 프레임) ========" % (self._episode_index, num_frames))
        print("")

    def close(self):
        """HDF5 파일을 닫고 리소스를 정리합니다."""
        if self._hdf5_file is not None:
            total_demos = len(self._data_group)
            total_frames = self._data_group.attrs.get("total", 0)
            self._hdf5_file.close()
            self._hdf5_file = None
            print("")
            print("=" * 60)
            print("[DataCollector] 수집 완료!")
            print(f"  총 에피소드: {total_demos}")
            print(f"  총 프레임:   {total_frames}")
            print(f"  저장 위치:   {self._hdf5_path}")
            print("=" * 60)
            print("")

    @property
    def is_recording(self) -> bool:
        """현재 수집 중인지 여부."""
        return self._recording

    @property
    def episode_index(self) -> int:
        """현재 에피소드 번호."""
        return self._episode_index

    @property
    def frame_count(self) -> int:
        """현재 에피소드의 수집된 프레임 수."""
        return len(self._images)
