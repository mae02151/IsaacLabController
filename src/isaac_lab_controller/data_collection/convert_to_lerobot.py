"""
HDF5 → Hugging Face Dataset 변환 스크립트

수집된 HDF5 파일들을 Hugging Face datasets.Dataset 객체로 변환합니다.
LeRobot / Open X-Embodiment 호환 스키마를 사용합니다.

사용법:
    python convert_to_lerobot.py --input_dir ./collected_data --output_dir ./hf_dataset

    # Hub에 업로드하려면:
    python convert_to_lerobot.py --input_dir ./collected_data --push_to_hub --hub_repo user/dataset_name
"""

import argparse
import glob
import os
from typing import Generator

import h5py
import numpy as np
from PIL import Image

try:
    import datasets
except ImportError:
    raise ImportError(
        "huggingface datasets 패키지가 필요합니다. "
        "설치: pip install datasets pillow"
    )


def _generate_examples(hdf5_paths: list[str]) -> Generator[dict, None, None]:
    """HDF5 파일들에서 row 단위로 데이터를 생성하는 제너레이터.

    Args:
        hdf5_paths: HDF5 파일 경로 리스트

    Yields:
        dict: 한 프레임의 데이터 (observation, action, metadata)
    """
    global_idx = 0

    for path in sorted(hdf5_paths):
        with h5py.File(path, "r") as f:
            if "data" not in f:
                print(f"[WARN] 'data' 그룹이 없음, 스킵: {path}")
                continue

            data_group = f["data"]

            for demo_name in sorted(data_group.keys()):
                demo = data_group[demo_name]
                num_frames = demo.attrs.get("num_samples", 0)
                task_desc = demo.attrs.get("task_description", "unknown")

                if num_frames == 0:
                    continue

                # 데이터 로드
                images = demo["obs"]["image"][:]      # (T, H, W, 3) uint8
                states = demo["obs"]["state"][:]      # (T, N) float32
                actions = demo["actions"][:]           # (T, M) float32
                episode_indices = demo["episode_index"][:]  # (T,) int32
                frame_indices = demo["frame_index"][:]      # (T,) int32
                timestamps = demo["timestamp"][:]           # (T,) float64

                for i in range(num_frames):
                    # PIL Image로 변환 (datasets.Image() 타입 호환)
                    pil_image = Image.fromarray(images[i])

                    yield {
                        "observation.image": pil_image,
                        "observation.state": states[i].tolist(),
                        "action": actions[i].tolist(),
                        "episode_index": int(episode_indices[i]),
                        "frame_index": int(frame_indices[i]),
                        "timestamp": float(timestamps[i]),
                        "task_description": str(task_desc),
                    }
                    global_idx += 1

                print(f"  변환: {path}:{demo_name} ({num_frames} 프레임)")


def convert_hdf5_to_hf_dataset(
    input_dir: str,
    output_dir: str | None = None,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
) -> datasets.Dataset:
    """HDF5 파일들을 Hugging Face Dataset으로 변환합니다.

    Args:
        input_dir: HDF5 파일들이 있는 디렉토리
        output_dir: 로컬 저장 경로 (None이면 저장 안함)
        push_to_hub: Hub 업로드 여부
        hub_repo: Hub 레포지토리 이름 (예: "user/dataset_name")

    Returns:
        datasets.Dataset: 변환된 데이터셋
    """
    # 1. HDF5 파일 검색
    hdf5_paths = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    if not hdf5_paths:
        raise FileNotFoundError(f"HDF5 파일이 없습니다: {input_dir}")

    print(f"[변환] 입력 디렉토리: {input_dir}")
    print(f"[변환] HDF5 파일 {len(hdf5_paths)}개 발견")

    # 2. 첫 번째 파일에서 state/action 차원 확인
    with h5py.File(hdf5_paths[0], "r") as f:
        first_demo = list(f["data"].keys())[0]
        state_dim = f["data"][first_demo]["obs"]["state"].shape[1]
        action_dim = f["data"][first_demo]["actions"].shape[1]
        img_shape = f["data"][first_demo]["obs"]["image"].shape[1:]
        print(f"[변환] 이미지: {img_shape}, state: {state_dim}D, action: {action_dim}D")

    # 3. Features 정의 (LeRobot 호환)
    features = datasets.Features({
        "observation.image": datasets.Image(),
        "observation.state": datasets.Sequence(
            datasets.Value("float32"), length=state_dim
        ),
        "action": datasets.Sequence(
            datasets.Value("float32"), length=action_dim
        ),
        "episode_index": datasets.Value("int32"),
        "frame_index": datasets.Value("int32"),
        "timestamp": datasets.Value("float64"),
        "task_description": datasets.Value("string"),
    })

    # 4. Dataset 생성 (제너레이터 기반 - 메모리 효율적)
    print("[변환] 데이터셋 생성 중...")
    ds = datasets.Dataset.from_generator(
        lambda: _generate_examples(hdf5_paths),
        features=features,
    )

    print(f"[변환] 총 {len(ds)} 프레임 변환 완료")

    # 5. 로컬 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ds.save_to_disk(output_dir)
        print(f"[변환] 저장 완료: {output_dir}")

    # 6. Hub 업로드
    if push_to_hub and hub_repo:
        ds.push_to_hub(hub_repo)
        print(f"[변환] Hub 업로드 완료: {hub_repo}")

    return ds


def main():
    parser = argparse.ArgumentParser(
        description="HDF5 데이터셋을 Hugging Face Dataset으로 변환"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./collected_data",
        help="HDF5 파일들이 있는 디렉토리",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hf_dataset",
        help="변환된 데이터셋 저장 경로",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Hugging Face Hub에 업로드",
    )
    parser.add_argument(
        "--hub_repo",
        type=str,
        default=None,
        help="Hub 레포지토리 이름 (예: user/dataset_name)",
    )
    args = parser.parse_args()

    ds = convert_hdf5_to_hf_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )

    # 데이터셋 정보 출력
    print("\n" + "=" * 60)
    print("[결과] 데이터셋 정보:")
    print(ds)
    print(f"\n[결과] Features:")
    for name, feature in ds.features.items():
        print(f"  {name}: {feature}")
    print("=" * 60)


if __name__ == "__main__":
    main()
