"""
RL 스크립트 런처 (프레임 캡처 지원)

원본 train.py/play.py를 수정하지 않고 프레임 캡처 기능을 주입합니다.
server_rl.py에서 서브프로세스로 실행됩니다.

사용법:
    python rl_launcher.py <script_path> [script_args...]

예시:
    python rl_launcher.py ~/IsaacLab/scripts/.../train.py --task=Isaac-Reach-OpenManipulatorX-v0 --num_envs=256
    python rl_launcher.py ~/IsaacLab/scripts/.../play.py --task=Isaac-Reach-OpenManipulatorX-v0 --num_envs=1 --hardware
"""

import os
import sys


def main():
    if len(sys.argv) < 2:
        print("사용법: python rl_launcher.py <script_path> [args...]")
        sys.exit(1)

    # 대상 스크립트 경로
    script_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(script_path):
        print(f"[Launcher] 스크립트를 찾을 수 없습니다: {script_path}")
        sys.exit(1)

    # 프레임 캡처 모듈 설치
    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, launcher_dir)
    import frame_capture

    frame_capture.install()
    print(f"[Launcher] 프레임 캡처 모듈 설치 완료")

    # 대상 스크립트 디렉토리를 sys.path에 추가 (cli_args 등 로컬 import용)
    script_dir = os.path.dirname(script_path)
    sys.path.insert(0, script_dir)

    # sys.argv를 원본 스크립트 기준으로 재설정
    sys.argv = sys.argv[1:]

    # 원본 스크립트 실행
    print(f"[Launcher] 실행: {script_path}")
    print(f"[Launcher] 인자: {sys.argv[1:]}")

    with open(script_path) as f:
        code = compile(f.read(), script_path, "exec")
        exec(code, {"__name__": "__main__", "__file__": script_path})


if __name__ == "__main__":
    main()
