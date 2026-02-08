"""
RL 환경 프레임 캡처 모듈

gymnasium.make()를 패치하여 RL 환경의 step()마다 주기적으로 프레임을 캡처합니다.
캡처된 프레임은 JPEG로 인코딩되어 임시 파일에 저장됩니다.
server_rl.py가 이 파일을 읽어 웹으로 스트리밍합니다.

원본 train.py/play.py를 수정하지 않고 프레임 캡처를 추가합니다.

환경변수:
    RL_FRAME_PATH: 프레임 저장 경로 (기본: /tmp/isaaclab_rl_frame.jpg)
    RL_FRAME_INTERVAL: 캡처 간격 (N 스텝마다, 기본: 10)
    RL_CAMERA_PARAMS_PATH: 카메라 파라미터 JSON 경로 (기본: /tmp/isaaclab_rl_camera.json)
"""

import json
import math
import os

FRAME_PATH = os.environ.get("RL_FRAME_PATH", "/tmp/isaaclab_rl_frame.jpg")
CAPTURE_INTERVAL = int(os.environ.get("RL_FRAME_INTERVAL", "10"))
CAMERA_PARAMS_PATH = os.environ.get("RL_CAMERA_PARAMS_PATH", "/tmp/isaaclab_rl_camera.json")

_installed = False
_last_camera_mtime = 0.0


def install():
    """
    gymnasium.make를 패치하여 프레임 캡처 기능을 추가합니다.

    이 함수는 원본 RL 스크립트(train.py/play.py) 실행 전에 호출되어야 합니다.
    rl_launcher.py에서 자동으로 호출됩니다.
    """
    global _installed
    if _installed:
        return
    _installed = True

    import gymnasium as gym

    _original_make = gym.make

    def patched_make(*args, **kwargs):
        # render_mode가 설정되지 않았으면 rgb_array로 강제 설정
        if kwargs.get("render_mode") is None:
            kwargs["render_mode"] = "rgb_array"

        env = _original_make(*args, **kwargs)

        # step 함수를 감싸서 프레임 캡처 추가
        original_step = env.step
        step_counter = [0]

        def capturing_step(action):
            result = original_step(action)
            step_counter[0] += 1
            if step_counter[0] % CAPTURE_INTERVAL == 0:
                _apply_camera_params()
                _try_capture_frame(env)
            return result

        env.step = capturing_step

        print(f"[FrameCapture] 프레임 캡처 활성화 (간격: {CAPTURE_INTERVAL}스텝, 경로: {FRAME_PATH})")
        print(f"[FrameCapture] 카메라 파라미터 경로: {CAMERA_PARAMS_PATH}")
        return env

    gym.make = patched_make


def _apply_camera_params():
    """카메라 파라미터 파일을 읽어 뷰포트 카메라에 적용"""
    global _last_camera_mtime

    try:
        if not os.path.exists(CAMERA_PARAMS_PATH):
            return

        # 파일이 변경되었을 때만 적용 (성능 최적화)
        mtime = os.path.getmtime(CAMERA_PARAMS_PATH)
        if mtime == _last_camera_mtime:
            return
        _last_camera_mtime = mtime

        with open(CAMERA_PARAMS_PATH) as f:
            params = json.load(f)

        azimuth = params.get("azimuth", 45.0)
        elevation = params.get("elevation", 30.0)
        distance = params.get("distance", 5.0)
        target = params.get("target", [0.0, 0.0, 0.5])

        # 구면 좌표 → 직교 좌표 변환
        az_rad = math.radians(azimuth)
        el_rad = math.radians(elevation)
        eye_x = target[0] + distance * math.cos(el_rad) * math.cos(az_rad)
        eye_y = target[1] + distance * math.cos(el_rad) * math.sin(az_rad)
        eye_z = target[2] + distance * math.sin(el_rad)

        import numpy as np
        from omni.isaac.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=np.array([eye_x, eye_y, eye_z]),
            target=np.array(target),
        )
    except Exception:
        pass


def _try_capture_frame(env):
    """프레임 캡처 시도 (실패해도 학습에 영향 없음)"""
    try:
        frame = env.render()
        if frame is not None:
            _save_frame(frame)
    except Exception:
        pass


def _save_frame(frame):
    """프레임을 JPEG로 인코딩하여 파일에 저장"""
    try:
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return

        # OpenCV 사용 (IsaacLab 환경에 포함됨)
        import cv2

        if frame.ndim == 3 and frame.shape[2] == 4:
            # RGBA → BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # RGB → BGR
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            return

        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

        # 원자적 쓰기 (임시 파일 → 이름 변경)
        tmp_path = FRAME_PATH + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp_path, FRAME_PATH)
    except Exception:
        pass
