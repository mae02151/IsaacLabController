"""
강화학습(RL) 제어 API 라우트
"""

from typing import Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class TrainRequest(BaseModel):
    """학습 시작 요청"""
    num_envs: int = 64


@router.post("/train/start")
async def start_train(request: Request, body: TrainRequest):
    """
    학습 시작

    Args:
        num_envs: 동시에 학습할 환경(로봇) 수
    """
    try:
        rl_state = getattr(request.app.state, "rl", None)
        if rl_state is None:
            request.app.state.rl = {"mode": "train", "num_envs": body.num_envs, "running": True}
        else:
            rl_state["mode"] = "train"
            rl_state["num_envs"] = body.num_envs
            rl_state["running"] = True

        return {"success": True, "mode": "train", "num_envs": body.num_envs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/stop")
async def stop_train(request: Request):
    """학습 중지"""
    try:
        rl_state = getattr(request.app.state, "rl", None)
        if rl_state:
            rl_state["running"] = False
            rl_state["mode"] = None

        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/start")
async def start_inference(request: Request):
    """추론 시작"""
    try:
        rl_state = getattr(request.app.state, "rl", None)
        if rl_state is None:
            request.app.state.rl = {"mode": "inference", "num_envs": 1, "running": True}
        else:
            rl_state["mode"] = "inference"
            rl_state["running"] = True

        return {"success": True, "mode": "inference"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/stop")
async def stop_inference(request: Request):
    """추론 중지"""
    try:
        rl_state = getattr(request.app.state, "rl", None)
        if rl_state:
            rl_state["running"] = False
            rl_state["mode"] = None

        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_rl_status(request: Request):
    """현재 RL 상태 조회"""
    try:
        rl_state = getattr(request.app.state, "rl", None)
        if rl_state is None:
            return {"success": True, "data": {"mode": None, "running": False, "num_envs": 0}}

        return {"success": True, "data": rl_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
