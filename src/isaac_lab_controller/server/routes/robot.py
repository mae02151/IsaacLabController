"""
로봇 제어 API 라우트
"""

from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class JointPositionsRequest(BaseModel):
    """관절 위치 설정 요청"""
    positions: List[float]


class TeleopStartRequest(BaseModel):
    """텔레오퍼레이션 시작 요청"""
    mode: str = "demo"  # "ros2" | "demo"


@router.get("/info")
async def get_robot_info(request: Request):
    """
    로봇 정보 조회 (이름, DOF, 관절명, 관절 리미트)
    """
    robot = request.app.state.robot
    if robot is None:
        raise HTTPException(status_code=404, detail="로봇이 없습니다")
    try:
        info = robot.get_robot_info()
        return {"success": True, "data": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/joints")
async def get_joint_positions(request: Request):
    """
    현재 관절 위치 조회
    """
    robot = request.app.state.robot
    if robot is None:
        raise HTTPException(status_code=404, detail="로봇이 없습니다")
    try:
        names = robot.get_joint_names()
        positions = robot.get_joint_positions()
        return {
            "success": True,
            "data": {
                "joint_names": names,
                "positions": positions,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/joints")
async def set_joint_positions(request: Request, body: JointPositionsRequest):
    """
    관절 위치 설정 (수동 제어)
    """
    robot = request.app.state.robot
    if robot is None:
        raise HTTPException(status_code=404, detail="로봇이 없습니다")
    try:
        success = robot.set_joint_positions(body.positions)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/teleop/start")
async def start_teleop(request: Request, body: TeleopStartRequest):
    """
    텔레오퍼레이션 시작

    Args:
        mode: "ros2" (실제 로봇 미러링) 또는 "demo" (사인파 테스트)
    """
    robot = request.app.state.robot
    if robot is None:
        raise HTTPException(status_code=404, detail="로봇이 없습니다")
    try:
        success = robot.start_teleop(body.mode)
        return {"success": success, "mode": body.mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/teleop/stop")
async def stop_teleop(request: Request):
    """
    텔레오퍼레이션 중지
    """
    robot = request.app.state.robot
    if robot is None:
        raise HTTPException(status_code=404, detail="로봇이 없습니다")
    try:
        success = robot.stop_teleop()
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/teleop/status")
async def get_teleop_status(request: Request):
    """
    텔레오퍼레이션 상태 조회
    """
    robot = request.app.state.robot
    if robot is None:
        raise HTTPException(status_code=404, detail="로봇이 없습니다")
    try:
        status = robot.get_teleop_status()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
