"""
WebSocket 핸들러 - 실시간 스트리밍 및 제어
"""

import asyncio
import logging
from typing import Set, Optional
import json

from fastapi import WebSocket, WebSocketDisconnect

from isaac_lab_controller.adapters.base import SceneAdapter

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    WebSocket 연결 관리 및 실시간 스트리밍
    
    기능:
    - 카메라 프레임 스트리밍
    - 실시간 제어 명령 수신
    """
    
    def __init__(self, scene_adapter: SceneAdapter):
        self.scene = scene_adapter
        self._connections: Set[WebSocket] = set()
        self._streaming_enabled = True
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """새 WebSocket 연결 수락"""
        await websocket.accept()
        self._connections.add(websocket)
        logger.info(f"WebSocket 연결됨. 총 연결: {len(self._connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """WebSocket 연결 해제"""
        self._connections.discard(websocket)
        logger.info(f"WebSocket 연결 해제됨. 총 연결: {len(self._connections)}")
    
    def broadcast_frame(self) -> None:
        """
        모든 연결된 클라이언트에 프레임 전송 (동기 호출용)
        
        시뮬레이션 루프에서 호출됩니다.
        """
        if not self._connections or not self._streaming_enabled:
            return
        
        try:
            camera = self.scene.get_camera_adapter()
            frame_bytes = camera.get_frame(format="jpeg")
            
            # 큐가 가득 차면 오래된 프레임 버림
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            
            self._frame_queue.put_nowait(frame_bytes)
        except Exception as e:
            logger.warning(f"프레임 캡처 실패: {e}")
    
    async def stream_handler(self, websocket: WebSocket) -> None:
        """
        스트리밍 WebSocket 핸들러
        
        /ws/stream 엔드포인트에서 사용
        """
        await self.connect(websocket)
        
        try:
            while True:
                # 새 프레임 대기
                try:
                    frame_bytes = await asyncio.wait_for(
                        self._frame_queue.get(), 
                        timeout=1.0
                    )
                    await websocket.send_bytes(frame_bytes)
                except asyncio.TimeoutError:
                    # 타임아웃 시 핑 전송
                    await websocket.send_text('{"type": "ping"}')
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"스트리밍 오류: {e}")
        finally:
            await self.disconnect(websocket)
    
    async def control_handler(self, websocket: WebSocket) -> None:
        """
        제어 명령 WebSocket 핸들러
        
        /ws/control 엔드포인트에서 사용
        
        수신 메시지 형식:
        {
            "action": "camera/lookat",
            "data": {"eye": [1, 0, 1], "target": [0, 0, 0]}
        }
        """
        await self.connect(websocket)
        
        try:
            while True:
                message = await websocket.receive_text()
                response = await self._handle_control_message(message)
                await websocket.send_text(json.dumps(response))
                
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error(f"제어 핸들러 오류: {e}")
        finally:
            await self.disconnect(websocket)
    
    async def _handle_control_message(self, message: str) -> dict:
        """제어 메시지 처리"""
        try:
            data = json.loads(message)
            action = data.get("action", "")
            payload = data.get("data", {})
            
            # 카메라 제어
            if action == "camera/lookat":
                camera = self.scene.get_camera_adapter()
                success = camera.set_lookat(
                    eye=payload.get("eye"),
                    target=payload.get("target")
                )
                return {"success": success, "action": action}
            
            elif action == "camera/orbit":
                camera = self.scene.get_camera_adapter()
                success = camera.orbit(
                    azimuth=payload.get("azimuth", 0),
                    elevation=payload.get("elevation", 30),
                    distance=payload.get("distance", 2),
                    target=payload.get("target")
                )
                return {"success": success, "action": action}
            
            elif action == "camera/pose":
                camera = self.scene.get_camera_adapter()
                success = camera.set_pose(
                    position=payload.get("position"),
                    orientation=payload.get("orientation"),
                    convention=payload.get("convention", "ros")
                )
                return {"success": success, "action": action}
            
            # 스트리밍 제어
            elif action == "streaming/toggle":
                self._streaming_enabled = payload.get("enabled", True)
                return {"success": True, "streaming_enabled": self._streaming_enabled}
            
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except json.JSONDecodeError:
            return {"success": False, "error": "Invalid JSON"}
        except Exception as e:
            return {"success": False, "error": str(e)}
