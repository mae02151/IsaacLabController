"""
ZMQ Bridge for IPC between IsaacLab and FastAPI Server

SimulationBridge: IsaacLab 측에서 실행 (REP 소켓)
ServerBridge: FastAPI 측에서 실행 (REQ 소켓)
"""

import json
import zmq
import base64
import threading
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, asdict, is_dataclass


def serialize_data(data: Any) -> Any:
    """
    데이터를 JSON 직렬화 가능한 형태로 변환
    
    - dataclass → dict
    - list of dataclass → list of dict
    - bytes → base64 string (별도 처리)
    """
    if data is None:
        return None
    
    if is_dataclass(data) and not isinstance(data, type):
        return asdict(data)
    
    if isinstance(data, list):
        return [serialize_data(item) for item in data]
    
    if isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    
    # 기본 타입 (int, float, str, bool)
    return data


@dataclass
class IPCMessage:
    """IPC 메시지 구조"""
    adapter: str  # "camera", "object", "material", "scene"
    method: str   # 호출할 메서드명
    args: list = None
    kwargs: dict = None
    
    def to_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "method": self.method,
            "args": self.args or [],
            "kwargs": self.kwargs or {}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "IPCMessage":
        return cls(
            adapter=data["adapter"],
            method=data["method"],
            args=data.get("args", []),
            kwargs=data.get("kwargs", {})
        )


@dataclass
class IPCResponse:
    """IPC 응답 구조"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.data is not None:
            # bytes 타입은 base64로 인코딩
            if isinstance(self.data, bytes):
                result["data"] = base64.b64encode(self.data).decode('utf-8')
                result["data_type"] = "bytes"
            else:
                # dataclass, list 등을 직렬화
                result["data"] = serialize_data(self.data)
                result["data_type"] = "json"
        if self.error:
            result["error"] = self.error
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "IPCResponse":
        response_data = data.get("data")
        # bytes 타입 복원
        if data.get("data_type") == "bytes" and response_data:
            response_data = base64.b64decode(response_data)
        return cls(
            success=data["success"],
            data=response_data,
            error=data.get("error")
        )


class SimulationBridge:
    """
    IsaacLab 시뮬레이션 측 ZMQ 브릿지
    
    시뮬레이션 루프에서 non-blocking으로 메시지를 처리합니다.
    
    사용법:
        bridge = SimulationBridge(port=5555)
        bridge.register_adapter("camera", camera_adapter)
        bridge.register_adapter("object", object_adapter)
        
        while sim_running:
            sim.step()
            bridge.process_messages()  # non-blocking
    """
    
    def __init__(self, port: int = 5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 0)  # non-blocking
        
        self._adapters: Dict[str, Any] = {}
        self._running = True
        
        print(f"[SimulationBridge] ZMQ REP 소켓 바인딩: tcp://*:{port}")
    
    def register_adapter(self, name: str, adapter: Any) -> None:
        """어댑터 등록"""
        self._adapters[name] = adapter
        print(f"[SimulationBridge] 어댑터 등록: {name}")
    
    def process_messages(self) -> None:
        """
        메시지 처리 (non-blocking)
        
        시뮬레이션 루프에서 매 프레임 호출합니다.
        """
        try:
            # non-blocking 수신
            raw_msg = self.socket.recv_string(zmq.NOBLOCK)
            msg = IPCMessage.from_dict(json.loads(raw_msg))
            
            # 메시지 처리
            response = self._handle_message(msg)
            
            # 응답 전송
            self.socket.send_string(json.dumps(response.to_dict()))
            
        except zmq.Again:
            # 메시지 없음 (정상)
            pass
        except Exception as e:
            print(f"[SimulationBridge] 메시지 처리 오류: {e}")
            try:
                error_response = IPCResponse(success=False, error=str(e))
                self.socket.send_string(json.dumps(error_response.to_dict()))
            except:
                pass
    
    def _handle_message(self, msg: IPCMessage) -> IPCResponse:
        """메시지 처리 및 어댑터 메서드 호출"""
        adapter = self._adapters.get(msg.adapter)
        if adapter is None:
            return IPCResponse(success=False, error=f"Unknown adapter: {msg.adapter}")
        
        method = getattr(adapter, msg.method, None)
        if method is None:
            return IPCResponse(success=False, error=f"Unknown method: {msg.method}")
        
        try:
            result = method(*msg.args, **msg.kwargs)
            return IPCResponse(success=True, data=result)
        except Exception as e:
            return IPCResponse(success=False, error=str(e))
    
    def close(self) -> None:
        """소켓 종료"""
        self._running = False
        self.socket.close()
        self.context.term()
        print("[SimulationBridge] 종료됨")


class ServerBridge:
    """
    FastAPI 서버 측 ZMQ 브릿지
    
    시뮬레이션에 요청을 보내고 응답을 받습니다.
    
    사용법:
        bridge = ServerBridge(zmq_url="tcp://localhost:5555")
        
        # 카메라 프레임 요청
        result = bridge.call("camera", "get_frame")
        
        # 물체 생성
        result = bridge.call("object", "spawn", 
                            args=["box", [0,0,0.5], [1,0,0,0]])
    """
    
    def __init__(self, zmq_url: str = "tcp://localhost:5555", timeout_ms: int = 5000):
        self.zmq_url = zmq_url
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(zmq_url)
        
        self._lock = threading.Lock()
        
        print(f"[ServerBridge] ZMQ REQ 소켓 연결: {zmq_url}")
    
    def call(self, adapter: str, method: str, args: list = None, kwargs: dict = None) -> IPCResponse:
        """
        시뮬레이션에 요청 전송
        
        Args:
            adapter: 어댑터 이름 ("camera", "object", "material")
            method: 메서드 이름
            args: 위치 인자
            kwargs: 키워드 인자
        
        Returns:
            IPCResponse: 응답
        """
        msg = IPCMessage(adapter=adapter, method=method, args=args, kwargs=kwargs)
        
        with self._lock:
            try:
                # 요청 전송
                self.socket.send_string(json.dumps(msg.to_dict()))
                
                # 응답 수신
                raw_response = self.socket.recv_string()
                return IPCResponse.from_dict(json.loads(raw_response))
                
            except zmq.Again:
                return IPCResponse(success=False, error="Timeout: 시뮬레이션 응답 없음")
            except Exception as e:
                return IPCResponse(success=False, error=str(e))
    
    def is_connected(self) -> bool:
        """연결 상태 확인 (ping)"""
        try:
            response = self.call("scene", "is_running")
            return response.success
        except:
            return False
    
    def close(self) -> None:
        """소켓 종료"""
        self.socket.close()
        self.context.term()
        print("[ServerBridge] 종료됨")
