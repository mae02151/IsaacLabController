"""
설정 관리 모듈
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class StreamingConfig:
    """스트리밍 설정"""
    fps: int = 30
    quality: int = 85  # JPEG 품질


@dataclass
class AssetConfig:
    """에셋 설정"""
    name: str
    usd_path: Optional[str] = None
    asset_type: str = "primitive"  # "primitive" 또는 "usd"


@dataclass
class ControllerConfig:
    """전체 컨트롤러 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    objects: List[AssetConfig] = field(default_factory=list)
    materials: List[Dict[str, Any]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControllerConfig":
        """딕셔너리에서 설정 생성"""
        server_cfg = data.get("server", {})
        streaming_cfg = data.get("streaming", {})
        assets_cfg = data.get("assets", {})
        
        return cls(
            host=server_cfg.get("host", "0.0.0.0"),
            port=server_cfg.get("port", 8000),
            streaming=StreamingConfig(
                fps=streaming_cfg.get("fps", 30),
                quality=streaming_cfg.get("quality", 85),
            ),
            objects=[
                AssetConfig(
                    name=obj.get("name", ""),
                    usd_path=obj.get("usd_path"),
                    asset_type=obj.get("type", "primitive"),
                )
                for obj in assets_cfg.get("objects", [])
            ],
            materials=assets_cfg.get("materials", []),
        )


def load_config(config_path: str) -> ControllerConfig:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
    
    Returns:
        ControllerConfig 인스턴스
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return ControllerConfig.from_dict(data or {})
