from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import os

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class AppSettings:
    github_token: str = ""
    request_timeout_seconds: int = 30
    default_language: str = "ko"
    default_limit: int = 5
    ui_title: str = "Translation MCP Server"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def load_settings(config_path: str = "configs/default.yaml") -> AppSettings:
    cfg = _load_yaml(Path(config_path))

    github_cfg = cfg.get("github", {}) if isinstance(cfg.get("github"), dict) else {}
    trans_cfg = cfg.get("translation", {}) if isinstance(cfg.get("translation"), dict) else {}
    ui_cfg = cfg.get("ui", {}) if isinstance(cfg.get("ui"), dict) else {}

    # ENV > YAML
    github_token = os.getenv("GITHUB_TOKEN", github_cfg.get("token", ""))
    request_timeout_seconds = int(
        os.getenv("REQUEST_TIMEOUT_SECONDS", github_cfg.get("request_timeout_seconds", 30))
    )
    default_language = os.getenv("DEFAULT_LANGUAGE", trans_cfg.get("default_language", "ko"))
    default_limit = int(
        os.getenv("DEFAULT_LIMIT", trans_cfg.get("default_limit", 5))
    )
    ui_title = ui_cfg.get("title", "Translation MCP Server")

    return AppSettings(
        github_token=github_token,
        request_timeout_seconds=request_timeout_seconds,
        default_language=default_language,
        default_limit=default_limit,
        ui_title=ui_title,
    )


# 전역 설정 인스턴스
SETTINGS: AppSettings = load_settings()
