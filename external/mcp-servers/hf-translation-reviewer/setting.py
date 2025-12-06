from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import os

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class AppSettings:
    default_provider: str = "openai"
    default_model: str = "gpt-5"
    github_api_base: str = "https://api.github.com"
    ui_title: str = "LLM Translation Reviewer (PR) — MCP Tools"
    ui_share: bool = True
    ui_launch_mcp_server: bool = True


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    if yaml is None:
        # yaml 없으면 config 없이 동작
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_settings(config_path: str = "configs/default.yaml") -> AppSettings:
    cfg_path = Path(config_path)
    data = _load_yaml(cfg_path)

    provider_cfg = data.get("provider", {}) if isinstance(data.get("provider"), dict) else {}
    github_cfg = data.get("github", {}) if isinstance(data.get("github"), dict) else {}
    ui_cfg = data.get("ui", {}) if isinstance(data.get("ui"), dict) else {}

    default_provider = os.getenv("DEFAULT_PROVIDER", provider_cfg.get("default", "openai"))
    default_model = os.getenv("DEFAULT_MODEL", provider_cfg.get("model", "gpt-5"))
    github_api_base = os.getenv("GITHUB_API_BASE", github_cfg.get("api_base", "https://api.github.com"))
    ui_title = ui_cfg.get("title", "LLM Translation Reviewer (PR) — MCP Tools")
    ui_share = bool(ui_cfg.get("share", True))
    ui_launch_mcp_server = bool(ui_cfg.get("launch_mcp_server", True))

    return AppSettings(
        default_provider=default_provider,
        default_model=default_model,
        github_api_base=github_api_base,
        ui_title=ui_title,
        ui_share=ui_share,
        ui_launch_mcp_server=ui_launch_mcp_server,
    )


# 전역 설정 인스턴스
SETTINGS: AppSettings = load_settings()
