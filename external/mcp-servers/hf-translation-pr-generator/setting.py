"""Configuration settings for HuggingFace Translation PR Generator MCP server."""

import os
import yaml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path


@dataclass
class Settings:
    """Global settings for the application."""
    ui_title: str = "HuggingFace Translation PR Generator"
    default_project: str = "transformers"
    default_language: str = "ko"
    default_limit: int = 10
    port: int = 7862
    
    # GitHub settings
    github_api_url: str = "https://api.github.com"
    github_timeout: int = 30
    search_per_page: int = 30
    
    # PR generation settings
    default_base_branch: str = "main"
    max_file_size: int = 1048576
    
    # Language search terms
    language_terms: Dict[str, List[str]] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return {}
    else:
        print(f"Warning: Config file not found at {config_path}")
        return {}


def create_settings() -> Settings:
    """Create settings from configuration file."""
    config = load_config()
    
    # Extract values with defaults
    ui_config = config.get("ui", {})
    defaults_config = config.get("defaults", {})
    github_config = config.get("github", {})
    pr_config = config.get("pr_generation", {})
    search_config = config.get("search", {})
    
    return Settings(
        ui_title=ui_config.get("title", "HuggingFace Translation PR Generator"),
        port=ui_config.get("port", 7862),
        default_project=defaults_config.get("project", "transformers"),
        default_language=defaults_config.get("language", "ko"),
        default_limit=defaults_config.get("limit", 10),
        github_api_url=github_config.get("api_url", "https://api.github.com"),
        github_timeout=github_config.get("timeout", 30),
        search_per_page=github_config.get("search_per_page", 30),
        default_base_branch=pr_config.get("default_base_branch", "main"),
        max_file_size=pr_config.get("max_file_size", 1048576),
        language_terms=search_config.get("language_terms", {})
    )


# Language choices: (display_name, language_code)
LANGUAGE_CHOICES: List[Tuple[str, str]] = [
    ("Korean", "ko"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
]

# Create global settings instance
SETTINGS = create_settings()