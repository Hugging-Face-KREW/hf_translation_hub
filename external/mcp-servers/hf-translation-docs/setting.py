"""Configuration settings for HF Translation Docs MCP Server."""

from dataclasses import dataclass
from typing import List


@dataclass
class Settings:
    """Settings for the MCP server."""
    ui_title: str = "HuggingFace Translation Documentation MCP Server"
    default_project: str = "transformers"
    default_language: str = "ko"
    default_limit: int = 10
    supported_languages: List[str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["ko", "zh", "ja", "es", "fr", "de", "it", "pt"]


# Global settings instance
SETTINGS = Settings()

# Language choices for UI
LANGUAGE_CHOICES = [
    ("한국어 (Korean)", "ko"),
    ("中文 (Chinese)", "zh"),
    ("日本語 (Japanese)", "ja"),
    ("Español (Spanish)", "es"),
    ("Français (French)", "fr"),
    ("Deutsch (German)", "de"),
    ("Italiano (Italian)", "it"),
    ("Português (Portuguese)", "pt"),
]