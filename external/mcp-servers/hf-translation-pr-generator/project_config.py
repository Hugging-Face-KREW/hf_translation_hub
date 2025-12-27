"""Project configuration for HuggingFace Translation PR Generator."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ProjectConfig:
    """Configuration for a specific project."""
    project_name: str
    repo_url: str
    docs_path: str
    api_url: str
    reference_pr_url: str
    github_issues: Dict[str, str]


# Project configurations
PROJECTS = {
    "transformers": ProjectConfig(
        project_name="transformers",
        repo_url="https://github.com/huggingface/transformers",
        docs_path="docs/source",
        api_url="https://api.github.com/repos/huggingface/transformers/git/trees/main?recursive=1",
        reference_pr_url="https://github.com/huggingface/transformers/pull/30557",
        github_issues={
            "ko": "30675",
            "zh": "30676", 
            "ja": "30677",
            "es": "30678",
            "fr": "30679"
        }
    ),
    "diffusers": ProjectConfig(
        project_name="diffusers",
        repo_url="https://github.com/huggingface/diffusers",
        docs_path="docs/source",
        api_url="https://api.github.com/repos/huggingface/diffusers/git/trees/main?recursive=1",
        reference_pr_url="https://github.com/huggingface/diffusers/pull/8000",
        github_issues={
            "ko": "8001",
            "zh": "8002",
            "ja": "8003",
            "es": "8004",
            "fr": "8005"
        }
    ),
    "smolagents": ProjectConfig(
        project_name="smolagents",
        repo_url="https://github.com/huggingface/smolagents",
        docs_path="docs/source",
        api_url="https://api.github.com/repos/huggingface/smolagents/git/trees/main?recursive=1",
        reference_pr_url="https://github.com/huggingface/smolagents/pull/100",
        github_issues={
            "ko": "101",
            "zh": "102",
            "ja": "103",
            "es": "104",
            "fr": "105"
        }
    )
}


def get_project_config(project: str) -> ProjectConfig:
    """Get configuration for a specific project."""
    if project not in PROJECTS:
        raise ValueError(f"Unknown project: {project}. Available projects: {list(PROJECTS.keys())}")
    return PROJECTS[project]


def get_available_projects() -> List[str]:
    """Get list of all available projects."""
    return list(PROJECTS.keys())


def is_valid_project(project: str) -> bool:
    """Check if project is valid."""
    return project in PROJECTS