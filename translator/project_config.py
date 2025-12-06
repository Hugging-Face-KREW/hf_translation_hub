"""Project configuration for different HuggingFace repositories."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ProjectConfig:
    """Configuration for a specific HuggingFace project."""
    name: str
    repo_url: str
    api_url: str
    docs_path: str
    github_issues: Dict[str, str]  # language -> issue_id
    reference_pr_url: str


# Project configurations
PROJECTS = {
    "transformers": ProjectConfig(
        name="Transformers",
        repo_url="https://github.com/huggingface/transformers",
        api_url="https://api.github.com/repos/huggingface/transformers/git/trees/main?recursive=1",
        docs_path="docs/source",
        github_issues={"ko": "20179"},
        reference_pr_url="https://github.com/huggingface/transformers/pull/24968"
    ),
    "smolagents": ProjectConfig(
        name="SmolAgents",
        repo_url="https://github.com/huggingface/smolagents",
        api_url="https://api.github.com/repos/huggingface/smolagents/git/trees/main?recursive=1",
        docs_path="docs/source", 
        github_issues={"ko": "20179"},  # To be filled when issue is created
        reference_pr_url="https://github.com/huggingface/smolagents/pull/1581"  # To be filled with actual PR URL
    )
}


def get_project_config(project_key: str) -> ProjectConfig:
    """Get project configuration by key."""
    if project_key not in PROJECTS:
        raise ValueError(f"Unknown project: {project_key}. Available: {list(PROJECTS.keys())}")
    return PROJECTS[project_key]


def get_available_projects() -> list[str]:
    """Get list of available project keys."""
    return list(PROJECTS.keys())