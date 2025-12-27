"""MCP tool endpoints for HuggingFace Translation PR Generator."""

from __future__ import annotations
from typing import Any, Dict

from services import (
    validate_pr_config_data,
    search_reference_pr_data,
    analyze_translation_data,
    generate_pr_draft_data,
    create_github_pr_data,
)


def validate_pr_config(
    owner: str,
    repo_name: str,
    project: str
) -> Dict[str, Any]:
    """
    Validate GitHub PR configuration and settings.
    
    MCP endpoint: pr_validate_config
    """
    try:
        return {
            "status": "success",
            "data": validate_pr_config_data(owner, repo_name, project)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "PR_CONFIG_VALIDATION_ERROR",
                "message": str(e),
                "details": {
                    "owner": owner,
                    "repo_name": repo_name,
                    "project": project
                }
            }
        }


def search_reference_pr(
    target_language: str,
    context: str = "documentation translation"
) -> Dict[str, Any]:
    """
    Search for reference PRs using GitHub API.
    
    MCP endpoint: pr_search_reference
    """
    try:
        return {
            "status": "success",
            "data": search_reference_pr_data(target_language, context)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "REFERENCE_PR_SEARCH_ERROR",
                "message": str(e),
                "details": {
                    "target_language": target_language,
                    "context": context
                }
            }
        }


def analyze_translation(
    filepath: str,
    translated_content: str,
    target_language: str,
    project: str = "transformers"
) -> Dict[str, Any]:
    """
    Analyze translated content and generate metadata.
    
    MCP endpoint: pr_analyze_translation
    """
    try:
        return {
            "status": "success",
            "data": analyze_translation_data(filepath, translated_content, target_language, project)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "TRANSLATION_ANALYSIS_ERROR",
                "message": str(e),
                "details": {
                    "filepath": filepath,
                    "content_length": len(translated_content),
                    "target_language": target_language,
                    "project": project
                }
            }
        }


def generate_pr_draft(
    filepath: str,
    translated_content: str,
    target_language: str,
    reference_pr_url: str,
    project: str = "transformers"
) -> Dict[str, Any]:
    """
    Generate PR draft structure and metadata.
    
    MCP endpoint: pr_generate_draft
    """
    try:
        return {
            "status": "success",
            "data": generate_pr_draft_data(
                filepath, translated_content, target_language, reference_pr_url, project
            )
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "PR_DRAFT_GENERATION_ERROR",
                "message": str(e),
                "details": {
                    "filepath": filepath,
                    "content_length": len(translated_content),
                    "target_language": target_language,
                    "reference_pr_url": reference_pr_url,
                    "project": project
                }
            }
        }


def create_github_pr(
    github_token: str,
    owner: str,
    repo_name: str,
    filepath: str,
    translated_content: str,
    target_language: str,
    reference_pr_url: str,
    project: str,
    pr_title: str = "",
    pr_description: str = "",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create GitHub PR with translated content.
    
    MCP endpoint: pr_create_github_pr
    """
    try:
        return {
            "status": "success",
            "data": create_github_pr_data(
                github_token, owner, repo_name, filepath, translated_content,
                target_language, reference_pr_url, project, pr_title, pr_description, metadata
            )
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "GITHUB_PR_CREATION_ERROR",
                "message": str(e),
                "details": {
                    "github_token": bool(github_token),
                    "owner": owner,
                    "repo_name": repo_name,
                    "filepath": filepath,
                    "content_length": len(translated_content),
                    "target_language": target_language,
                    "project": project,
                    "has_pr_title": bool(pr_title),
                    "has_pr_description": bool(pr_description)
                }
            }
        }