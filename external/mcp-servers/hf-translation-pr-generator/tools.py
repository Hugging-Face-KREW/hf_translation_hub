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
    project: str,
    toolCallId: str = ""
) -> Dict[str, Any]:
    """
    Validate GitHub PR configuration and settings.
    
    MCP endpoint: pr_validate_config

    Args:
        owner: The owner of the repository.
        repo_name: The name of the repository.
        project: The name of the project.
        toolCallId: The ID of the tool call.

    Returns:
        A dictionary containing the validation results.

    Raises:
        Exception: If the PR configuration cannot be validated.
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
    context: str = "documentation translation",
    toolCallId: str = ""
) -> Dict[str, Any]:
    """
    Search for reference PRs using GitHub API.
    
    MCP endpoint: pr_search_reference

    Args:
        target_language: The target language to search for reference PRs in.
        context: The context to search for reference PRs in.
        toolCallId: The ID of the tool call.

    Returns:
        A dictionary containing the search results.

    Raises:
        Exception: If the reference PRs cannot be searched for.
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
    project: str = "transformers",
    toolCallId: str = ""
) -> Dict[str, Any]:
    """
    Analyze translated content and generate metadata.
    
    MCP endpoint: pr_analyze_translation

    Args:
        filepath: The path of the file to analyze.
        translated_content: The translated content to analyze.
        target_language: The target language to analyze the content in.
        project: The name of the project to analyze the content in.
        toolCallId: The ID of the tool call.

    Returns:
        A dictionary containing the analysis results.

    Raises:
        Exception: If the content cannot be analyzed.
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
    project: str = "transformers",
    toolCallId: str = ""
) -> Dict[str, Any]:
    """
    Generate PR draft structure and metadata.
    
    MCP endpoint: pr_generate_draft

    Args:
        filepath: The path of the file to generate the PR draft for.
        translated_content: The translated content to generate the PR draft for.
        target_language: The target language to generate the PR draft for.
        reference_pr_url: The URL of the reference PR to generate the PR draft for.
        project: The name of the project to generate the PR draft for.
        toolCallId: The ID of the tool call.

    Returns:
        A dictionary containing the PR draft results.

    Raises:
        Exception: If the PR draft cannot be generated.
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
    metadata: Dict[str, Any] = None,
    toolCallId: str = ""
) -> Dict[str, Any]:
    """
    Create GitHub PR with translated content.
    
    MCP endpoint: pr_create_github_pr

    Args:
        github_token: The GitHub token to use for the PR.
        owner: The owner of the repository.
        repo_name: The name of the repository.
        filepath: The path of the file to create the PR for.
        translated_content: The translated content to create the PR for.
        target_language: The target language to create the PR for.
        reference_pr_url: The URL of the reference PR to create the PR for.
        project: The name of the project to create the PR for.
        pr_title: The title of the PR.
        pr_description: The description of the PR.
        metadata: The metadata to use for the PR.
        toolCallId: The ID of the tool call.

    Returns:
        A dictionary containing the PR creation results.

    Raises:
        Exception: If the PR cannot be created.
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