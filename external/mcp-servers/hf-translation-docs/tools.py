"""MCP tool endpoints for HuggingFace translation documentation."""

from __future__ import annotations
from typing import Any, Dict

from services import (
    get_project_configuration,
    search_translation_files_data,
    get_file_content_data,
    generate_translation_prompt_data,
    validate_translation_data,
    save_translation_result_data,
)


def get_project_config(project: str) -> Dict[str, Any]:
    """
    Get project-specific configuration and settings.
    
    MCP endpoint: translation_get_project_config
    """
    try:
        return {
            "status": "success",
            "data": get_project_configuration(project)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "PROJECT_CONFIG_ERROR",
                "message": str(e),
                "details": {"project": project}
            }
        }


def search_translation_files(
    project: str,
    target_language: str,
    max_files: int = 10
) -> Dict[str, Any]:
    """
    Search for files that need translation.
    
    MCP endpoint: translation_search_files
    """
    try:
        data = search_translation_files_data(project, target_language, max_files)
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "SEARCH_FILES_ERROR",
                "message": str(e),
                "details": {
                    "project": project,
                    "target_language": target_language,
                    "max_files": max_files
                }
            }
        }


def get_file_content(
    project: str,
    file_path: str,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Get file content for translation.
    
    MCP endpoint: translation_get_file_content
    """
    try:
        data = get_file_content_data(project, file_path, include_metadata)
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "FILE_CONTENT_ERROR",
                "message": str(e),
                "details": {
                    "project": project,
                    "file_path": file_path
                }
            }
        }


def generate_translation_prompt(
    target_language: str,
    content: str,
    additional_instruction: str = "",
    project: str = "transformers",
    file_path: str = ""
) -> Dict[str, Any]:
    """
    Generate translation prompt for content.
    
    MCP endpoint: translation_generate_prompt
    """
    try:
        data = generate_translation_prompt_data(
            target_language, content, additional_instruction, project, file_path
        )
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "PROMPT_GENERATION_ERROR",
                "message": str(e),
                "details": {
                    "target_language": target_language,
                    "content_length": len(content),
                    "project": project,
                    "file_path": file_path
                }
            }
        }


def validate_translation(
    original_content: str,
    translated_content: str,
    target_language: str,
    file_path: str = ""
) -> Dict[str, Any]:
    """
    Validate translated content for quality and formatting.
    
    MCP endpoint: translation_validate
    """
    try:
        data = validate_translation_data(
            original_content, translated_content, target_language, file_path
        )
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": str(e),
                "details": {
                    "original_content_length": len(original_content),
                    "translated_content_length": len(translated_content),
                    "target_language": target_language,
                    "file_path": file_path
                }
            }
        }


def save_translation_result(
    project: str,
    original_file_path: str,
    translated_content: str,
    target_language: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Save translation result to file system.
    
    MCP endpoint: translation_save_result
    """
    try:
        data = save_translation_result_data(
            project, original_file_path, translated_content, target_language, metadata
        )
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": {
                "code": "SAVE_RESULT_ERROR",
                "message": str(e),
                "details": {
                    "project": project,
                    "original_file_path": original_file_path,
                    "target_language": target_language,
                    "content_length": len(translated_content)
                }
            }
        }