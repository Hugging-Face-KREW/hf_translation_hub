"""Business logic for HuggingFace translation documentation."""

from pathlib import Path
from typing import Dict, Any

from project_config import get_project_config as get_base_config, get_available_projects
from retriever import report, get_github_repo_files, get_github_issue_open_pr
from adapters import get_content, preprocess_content, get_full_prompt, get_language_name
from datetime import datetime
import hashlib


def get_project_configuration(project: str) -> Dict[str, Any]:
    """
    Get project-specific configuration and settings.
    """
    if project not in get_available_projects():
        raise ValueError(f"Unknown project: {project}. Available: {get_available_projects()}")
    
    config = get_base_config(project)
    
    return {
        "project": project,
        "repo_url": config.repo_url,
        "docs_path": config.docs_path,
        "supported_languages": ["ko", "zh", "ja", "es", "fr", "de", "it", "pt"],
        "reference_pr_url": config.reference_pr_url,
        "translation_guidelines": {
            "preserve_code_blocks": True,
            "keep_english_terms": ["API", "token", "embedding", "transformer", "model"],
            "style_guide_url": f"{config.repo_url}/blob/main/docs/TRANSLATION_GUIDE.md"
        }
    }


def get_supported_projects() -> list[str]:
    """
    Get list of supported projects.
    """
    return get_available_projects()


def search_translation_files_data(project: str, target_language: str, max_files: int = 10) -> Dict[str, Any]:
    """
    Search for files that need translation in a HuggingFace project.
    """
    try:
        # Get all repository files
        all_repo_files = get_github_repo_files(project)
        
        # Get translation status report and file list
        status_report, missing_files = report(project, target_language, max_files * 2, all_repo_files)
        
        # Get files already in progress (if available)
        try:
            docs_in_progress, pr_info_list = get_github_issue_open_pr(project, target_language, all_repo_files)
            # Filter out files already in progress
            available_files = [f for f in missing_files if f not in docs_in_progress]
        except (ValueError, Exception):
            # If no issue tracking or error, use all missing files
            available_files = missing_files
            docs_in_progress = []
        
        # Limit to max_files
        files_to_return = available_files[:max_files]
        
        # Build file list with metadata
        files_data = []
        for file_path in files_to_return:
            # Estimate file size and priority (simplified)
            file_size = len(file_path) * 100  # Rough estimate
            priority = "high" if "model_doc" in file_path else "medium"
            
            files_data.append({
                "path": file_path,
                "size": file_size,
                "last_modified": datetime.now().isoformat() + "Z",
                "priority": priority,
                "translation_status": "missing"
            })
        
        return {
            "project": project,
            "target_language": target_language,
            "total_found": len(missing_files),
            "files": files_data,
            "statistics": {
                "missing": len(missing_files),
                "in_progress": len(docs_in_progress) if 'docs_in_progress' in locals() else 0,
                "up_to_date": 0
            },
            "status_report": status_report
        }
        
    except Exception as e:
        raise ValueError(f"Failed to search translation files: {str(e)}")


def get_file_content_data(project: str, file_path: str, include_metadata: bool = True) -> Dict[str, Any]:
    """
    Retrieve original file content for translation.
    """
    try:
        # Get raw file content
        content = get_content(file_path, project)
        
        # Generate metadata
        metadata = {}
        if include_metadata:
            content_bytes = content.encode('utf-8')
            metadata = {
                "encoding": "utf-8",
                "size": len(content_bytes),
                "last_modified": datetime.now().isoformat() + "Z",
                "content_hash": f"sha256:{hashlib.sha256(content_bytes).hexdigest()[:12]}..."
            }
        
        # Process content for translation
        processed_content = preprocess_content(content)
        
        # Count removed elements (simplified)
        original_code_blocks = content.count('```')
        processed_code_blocks = processed_content.count('```')
        code_blocks_removed = max(0, original_code_blocks - processed_code_blocks)
        
        original_tables = content.count('|')
        processed_tables = processed_content.count('|')
        tables_removed = max(0, (original_tables - processed_tables) // 4)  # Rough estimate
        
        return {
            "file_path": file_path,
            "content": content,
            "metadata": metadata,
            "processed_content": {
                "to_translate": processed_content,
                "code_blocks_removed": code_blocks_removed,
                "tables_removed": tables_removed
            }
        }
        
    except Exception as e:
        raise ValueError(f"Failed to get file content: {str(e)}")


def generate_translation_prompt_data(
    target_language: str, 
    content: str, 
    additional_instruction: str = "",
    project: str = "transformers",
    file_path: str = ""
) -> Dict[str, Any]:
    """
    Generate optimized translation prompt for the content.
    """
    try:
        # Convert language code to full language name
        target_language_name = get_language_name(target_language)
        
        # Generate the complete translation prompt
        prompt = get_full_prompt(target_language_name, content, additional_instruction)
        
        # Determine content type and domain based on file path
        content_type = "technical_documentation"
        domain = "machine_learning"
        file_type = "general_documentation"
        
        if file_path:
            if "model_doc" in file_path:
                file_type = "model_documentation"
            elif "tutorial" in file_path:
                file_type = "tutorial"
            elif "api" in file_path:
                file_type = "api_reference"
        
        # Translation guidelines based on project
        guidelines = [
            "Preserve markdown formatting",
            "Keep technical terms in English where appropriate",
            "Maintain code block integrity",
            "Use glossary terms when available",
            "Do not translate product names after 🤗 emoji"
        ]
        
        return {
            "prompt": prompt,
            "context": {
                "target_language_name": target_language_name,
                "content_type": content_type,
                "domain": domain,
                "file_type": file_type,
                "project": project
            },
            "guidelines": guidelines,
            "metadata": {
                "prompt_length": len(prompt),
                "content_length": len(content),
                "has_additional_instruction": bool(additional_instruction.strip()),
                "language_code": target_language
            }
        }
        
    except Exception as e:
        raise ValueError(f"Failed to generate translation prompt: {str(e)}")


def validate_translation_data(
    original_content: str,
    translated_content: str,
    target_language: str,
    file_path: str = ""
) -> Dict[str, Any]:
    """
    Validate translated content for quality and formatting.
    """
    try:
        issues = []
        suggestions = []
        quality_score = 1.0
        
        # Basic validation checks
        if not translated_content.strip():
            issues.append({
                "type": "content",
                "message": "Translated content is empty",
                "severity": "error"
            })
            quality_score = 0.0
        
        # Check if content length is reasonable
        original_length = len(original_content)
        translated_length = len(translated_content)
        length_ratio = translated_length / original_length if original_length > 0 else 0
        
        if length_ratio < 0.3:
            issues.append({
                "type": "length",
                "message": "Translated content seems too short",
                "severity": "warning"
            })
            quality_score -= 0.2
        elif length_ratio > 3.0:
            issues.append({
                "type": "length", 
                "message": "Translated content seems too long",
                "severity": "warning"
            })
            quality_score -= 0.1
        
        # Markdown formatting validation
        formatting_valid = True
        links_preserved = True
        code_blocks_intact = True
        
        # Check markdown headers
        original_headers = original_content.count('#')
        translated_headers = translated_content.count('#')
        if abs(original_headers - translated_headers) > 2:
            formatting_valid = False
            issues.append({
                "type": "formatting",
                "message": f"Header count mismatch: {original_headers} vs {translated_headers}",
                "severity": "warning"
            })
        
        # Check code blocks
        original_code_blocks = original_content.count('```')
        translated_code_blocks = translated_content.count('```')
        if original_code_blocks != translated_code_blocks:
            code_blocks_intact = False
            issues.append({
                "type": "formatting",
                "message": f"Code block count mismatch: {original_code_blocks} vs {translated_code_blocks}",
                "severity": "error"
            })
            quality_score -= 0.3
        
        # Check links
        original_links = original_content.count('](')
        translated_links = translated_content.count('](')
        if abs(original_links - translated_links) > 1:
            links_preserved = False
            issues.append({
                "type": "formatting",
                "message": f"Link count mismatch: {original_links} vs {translated_links}",
                "severity": "warning"
            })
            quality_score -= 0.1
        
        # Language-specific suggestions
        if target_language == "ko":
            suggestions.append({
                "type": "terminology",
                "message": "Consider using '모델' consistently for 'model'",
                "line": 0
            })
            
            # Check for common Korean translation issues
            if "transformer" in translated_content.lower() and "트랜스포머" not in translated_content:
                suggestions.append({
                    "type": "terminology",
                    "message": "Consider using '트랜스포머' for 'transformer'",
                    "line": 0
                })
        
        # Final quality score adjustment
        quality_score = max(0.0, min(1.0, quality_score))
        
        is_valid = quality_score >= 0.7 and not any(issue["severity"] == "error" for issue in issues)
        
        return {
            "is_valid": is_valid,
            "quality_score": quality_score,
            "issues": issues,
            "suggestions": suggestions,
            "formatting": {
                "markdown_valid": formatting_valid,
                "links_preserved": links_preserved,
                "code_blocks_intact": code_blocks_intact
            },
            "statistics": {
                "original_length": original_length,
                "translated_length": translated_length,
                "length_ratio": length_ratio,
                "header_count": translated_headers,
                "code_block_count": translated_code_blocks // 2 if translated_code_blocks % 2 == 0 else 0
            }
        }
        
    except Exception as e:
        raise ValueError(f"Failed to validate translation: {str(e)}")


def save_translation_result_data(
    project: str,
    original_file_path: str,
    translated_content: str,
    target_language: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Save translation result to file system.
    """
    try:
        from pathlib import Path
        import os
        import time
        
        if metadata is None:
            metadata = {}
        
        # Create target directory structure
        base_path = Path("translation_result")
        
        # Convert English path to target language path
        original_path = Path(original_file_path)
        if "docs/source/en/" in original_file_path:
            # Replace /en/ with target language
            target_path = Path(original_file_path.replace("/en/", f"/{target_language}/"))
        else:
            # Add language prefix to filename
            target_path = original_path.parent / f"{target_language}_{original_path.name}"
        
        # Full save path
        save_path = base_path / target_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists
        backup_path = None
        if save_path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = save_path.parent / f"{save_path.stem}_backup_{timestamp}.md"
            backup_path.write_text(save_path.read_text(encoding='utf-8'), encoding='utf-8')
        
        # Write translated content
        save_path.write_text(translated_content, encoding='utf-8')
        
        # Calculate file info
        file_size = len(translated_content.encode('utf-8'))
        checksum = hashlib.sha256(translated_content.encode('utf-8')).hexdigest()
        
        # Create metadata file
        metadata_info = {
            "project": project,
            "original_file": original_file_path,
            "target_language": target_language,
            "translation_date": datetime.now().isoformat(),
            "file_size": file_size,
            "checksum": checksum,
            **metadata
        }
        
        metadata_path = save_path.with_suffix('.meta.json')
        import json
        metadata_path.write_text(json.dumps(metadata_info, indent=2), encoding='utf-8')
        
        return {
            "saved_path": str(save_path),
            "backup_path": str(backup_path) if backup_path else None,
            "file_size": file_size,
            "checksum": f"sha256:{checksum[:12]}...",
            "created_directories": [str(save_path.parent)],
            "metadata_path": str(metadata_path)
        }
        
    except Exception as e:
        raise ValueError(f"Failed to save translation result: {str(e)}")