"""Business logic for HuggingFace Translation PR Generator."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import hashlib

from project_config import get_project_config as get_base_config, get_available_projects
from adapters import check_github_token_validity, get_repository_info, search_github_prs


def get_supported_projects() -> List[str]:
    """Get list of supported projects."""
    return get_available_projects()


def validate_pr_config_data(
    owner: str,
    repo_name: str,
    project: str
) -> Dict[str, Any]:
    """Validate GitHub PR configuration and settings."""
    try:
        validation_results = {
            "is_valid": True,
            "github_config": {
                "token_valid": False,
                "repo_access": False,
                "fork_exists": False,
                "permissions": []
            },
            "project_config": {},
            "missing_requirements": [],
            "recommendations": []
        }

        # Check if project exists
        if project not in get_available_projects():
            validation_results["missing_requirements"].append(f"Unknown project: {project}")
            validation_results["is_valid"] = False

        # Get project configuration
        try:
            config = get_base_config(project)
            validation_results["project_config"] = {
                "base_repo_url": config.repo_url,
                "docs_path": config.docs_path,
                "supported_languages": ["ko", "zh", "ja", "es", "fr", "de", "it", "pt"]
            }
        except Exception as e:
            validation_results["missing_requirements"].append(f"Project config error: {str(e)}")
            validation_results["is_valid"] = False

        # Get GitHub token from environment
        github_token = os.environ.get("GITHUB_TOKEN", "").strip()
        
        # Validate GitHub token
        if not github_token:
            validation_results["missing_requirements"].append("GITHUB_TOKEN environment variable is required")
            validation_results["is_valid"] = False
        else:
            # Check token validity using adapter
            token_check = check_github_token_validity(github_token)
            if token_check["valid"]:
                validation_results["github_config"]["token_valid"] = True
            else:
                validation_results["missing_requirements"].append(f"Invalid GITHUB_TOKEN: {token_check.get('error', 'Unknown error')}")
                validation_results["is_valid"] = False

            # Check repository access
            if owner and repo_name:
                repo_info = get_repository_info(owner, repo_name, github_token)
                
                if repo_info.get("exists"):
                    validation_results["github_config"]["repo_access"] = True
                    validation_results["github_config"]["fork_exists"] = True
                    
                    if repo_info.get("fork"):
                        validation_results["recommendations"].append("Repository is a fork - perfect for PRs!")
                    else:
                        validation_results["recommendations"].append("Repository is not a fork - consider using a fork")
                        
                    # Check permissions
                    permissions = repo_info.get("permissions", {})
                    if permissions.get("admin"):
                        validation_results["github_config"]["permissions"].append("admin")
                    if permissions.get("push"):
                        validation_results["github_config"]["permissions"].append("write")
                    if permissions.get("pull"):
                        validation_results["github_config"]["permissions"].append("read")
                else:
                    validation_results["missing_requirements"].append(f"Cannot access repository {owner}/{repo_name}: {repo_info.get('error', 'Unknown error')}")
                    validation_results["is_valid"] = False
            else:
                if not owner:
                    validation_results["missing_requirements"].append("GitHub owner is required")
                if not repo_name:
                    validation_results["missing_requirements"].append("Repository name is required")
                validation_results["is_valid"] = False

        # Add recommendations
        if validation_results["is_valid"]:
            validation_results["recommendations"].append("All configurations are valid - ready for PR creation!")
        
        return validation_results
        
    except Exception as e:
        raise ValueError(f"Failed to validate PR configuration: {str(e)}")


def search_reference_pr_data(
    target_language: str,
    context: str = "documentation translation"
) -> Dict[str, Any]:
    """Search for reference PRs using GitHub API (no LLM completion)."""
    try:
        reference_prs = []
        search_metadata = {
            "total_found": 0,
            "search_criteria": {
                "language": target_language,
                "context": context,
                "search_terms": []
            },
            "search_time": datetime.now().isoformat()
        }

        # Build search query
        lang_map = {
            "ko": ["korean", "ko", "i18n-ko", "[ko]"],
            "zh": ["chinese", "zh", "i18n-zh", "[zh]"],
            "ja": ["japanese", "ja", "i18n-ja", "[ja]"],
            "es": ["spanish", "es", "i18n-es", "[es]"],
            "fr": ["french", "fr", "i18n-fr", "[fr]"]
        }
        
        search_terms = lang_map.get(target_language, [target_language])
        search_metadata["search_criteria"]["search_terms"] = search_terms
        
        # Search across multiple HF repositories
        repos_to_search = [
            "huggingface/transformers",
            "huggingface/diffusers", 
            "huggingface/smolagents"
        ]

        for repo in repos_to_search:
            for term in search_terms:
                try:
                    # Build search query
                    query = f"repo:{repo} is:pr is:merged {term} translation"
                    search_result = search_github_prs(query, per_page=5)
                    
                    if search_result["success"]:
                        data = search_result["data"]
                        search_metadata["total_found"] += data.get("total_count", 0)
                        
                        for item in data.get("items", []):
                            # Get PR details
                            pr_url = item.get("pull_request", {}).get("html_url")
                            if pr_url:
                                # Calculate relevance score
                                score = 0.0
                                title = item.get("title", "").lower()
                                body = item.get("body", "") or ""
                                
                                # Score based on title matches
                                for search_term in search_terms:
                                    if search_term.lower() in title:
                                        score += 1.0
                                
                                # Score based on context
                                if "translation" in title or "translate" in title:
                                    score += 1.0
                                if "doc" in title or "documentation" in title:
                                    score += 0.5
                                
                                reference_prs.append({
                                    "url": pr_url,
                                    "title": item.get("title", ""),
                                    "description": body[:500] + ("..." if len(body) > 500 else ""),
                                    "files_changed": [],  # Would need separate API call
                                    "language": target_language,
                                    "score": score,
                                    "created_at": item.get("created_at", "")
                                })
                    else:
                        print(f"Error searching {repo} with term {term}: {search_result.get('error')}")
                                
                except Exception as e:
                    print(f"Error searching {repo} with term {term}: {e}")
                    continue

        # Sort by score and remove duplicates
        seen_urls = set()
        unique_prs = []
        for pr in sorted(reference_prs, key=lambda x: x["score"], reverse=True):
            if pr["url"] not in seen_urls:
                seen_urls.add(pr["url"])
                unique_prs.append(pr)
                
        # Limit to top 10 results
        reference_prs = unique_prs[:10]
        
        return {
            "reference_prs": reference_prs,
            "search_metadata": search_metadata,
            "suggestion": "Use MCP client to analyze these PRs and select the best reference"
        }
        
    except Exception as e:
        raise ValueError(f"Failed to search reference PRs: {str(e)}")


def analyze_translation_data(
    filepath: str,
    translated_content: str,
    target_language: str,
    project: str = "transformers"
) -> Dict[str, Any]:
    """Analyze translated content and generate metadata."""
    try:
        # File path analysis
        original_path = filepath
        if "/en/" in filepath:
            target_path = filepath.replace("/en/", f"/{target_language}/")
        else:
            path_parts = filepath.split("/")
            path_parts[-1] = f"{target_language}_{path_parts[-1]}"
            target_path = "/".join(path_parts)

        # Determine file type
        file_type = "documentation"
        if "model_doc" in filepath:
            file_type = "model_documentation"
        elif "tutorial" in filepath:
            file_type = "tutorial" 
        elif "api" in filepath:
            file_type = "api_reference"

        # Content analysis
        translated_size = len(translated_content.encode('utf-8'))
        
        # Markdown structure analysis
        headers_count = translated_content.count('#')
        code_blocks_count = translated_content.count('```')
        links_count = translated_content.count('](')
        tables_count = translated_content.count('|')

        # Generate branch name
        file_name = filepath.split('/')[-1].replace('.md', '').replace('_', '-')
        suggested_branch_name = f"{target_language}-{file_name}"

        # Determine priority
        priority = "medium"
        if "model_doc" in filepath:
            priority = "high"
        elif "tutorial" in filepath:
            priority = "high" 
        elif "index" in filepath or "readme" in filepath.lower():
            priority = "low"

        return {
            "file_analysis": {
                "original_path": original_path,
                "target_path": target_path,
                "file_type": file_type,
                "size_comparison": {
                    "original_size": 0,  # Would need original content
                    "translated_size": translated_size,
                    "size_ratio": 1.0  # Placeholder
                }
            },
            "content_analysis": {
                "markdown_structure": {
                    "headers_count": headers_count,
                    "code_blocks_count": code_blocks_count // 2,  # Pair count
                    "links_count": links_count,
                    "tables_count": max(0, tables_count // 4)  # Rough estimate
                },
                "translation_quality": {
                    "completeness": 1.0,  # Placeholder - would need comparison
                    "formatting_preserved": code_blocks_count % 2 == 0,  # Even number
                    "code_integrity": '```' not in translated_content.replace('```', '', code_blocks_count)
                }
            },
            "pr_metadata": {
                "suggested_branch_name": suggested_branch_name,
                "file_category": file_type,
                "priority": priority
            },
            "context_for_llm": {
                "file_description": f"{file_type.replace('_', ' ').title()} for {project}",
                "translation_guidelines": [
                    "Preserve markdown formatting",
                    "Keep technical terms in English where appropriate", 
                    "Maintain code block integrity",
                    "Use glossary terms when available"
                ],
                "formatting_notes": [
                    f"Contains {headers_count} headers",
                    f"Contains {code_blocks_count // 2} code blocks",
                    f"Contains {links_count} links"
                ]
            }
        }
        
    except Exception as e:
        raise ValueError(f"Failed to analyze translation: {str(e)}")


def generate_pr_draft_data(
    filepath: str,
    translated_content: str,
    target_language: str,
    reference_pr_url: str,
    project: str = "transformers"
) -> Dict[str, Any]:
    """Generate PR draft structure and metadata."""
    try:
        # Generate branch name
        file_name = filepath.split('/')[-1].replace('.md', '').replace('_', '-')
        branch_name = f"{target_language}-{file_name}"
        
        # Determine target file path
        if "/en/" in filepath:
            target_file_path = filepath.replace("/en/", f"/{target_language}/")
        else:
            path_parts = filepath.split("/")
            path_parts[-1] = f"{target_language}_{path_parts[-1]}"
            target_file_path = "/".join(path_parts)

        # File changes
        file_changes = [
            {
                "action": "create",
                "path": target_file_path,
                "content": translated_content
            }
        ]

        # Check if toctree update is needed
        toctree_required = "docs/source" in filepath and target_language in ["ko", "zh", "ja"]
        toctree_files = []
        if toctree_required:
            # Common toctree files that might need updates
            lang_path = f"docs/source/{target_language}"
            toctree_files = [
                f"{lang_path}/_toctree.yml",
                f"{lang_path}/index.md"
            ]

        # Generate prompts for LLM
        file_title = filepath.split('/')[-1].replace('.md', '').replace('_', ' ').title()
        
        title_prompt = f"""Generate a concise GitHub PR title for translating the file '{filepath}' to {target_language}.

The file is: {file_title}
Target language: {target_language}

Reference PR URL for style: {reference_pr_url}

Requirements:
- Follow the pattern from the reference PR
- Include language code (e.g., [ko], [zh])
- Be specific about what was translated
- Keep it under 72 characters

Generate only the title, no explanations."""

        description_prompt = f"""Generate a GitHub PR description for translating '{filepath}' to {target_language}.

Context:
- File: {filepath} 
- Target language: {target_language}
- Project: {project}
- Reference PR: {reference_pr_url}

Requirements:
- Follow the format and style of the reference PR
- Mention what was translated
- Include any relevant technical details
- Be concise but informative
- Use proper GitHub markdown formatting

Generate only the description, no explanations."""

        return {
            "pr_structure": {
                "branch_name": branch_name,
                "target_file_path": target_file_path,
                "base_branch": "main"
            },
            "file_changes": file_changes,
            "reference_analysis": {
                "reference_pr_url": reference_pr_url,
                "title_pattern": "Please analyze the reference PR",
                "description_template": "Please analyze the reference PR",
                "common_elements": ["Please analyze using LLM"]
            },
            "llm_prompts": {
                "title_generation_prompt": title_prompt,
                "description_generation_prompt": description_prompt,
                "context": {
                    "filepath": filepath,
                    "target_language": target_language,
                    "project": project,
                    "reference_pr_url": reference_pr_url
                }
            },
            "toctree_updates": {
                "required": toctree_required,
                "target_files": toctree_files,
                "updates": []  # Would be populated by toctree handler
            }
        }
        
    except Exception as e:
        raise ValueError(f"Failed to generate PR draft: {str(e)}")


def create_github_pr_data(
    github_token: str,
    owner: str,
    repo_name: str,
    filepath: str,
    translated_content: str,
    target_language: str,
    reference_pr_url: str,
    project: str,
    pr_title: str,
    pr_description: str,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Create GitHub PR with translated content using GitHubPRAgent."""
    try:
        if metadata is None:
            metadata = {}

        # Import GitHubAgent (no LLM dependencies - copied from original)
        try:
            from github_agent import GitHubAgent
            GITHUB_PR_AVAILABLE = True
        except ImportError as e:
            print(f"⚠️ GitHubAgent not available: {e}")
            GITHUB_PR_AVAILABLE = False

        if not GITHUB_PR_AVAILABLE:
            # Fallback to simulation
            return _simulate_pr_creation(owner, repo_name, filepath, translated_content, 
                                       target_language, pr_title, pr_description)

        # Get project configuration
        config = get_base_config(project)
        base_repo_path = config.repo_url.replace("https://github.com/", "")
        base_owner, base_repo = base_repo_path.split("/")

        # Set GitHub token in environment
        os.environ["GITHUB_TOKEN"] = github_token

        # Initialize GitHubAgent (no LLM - copied from original)
        agent = GitHubAgent(
            user_owner=owner,
            user_repo=repo_name,
            base_owner=base_owner,
            base_repo=base_repo,
        )

        # Execute PR creation (title/description from MCP client)
        print(f"🚀 Creating GitHub PR...")
        print(f"   📁 File: {filepath}")
        print(f"   🌍 Language: {target_language}")
        print(f"   📊 Reference PR: {reference_pr_url}")
        print(f"   🏠 User Fork: {owner}/{repo_name}")
        print(f"   🎯 Base Repository: {base_owner}/{base_repo}")

        result = agent.run_translation_pr_workflow(
            reference_pr_url=reference_pr_url,
            target_language=target_language,
            filepath=filepath,
            translated_doc=translated_content,
            pr_title=pr_title or f"[{target_language}] Translate {filepath.split('/')[-1]}",
            pr_description=pr_description or f"Add {target_language} translation for {filepath}",
            base_branch=metadata.get("base_branch", "main"),
        )

        # Convert GitHubPRAgent result to our format
        return _convert_agent_result_to_mcp_format(result, owner, repo_name)
        
    except Exception as e:
        raise ValueError(f"Failed to create GitHub PR: {str(e)}")


def _simulate_pr_creation(owner: str, repo_name: str, filepath: str, translated_content: str,
                         target_language: str, pr_title: str, pr_description: str) -> Dict[str, Any]:
    """Fallback simulation when GitHubPRAgent is not available."""
    file_name = filepath.split('/')[-1].replace('.md', '').replace('_', '-')
    branch_name = f"{target_language}-{file_name}"
    
    if "/en/" in filepath:
        target_file_path = filepath.replace("/en/", f"/{target_language}/")
    else:
        path_parts = filepath.split("/")
        path_parts[-1] = f"{target_language}_{path_parts[-1]}"
        target_file_path = "/".join(path_parts)

    pr_number = 1234
    pr_url = f"https://github.com/{owner}/{repo_name}/pull/{pr_number}"
    commit_sha = hashlib.sha256(f"{translated_content}{datetime.now()}".encode()).hexdigest()[:8]

    return {
        "pr_url": pr_url,
        "branch_name": branch_name,
        "files_created": [
            {
                "path": target_file_path,
                "status": "created",
                "commit_sha": commit_sha
            }
        ],
        "pr_details": {
            "number": pr_number,
            "title": pr_title,
            "description": pr_description,
            "state": "open",
            "created_at": datetime.now().isoformat()
        },
        "toctree_status": {
            "updated": False,
            "files_modified": [],
            "commit_sha": None
        },
        "additional_info": {
            "existing_pr_updated": False,
            "conflicts": [],
            "warnings": [
                "GitHubPRAgent not available - using simulation mode"
            ]
        }
    }


def _convert_agent_result_to_mcp_format(agent_result: Dict[str, Any], owner: str, repo_name: str) -> Dict[str, Any]:
    """Convert GitHubPRAgent result format to MCP format."""
    status = agent_result.get("status", "unknown")
    
    if status == "success":
        return {
            "pr_url": agent_result.get("pr_url", f"https://github.com/{owner}/{repo_name}/pull/unknown"),
            "branch_name": agent_result.get("branch", "unknown"),
            "files_created": [
                {
                    "path": agent_result.get("file_path", "unknown"),
                    "status": "created",
                    "commit_sha": "unknown"
                }
            ],
            "pr_details": {
                "number": 0,  # Would need to parse from pr_url
                "title": "Translation PR",
                "description": "Generated by MCP server",
                "state": "open",
                "created_at": datetime.now().isoformat()
            },
            "toctree_status": {
                "updated": False,
                "files_modified": [],
                "commit_sha": None
            },
            "additional_info": {
                "existing_pr_updated": False,
                "conflicts": [],
                "warnings": [],
                "original_message": agent_result.get("message", "")
            }
        }
    elif status == "partial_success":
        return {
            "pr_url": agent_result.get("pr_url", f"https://github.com/{owner}/{repo_name}/pull/unknown"),
            "branch_name": agent_result.get("branch", "unknown"),
            "files_created": [
                {
                    "path": agent_result.get("file_path", "unknown"),
                    "status": "created",
                    "commit_sha": "unknown"
                }
            ],
            "pr_details": {
                "number": 0,
                "title": "Translation PR",
                "description": "Generated by MCP server", 
                "state": "open",
                "created_at": datetime.now().isoformat()
            },
            "toctree_status": {
                "updated": False,
                "files_modified": [],
                "commit_sha": None
            },
            "additional_info": {
                "existing_pr_updated": "Existing PR found" in agent_result.get("error_details", ""),
                "conflicts": [],
                "warnings": [agent_result.get("error_details", "")],
                "original_message": agent_result.get("message", "")
            }
        }
    else:
        # Error case
        raise ValueError(f"GitHub PR creation failed: {agent_result.get('message', 'Unknown error')} - {agent_result.get('error_details', '')}")