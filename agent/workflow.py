"""Module for gradio interfaces."""

import os
from pathlib import Path
import gradio as gr

from translator.content import (
    fill_scaffold,
    get_content,
    get_full_prompt,
    llm_translate,
    preprocess_content,
)
from translator.retriever import report, get_github_issue_open_pr, get_github_repo_files
# GitHub PR Agent import
try:
    from pr_generator.agent import GitHubPRAgent

    GITHUB_PR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ GitHub PR Agent is not available: {e}")
    GITHUB_PR_AVAILABLE = False

import json
from logger.github_logger import GitHubLogger


def report_translation_target_files(
    project: str, translate_lang: str, top_k: int = 1
) -> tuple[str, list[list[str]]]:
    """Return the top-k files that need translation, excluding files already in progress.

    Args:
        project: Project to translate (e.g., "transformers", "smolagents")
        translate_lang: Target language to translate
        top_k: Number of top-first files to return for translation. (Default 1)
    """
    # Get repo files once to avoid duplicate API calls
    all_repo_files = get_github_repo_files(project)
    
    # Get all available files for translation using the file list
    all_status_report, all_filepath_list = report(project, translate_lang, top_k * 2, all_repo_files)  # Get more to account for filtering
    
    # Get files in progress using the same file list
    docs_in_progress, pr_info_list = get_github_issue_open_pr(project, translate_lang, all_repo_files)

    # Filter out files that are already in progress
    available_files = [f for f in all_filepath_list if f not in docs_in_progress]

    # Take only the requested number
    filepath_list = available_files[:top_k]
    
    # Build combined status report
    status_report = all_status_report

    if docs_in_progress:
        status_report += f"\n\n🤖 Found {len(docs_in_progress)} files in progress for translation:"
        for i, file in enumerate(docs_in_progress):
            status_report += f"\n{i+1}. [`{file}`]({pr_info_list[i]})"
        status_report += f"\n\n📋 Showing {len(filepath_list)} available files (excluding in-progress):"

    return status_report, [[file] for file in filepath_list]


def translate_docs(lang: str, file_path: str, additional_instruction: str = "", project: str = "transformers", force_retranslate: bool = False) -> tuple[str, str]:
    """Translate documentation."""
    # Check if translation already exists (unless force retranslate is enabled)
    translation_file_path = (
        Path(__file__).resolve().parent.parent
        / f"translation_result/{file_path}"
    )

    if not force_retranslate and translation_file_path.exists():
        print(f"📄 Found existing translation: {translation_file_path}")
        with open(translation_file_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
        if existing_content.strip():
            existing_msg = f"♻️ **Existing translation loaded** (no tokens used)\n📁 **File:** `{file_path}`\n📅 **Loaded from:** `{translation_file_path}`\n💡 **To retranslate:** Check 'Force Retranslate' option."
            return existing_msg, existing_content

    # step 1. Get content from file path
    content = get_content(file_path, project)
    to_translate = preprocess_content(content)

    # step 2. Prepare prompt with docs content
    if lang == "ko":
        translation_lang = "Korean"
    to_translate_with_prompt = get_full_prompt(translation_lang, to_translate, additional_instruction)

    print("to_translate_with_prompt:\n", to_translate_with_prompt)

    # step 3. Translate with LLM
    # TODO: MCP clilent 넘길 부분
    callback_result, translated_content = llm_translate(to_translate_with_prompt)
    print("translated_content:\n")
    print(translated_content)
    if translated_content.startswith("```md\n") and translated_content.endswith("```"):
        print("Satisfied translated_content.startswith ``` md")
        translated_content = translated_content[5:-3].strip()
    # step 4. Add scaffold to translation result
    translated_doc = fill_scaffold(content, to_translate, translated_content)
    print("translated_doc:\n")
    print(translated_doc)
    return callback_result, translated_doc


def translate_docs_interactive(
    translate_lang: str, selected_files: list[list[str]], additional_instruction: str = "", project: str = "transformers", force_retranslate: bool = False
) -> tuple[str, str]:
    """Interactive translation function that processes files one by one.

    Args:
        translate_lang: Target language to translate
        selected_files: List of file paths to translate
    """
    # Extract file paths from the dataframe format
    file_paths = [row[0] for row in selected_files if row and len(row) > 0]

    # Start with the first file
    current_file = file_paths[0]

    callback_result, translated_content = translate_docs(translate_lang, current_file, additional_instruction, project, force_retranslate)
    
    # Check if existing translation was loaded
    if isinstance(callback_result, str) and "Existing translation loaded" in callback_result:
        status = callback_result  # Use the existing translation message
    else:
        if force_retranslate:
            status = f"🔄 **Force Retranslation completed**: `{current_file}` → `{translate_lang}`\n\n"
        else:
            status = f"✅ Translation completed: `{current_file}` → `{translate_lang}`\n\n"
        status += f"💰 Used token and cost: \n```\n{callback_result}\n```"

    print(callback_result)
    print(status)

    return status, translated_content


def generate_github_pr(
    target_language: str,
    filepath: str,
    translated_content: str = None,
    github_config: dict = None,
    en_title: str = None,
    project: str = "transformers",
) -> str:
    """Generate a GitHub PR for translated documentation.

    Args:
        target_language: Target language for translation (e.g., "ko")
        filepath: Original file path (e.g., "docs/source/en/accelerator_selection.md")
        translated_content: Translated content (if None, read from file)
        github_config: GitHub configuration dictionary
        en_title: English title for toctree mapping

    Returns:
        PR creation result message
    """
    if not GITHUB_PR_AVAILABLE:
        return "❌ GitHub PR Agent is not available. Please install required libraries."

    if not github_config:
        return "❌ GitHub configuration not provided. Please set up GitHub token, owner, and repository in Configuration panel."

    # Validate required configuration
    required_fields = ["token", "owner", "repo_name", "reference_pr_url"]
    missing_fields = [
        field for field in required_fields if not github_config.get(field)
    ]

    if missing_fields:
        return f"❌ Missing required GitHub configuration: {', '.join(missing_fields)}\n\n💡 Go to Configuration panel and set:\n" + "\n".join([f"  • {field}" for field in missing_fields])

    # Set token in environment for the agent.
    os.environ["GITHUB_TOKEN"] = github_config["token"]

    try:
        # Read translated content from file if not provided
        if translated_content is None:
            translation_file_path = (
                Path(__file__).resolve().parent.parent
                / f"translation_result/{filepath}"
            )
            if not translation_file_path.exists():
                return f"❌ Translation file not found: {translation_file_path}\n\n💡 Please complete translation first in Tab 2 for file: {filepath}"

            with open(translation_file_path, "r", encoding="utf-8") as f:
                translated_content = f.read()

        if not translated_content or not translated_content.strip():
            return f"❌ Translated content is empty for file: {filepath}\n\n💡 Please complete translation first in Tab 2."

        # Execute GitHub PR Agent
        # Get base repository from project config
        from translator.project_config import get_project_config
        project_config = get_project_config(project)
        base_repo_path = project_config.repo_url.replace("https://github.com/", "")
        base_owner, base_repo = base_repo_path.split("/")

        print(f"🚀 Starting GitHub PR creation...")
        print(f"   📁 File: {filepath}")
        print(f"   🌍 Language: {target_language}")
        print(f"   📊 Reference PR: {github_config['reference_pr_url']}")
        print(f"   🏠 User Fork: {github_config['owner']}/{github_config['repo_name']}")
        print(f"   🎯 Base Repository: {base_owner}/{base_repo}")

        agent = GitHubPRAgent(
            user_owner=github_config["owner"],
            user_repo=github_config["repo_name"],
            base_owner=base_owner,
            base_repo=base_repo,
        )
        result = agent.run_translation_pr_workflow(
            reference_pr_url=github_config["reference_pr_url"],
            target_language=target_language,
            filepath=filepath,
            translated_doc=translated_content,
            base_branch=github_config.get("base_branch", "main"),
        )
        # TEST CODE
        # result = {
        #     'status': 'partial_success',
        #     'branch': 'ko-attention_interface',
        #     'file_path': 'docs/source/ko/attention_interface.md',
        #     'message': 'File was saved and commit was successful.\nPR creation failed: ERROR: Existing PR found: https://github.com/Jwaminju/transformers/pull/1', 'error_details': 'ERROR: Existing PR found: https://github.com/Jwaminju/transformers/pull/1'
        #     }
        # Process toctree update after successful translation PR
        toctree_result = None
        if en_title:
            from agent.toctree_handler import TocTreeHandler
            toctree_handler = TocTreeHandler(project)
            toctree_result = toctree_handler.update_toctree_after_translation(
                result, filepath, agent, github_config, project
            )

        # Process result
        # Generate toctree status message (shared for both success and partial_success)
        toctree_status = ""
        if toctree_result:
            if toctree_result["status"] == "success":
                toctree_status = f"\n📋 **Toctree Updated:** ✅ {toctree_result['message']}"
            else:
                toctree_status = f"\n📋 **Toctree Update Failed:** ❌ {toctree_result['message']}"

        # Append full result JSON to dedicated GitHub logging repository (always)
        try:
            log_data = result.copy()
            if toctree_result:
                log_data["toctree_result"] = toctree_result
            log_entry = json.dumps(log_data, ensure_ascii=False) + "\n"
            log_res = GitHubLogger().append_jsonl(log_entry)
            print(f"📝 Log append result: {log_res}")
        except Exception as e:
            print(f"❌ Failed to append PR log via GitHub API: {e}")

        if result["status"] == "success":
            return f"""✅ **GitHub PR Creation Successful!**

🔗 **PR URL:** {result.get('pr_url', 'NO_PR_URL')}
🌿 **Branch:** {result["branch"]}
📁 **File:** {result["file_path"]}{toctree_status}

{result["message"]}"""

        elif result["status"] == "partial_success":
            error_details = result.get("error_details", "Unknown error")
            
            # Check if it's "existing PR" case (not really an error)
            if "Existing PR found" in error_details:
                existing_pr_url = error_details.split(": ")[-1] if ": " in error_details else "Unknown"
                return f"""🔄 **Translation Updated Successfully**

🎯 **Selected Project:** {project}
🌿 **Branch:** {result["branch"]}
📁 **File:** {result["file_path"]}{toctree_status}

🔗 **Existing PR Updated:** {existing_pr_url}

✅ Your translation has been added to the existing PR. The file and toctree have been successfully updated!"""
            else:
                # Actual error case
                return f"""⚠️ **Partial Success**

🎯 **Selected Project:** {project}
🏠 **User Fork:** {github_config.get('owner', 'USER')}/{github_config.get('repo_name', 'REPO')}
🎯 **Target Base:** {base_owner}/{base_repo}
🌿 **Branch:** {result["branch"]}
📁 **File:** {result["file_path"]}{toctree_status}

{result["message"]}

**Error Details:**
{error_details}

💡 **Project-Repository Mismatch Check:**
- Selected project '{project}' should match repository '{github_config.get('repo_name', 'REPO')}'
- For smolagents: use Jwaminju/smolagents fork
- For transformers: use Jwaminju/transformers fork"""

        else:
            error_details = result.get("error_details", "No additional details")
            return f"""❌ **GitHub PR Creation Failed**

🎯 **Selected Project:** {project}
🏠 **User Fork:** {github_config.get('owner', 'USER')}/{github_config.get('repo_name', 'REPO')}
🎯 **Target Base:** {base_owner}/{base_repo}

**Error Message:**
{result["message"]}

**Error Details:**
{error_details}

💡 **Project-Repository Mismatch:**
Selected project '{project}' but configured repository '{github_config.get('repo_name', 'REPO')}'
• For smolagents project: use 'smolagents' repository
• For transformers project: use 'transformers' repository"""

    except Exception as e:
        error_msg = f"""❌ **Unexpected Error During PR Creation**

**Error:** {str(e)}

**Configuration:**
• Project: {project}
• File: {filepath}
• Target: {github_config.get('owner', 'USER')}/{github_config.get('repo_name', 'REPO')} → {base_owner if 'base_owner' in locals() else 'BASE'}/{base_repo if 'base_repo' in locals() else 'REPO'}"""
        print(error_msg)
        return error_msg


# Backward compatibility function (replaces old mock function)
def mock_generate_PR():
    """Backward compatibility function - returns warning message only"""
    return (
        "⚠️ mock_generate_PR() is deprecated. Please use generate_github_pr() instead."
    )
