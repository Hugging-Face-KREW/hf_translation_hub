import re
import os
from pathlib import Path

import requests

from .model import Languages, Summary, TranslationDoc
from .project_config import get_project_config


def get_github_repo_files(project: str = "transformers"):
    """
    Get github repo files
    """
    config = get_project_config(project)
    
    # Add GitHub token if available to avoid rate limiting (optional)
    headers = {}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    response = requests.get(config.api_url, headers=headers)
    
    # Handle rate limit with helpful message
    if response.status_code == 403 and "rate limit" in response.text.lower():
        raise Exception(f"GitHub API rate limit exceeded. To avoid this, set GITHUB_TOKEN in your environment or provide a GitHub token in the UI. Details: {response.text}")

    data = response.json()
    all_items = data.get("tree", [])

    file_paths = [
        item["path"]
        for item in all_items
        if item["type"] == "blob" and (item["path"].startswith("docs"))
    ]
    return file_paths


def get_github_issue_open_pr(project: str = "transformers", lang: str = "ko", all_files: list = None):
    """
    Get open PR in the github issue, filtered by title containing '[i18n-KO]'.
    """
    config = get_project_config(project)
    issue_id = config.github_issues.get(lang)
    
    # For projects without GitHub issue tracking, still search for PRs
    if not issue_id:
        raise ValueError(f"⚠️ No GitHub issue registered for {project}.")

    # Require all_files parameter 
    if all_files is None:
        raise ValueError("Repository file list must be provided")
    
    headers = {
        "Accept": "application/vnd.github+json",
    }
    
    # Add GitHub token if available to avoid rate limiting (optional)
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    all_open_prs = []
    page = 1
    per_page = 100  # Maximum allowed by GitHub API
    
    while True:
        repo_path = config.repo_url.replace("https://github.com/", "")
        url = f"https://api.github.com/repos/{repo_path}/pulls?state=open&page={page}&per_page={per_page}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise Exception(f"GitHub API rate limit exceeded. To avoid this, set GITHUB_TOKEN in your environment or provide a GitHub token in the UI. Details: {response.text}")
        elif response.status_code != 200:
            raise Exception(f"GitHub API error: {response.status_code} {response.text}")
        
        page_prs = response.json()
        if not page_prs:  # No more PRs
            break
            
        all_open_prs.extend(page_prs)
        page += 1
        
        # Break if we got less than per_page results (last page)
        if len(page_prs) < per_page:
            break

    filtered_prs = [pr for pr in all_open_prs if "[i18n-KO]" in pr["title"]]

    # Pattern to match filenames after "Translated" keyword
    pattern = re.compile(r"Translated\s+(?:`([^`]+)`|(\S+))\s+to")

    def find_original_file_path(filename_from_title, all_files):
        """Find the exact file path from repo files by matching filename"""
        if not filename_from_title:
            return None
            
        # Remove .md extension for matching
        base_name = filename_from_title.replace('.md', '')
        
        # Look for exact matches in repo files
        for file_path in all_files:
            if file_path.startswith("docs/source/en/") and file_path.endswith(".md"):
                file_base = file_path.split("/")[-1].replace('.md', '')
                if file_base == base_name:
                    return file_path
                    
        # If no exact match, fallback to simple path
        return f"docs/source/en/{filename_from_title}"
    
    filenames = []
    pr_info_list = []
    
    for pr in filtered_prs:
        match = pattern.search(pr["title"])
        if match:
            # Use group 1 (with backticks) or group 2 (without backticks)
            filename = match.group(1) or match.group(2)
            # Add .md extension if not present
            if not filename.endswith('.md'):
                filename += '.md'
                
            # Find the correct file path by matching filename
            correct_path = None
            if filename:
                # Remove .md extension for matching
                base_name = filename.replace('.md', '')
                
                # Look for exact matches in repo files
                for file_path in all_files:
                    if file_path.startswith("docs/source/en/") and file_path.endswith(".md"):
                        file_base = file_path.split("/")[-1].replace('.md', '')
                        if file_base == base_name:
                            correct_path = file_path
                            break
                            
                # If no exact match, fallback to simple path
                if not correct_path:
                    correct_path = f"docs/source/en/{filename}"
            if correct_path:
                filenames.append(correct_path)
                pr_info_list.append(f"{config.repo_url}/pull/{pr['url'].rstrip('/').split('/')[-1]}")
    return filenames, pr_info_list


def retrieve(summary: Summary, table_size: int = 10) -> tuple[str, list[str]]:
    """
    Retrieve missing docs
    """

    report = f"""
| Item | Count | Percentage |
|------|-------|------------|
| 📂 HuggingFaces docs | {summary.files_analyzed} | - |
| 🪹 Missing translations | {summary.files_missing_translation} | {summary.percentage_missing_translation:.2f}% |
"""
    print(report)
    first_missing_docs = list()
    for file in summary.first_missing_translation_files(table_size):
        first_missing_docs.append(file.original_file)

    print(first_missing_docs)
    return report, first_missing_docs


def report(project: str, target_lang: str, top_k: int = 1, docs_file: list = None) -> tuple[str, list[str]]:
    """
    Generate a report for the translated docs
    """
    if docs_file is None:
        raise ValueError("Repository file list must be provided")

    base_docs_path = Path("docs/source")
    en_docs_path = Path("docs/source/en")

    lang = Languages[target_lang]
    summary = Summary(lang=lang.value)

    for file in docs_file:
        if file.endswith(".md"):
            try:
                file_relative_path = Path(file).relative_to(en_docs_path)
            except ValueError:
                continue

            translated_path = os.path.join(
                base_docs_path, lang.value, file_relative_path
            )
            translation_exists = translated_path in docs_file

            doc = TranslationDoc(
                translation_lang=lang.value,
                original_file=file,
                translation_file=translated_path,
                translation_exists=translation_exists,
            )
            summary.append_file(doc)
    return retrieve(summary, top_k)
