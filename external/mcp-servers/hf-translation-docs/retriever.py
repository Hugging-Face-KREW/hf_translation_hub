"""File retrieval and analysis for HuggingFace documentation."""

import os
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any
import requests

from project_config import get_project_config


def get_github_repo_files(project: str = "transformers") -> List[str]:
    """Get github repo files."""
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


def get_github_issue_open_pr(project: str = "transformers", lang: str = "ko", all_files: List[str] = None) -> Tuple[List[str], List[str]]:
    """Get open PR in the github issue, filtered by title containing '[i18n-KO]'."""
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
        
        # Find files that end with this base name
        matching_files = [f for f in all_files if f.endswith(f"{base_name}.md")]
        
        # Prefer files in English docs
        en_files = [f for f in matching_files if "/en/" in f]
        if en_files:
            return en_files[0]
        elif matching_files:
            return matching_files[0]
        
        return None

    docs_in_progress = []
    pr_info_list = []

    for pr in filtered_prs:
        title = pr["title"]
        pr_url = pr["html_url"]
        
        # Extract the filename from the title
        match = pattern.search(title)
        if match:
            # Get the filename (from either backticks or without)
            filename_from_title = match.group(1) if match.group(1) else match.group(2)
            
            # Find the actual file path in the repository
            original_file_path = find_original_file_path(filename_from_title, all_files)
            
            if original_file_path:
                docs_in_progress.append(original_file_path)
                pr_info_list.append(pr_url)

    return docs_in_progress, pr_info_list


# Simplified translation analysis classes
class LanguageInfo:
    def __init__(self, code: str, name: str):
        self.code = code
        self.value = code
        self.name = name

# Simple language lookup
def get_language_info(lang_code: str) -> LanguageInfo:
    """Get language info by code."""
    languages = {
        "ko": LanguageInfo("ko", "Korean"),
        "zh": LanguageInfo("zh", "Chinese"),
        "ja": LanguageInfo("ja", "Japanese"),
        "es": LanguageInfo("es", "Spanish"),
        "fr": LanguageInfo("fr", "French")
    }
    return languages.get(lang_code, languages["ko"])


class TranslationDoc:
    def __init__(self, translation_lang: str, original_file: str, translation_file: str, translation_exists: bool):
        self.translation_lang = translation_lang
        self.original_file = original_file
        self.translation_file = translation_file
        self.translation_exists = translation_exists


class Summary:
    def __init__(self, lang: str):
        self.lang = lang
        self.files: List[TranslationDoc] = []
        
    def append_file(self, doc: TranslationDoc):
        self.files.append(doc)
    
    @property
    def files_analyzed(self) -> int:
        return len(self.files)
    
    @property
    def files_missing_translation(self) -> int:
        return len([f for f in self.files if not f.translation_exists])
    
    @property 
    def percentage_missing_translation(self) -> float:
        if self.files_analyzed == 0:
            return 0.0
        return (self.files_missing_translation / self.files_analyzed) * 100
    
    def first_missing_translation_files(self, limit: int) -> List[TranslationDoc]:
        missing = [f for f in self.files if not f.translation_exists]
        return missing[:limit]


def retrieve(summary: Summary, table_size: int = 10) -> Tuple[str, List[str]]:
    """Retrieve missing docs"""
    
    report = f"""
| Item | Count | Percentage |
|------|-------|------------|
| 📂 HuggingFaces docs | {summary.files_analyzed} | - |
| 🪹 Missing translations | {summary.files_missing_translation} | {summary.percentage_missing_translation:.2f}% |
"""
    print(report)
    first_missing_docs = []
    for file in summary.first_missing_translation_files(table_size):
        first_missing_docs.append(file.original_file)

    print(first_missing_docs)
    return report, first_missing_docs


def report(project: str, target_lang: str, top_k: int = 1, docs_file: List[str] = None) -> Tuple[str, List[str]]:
    """Generate a report for the translated docs"""
    if docs_file is None:
        raise ValueError("Repository file list must be provided")

    base_docs_path = Path("docs/source")
    en_docs_path = Path("docs/source/en")

    lang = get_language_info(target_lang)
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