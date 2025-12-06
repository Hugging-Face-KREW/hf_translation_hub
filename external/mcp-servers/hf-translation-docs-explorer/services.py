from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from adapters import fetch_document_paths
from setting import SETTINGS


# Gradio / UI 에 노출할 언어 선택지
LANGUAGE_CHOICES: List[str] = [
    "ko",
    "ja",
    "zh",
    "fr",
    "de",
]


@dataclass(frozen=True)
class Project:
    """Store the minimum metadata required for documentation lookups."""

    slug: str
    name: str
    repo_url: str
    docs_path: str
    tree_api_url: str

    @property
    def repo_path(self) -> str:
        """Return the ``owner/repo`` identifier for GitHub API requests."""
        return self.repo_url.replace("https://github.com/", "")


# 지원 프로젝트 정의
PROJECTS: Dict[str, Project] = {
    "transformers": Project(
        slug="transformers",
        name="Transformers",
        repo_url="https://github.com/huggingface/transformers",
        docs_path="docs/source",
        tree_api_url=(
            "https://api.github.com/repos/huggingface/transformers/git/trees/main?recursive=1"
        ),
    ),
    "smolagents": Project(
        slug="smolagents",
        name="SmolAgents",
        repo_url="https://github.com/huggingface/smolagents",
        docs_path="docs/source",
        tree_api_url=(
            "https://api.github.com/repos/huggingface/smolagents/git/trees/main?recursive=1"
        ),
    ),
}


def get_available_projects() -> List[str]:
    """Return the list of project slugs supported by this module."""
    return sorted(PROJECTS.keys())


def _iter_english_docs(all_docs: Iterable[str], docs_root: str) -> Iterable[Path]:
    """Yield English documentation files as ``Path`` objects."""
    english_root = Path(docs_root) / "en"

    for doc_path in all_docs:
        if not doc_path.endswith(".md"):
            continue

        path = Path(doc_path)
        try:
            # en/ 아래에 있는지 필터링
            path.relative_to(english_root)
        except ValueError:
            continue

        yield path


def _compute_missing_translations(
    project_key: str,
    language: str,
    limit: int,
) -> Tuple[str, List[str], Project]:
    """
    영어 기준으로 누락 번역 파일을 계산하고,
    마크다운 요약 리포트 + 누락 경로 리스트 + Project 메타데이터를 반환.
    """
    project = PROJECTS[project_key]

    all_paths = fetch_document_paths(project.tree_api_url)
    english_docs = list(_iter_english_docs(all_paths, project.docs_path))
    english_total = len(english_docs)

    missing: List[str] = []
    docs_set = set(all_paths)

    for english_doc in english_docs:
        relative = english_doc.relative_to(Path(project.docs_path) / "en")
        translated_path = str(Path(project.docs_path) / language / relative)

        if translated_path not in docs_set:
            # 누락된 경우: 기준은 영어 경로(en/...)
            missing.append(str(english_doc))
            if len(missing) >= limit:
                break

    missing_count = len(missing)
    percentage = (missing_count / english_total * 100) if english_total else 0.0

    report = (
        "| Item | Count | Percentage |\n"
        "|------|-------|------------|\n"
        f"| English docs | {english_total} | - |\n"
        f"| Missing translations | {missing_count} | {percentage:.2f}% |"
    )

    return report, missing, project


def build_project_catalog(default: str | None) -> Dict[str, Any]:
    """Build the project catalog payload (API-neutral, pure logic)."""
    slugs = get_available_projects()
    default = default if default in slugs else None

    return {
        "type": "translation.project_list",
        "projects": [
            {
                "slug": slug,
                "display_name": PROJECTS[slug].name,
                "repo_url": PROJECTS[slug].repo_url,
                "docs_path": PROJECTS[slug].docs_path,
            }
            for slug in slugs
        ],
        "default_project": default,
        "total_projects": len(slugs),
    }


def build_search_response(
    project: str,
    lang: str,
    limit: int,
    include_status_report: bool,
) -> Dict[str, Any]:
    """
    누락 번역 파일 후보 + (선택) 상태 리포트를 포함한 검색 응답.
    MCP / Gradio 에서 사용 가능한 JSON 형태.
    """
    project = project.strip()
    lang = lang.strip()
    limit = max(1, int(limit))

    project_config = PROJECTS[project]

    status_report, candidate_paths, project_config = _compute_missing_translations(
        project_key=project,
        language=lang,
        limit=limit,
    )

    repo_url = project_config.repo_url.rstrip("/")

    return {
        "type": "translation.search.response",
        "request": {
            "type": "translation.search.request",
            "project": project,
            "target_language": lang,
            "limit": limit,
            "include_status_report": include_status_report,
        },
        "files": [
            {
                "rank": index,
                "path": path,
                "repo_url": f"{repo_url}/blob/main/{path}",
                "metadata": {
                    "project": project,
                    "target_language": lang,
                    "docs_path": project_config.docs_path,
                },
            }
            for index, path in enumerate(candidate_paths, start=1)
        ],
        "total_candidates": len(candidate_paths),
        "status_report": status_report if include_status_report else None,
    }


def build_missing_list_response(
    project: str,
    lang: str,
    limit: int,
) -> Dict[str, Any]:
    """
    누락 번역 파일 목록만 제공하는 응답(JSON).
    """
    project = project.strip()
    lang = lang.strip()
    limit_int = max(1, int(limit))

    status_report, missing_paths, project_config = _compute_missing_translations(
        project_key=project,
        language=lang,
        limit=limit_int,
    )

    repo_url = project_config.repo_url.rstrip("/")

    return {
        "type": "translation.missing_list",
        "project": project,
        "target_language": lang,
        "limit": limit_int,
        "count": len(missing_paths),
        "files": [
            {
                "rank": index,
                "path": path,
                "repo_url": f"{repo_url}/blob/main/{path}",
                "metadata": {
                    "project": project,
                    "target_language": lang,
                    "docs_path": project_config.docs_path,
                },
            }
            for index, path in enumerate(missing_paths, start=1)
        ],
        "status_report": status_report,  # 필요 없다면 제거 가능
    }
