from __future__ import annotations

from typing import Any, Dict

from services import (
    build_project_catalog,
    build_search_response,
    build_missing_list_response,
)


def list_projects() -> Dict[str, Any]:
    """
    Gradio + MCP에서 사용되는 'translation_project_catalog' 엔드포인트.
    입력값 없이 전체 프로젝트 카탈로그를 반환한다.
    """
    return build_project_catalog(default=None)


def search_files(
    project: str,
    lang: str,
    limit: float | int,
    include_status_report: bool,
) -> Dict[str, Any]:
    """
    Gradio + MCP에서 사용되는 'translation_file_search' 엔드포인트.
    """
    return build_search_response(
        project=project,
        lang=lang,
        limit=int(limit or 1),
        include_status_report=bool(include_status_report),
    )


def list_missing_files(
    project: str,
    lang: str,
    limit: float | int,
) -> Dict[str, Any]:
    """
    Gradio + MCP에서 사용되는 'translation_missing_list' 엔드포인트.
    누락 파일 리스트만 반환.
    """
    return build_missing_list_response(
        project=project,
        lang=lang,
        limit=int(limit or 1),
    )
