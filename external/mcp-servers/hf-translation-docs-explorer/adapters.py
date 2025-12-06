from __future__ import annotations

from typing import Dict, List

import requests

from setting import SETTINGS


def _build_auth_headers() -> Dict[str, str]:
    """
    GitHub 호출용 Authorization 헤더 생성.
    - 우선순위: SETTINGS.github_token → (fallback) 환경변수 GITHUB_TOKEN
    """
    token = SETTINGS.github_token
    if not token:
        # 환경변수 직접 조회
        import os
        token = os.environ.get("GITHUB_TOKEN", "")

    if not token:
        return {}
    return {"Authorization": f"token {token}"}


def fetch_document_paths(api_url: str) -> List[str]:
    """
    GitHub git/trees API에서 blob 경로 목록만 추출.

    Parameters
    ----------
    api_url : str
        예: https://api.github.com/repos/huggingface/transformers/git/trees/main?recursive=1
    """
    response = requests.get(
        api_url,
        headers=_build_auth_headers(),
        timeout=SETTINGS.request_timeout_seconds,
    )

    if response.status_code == 403 and "rate limit" in response.text.lower():
        raise RuntimeError(
            "GitHub API rate limit exceeded. Provide a GITHUB_TOKEN to continue."
        )

    response.raise_for_status()
    tree = response.json().get("tree", [])
    return [item["path"] for item in tree if item.get("type") == "blob"]
