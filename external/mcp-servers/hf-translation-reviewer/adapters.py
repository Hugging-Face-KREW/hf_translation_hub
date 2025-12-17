from __future__ import annotations

import base64
import os
from typing import Dict, Optional

import requests
from setting import SETTINGS


# ---------------- Token resolution (Space Secrets fallback) -----------------

def _resolve_token(explicit: str, env_key: str) -> str:
    """
    1) MCP tool arguments로 넘어온 token이 있으면 사용
    2) 없으면 Space Secrets(환경변수)에서 읽어서 사용
    """
    t = (explicit or "").strip()
    if t:
        return t

    t = (os.getenv(env_key) or "").strip()
    if t:
        return t

    raise RuntimeError(
        f"Missing token. Provide '{env_key}' as a HuggingFace Space Secret "
        "or pass it explicitly to the tool."
    )


def resolve_github_token(explicit: str) -> str:
    return _resolve_token(explicit, "GITHUB_TOKEN")


# ---------------- GitHub HTTP adapters -----------------

def github_request(
    url: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
) -> Dict:
    token = resolve_github_token(token)

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }

    response = requests.get(url, headers=headers, params=params, timeout=30)

    if response.status_code == 404:
        raise FileNotFoundError(f"GitHub resource not found: {url}")
    if response.status_code == 401:
        raise PermissionError("GitHub token is invalid or lacks necessary scopes.")
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed with status {response.status_code}: {response.text}"
        )

    return response.json()


def fetch_file_from_pr(
    repo_name: str,
    pr_number: int,
    path: str,
    head_sha: str,
    github_token: str,
) -> str:
    url = f"{SETTINGS.github_api_base}/repos/{repo_name}/contents/{path}"
    data = github_request(url, github_token, params={"ref": head_sha})

    content = data.get("content")
    encoding = data.get("encoding")

    if content is None or encoding != "base64":
        raise ValueError(
            f"Unexpected content response for '{path}' (encoding={encoding!r})."
        )

    decoded = base64.b64decode(content)
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File '{path}' in PR {pr_number} is not valid UTF-8 text"
        ) from exc
