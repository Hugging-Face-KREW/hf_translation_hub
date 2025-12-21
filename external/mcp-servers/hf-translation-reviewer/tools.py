from __future__ import annotations

from typing import Dict

from adapters import resolve_github_token
from services import (
    prepare_translation_context,
    review_and_emit_payload,
    submit_review_to_github,
    run_end_to_end,
)


# ---------------------------------------------------------------------
# Token resolution (infra concern, NOT MCP input)
# ---------------------------------------------------------------------

def github_token_or_env(token: str = "") -> str:
    """
    Resolve GitHub token.

    Priority:
    1. Explicit token (usually from UI, if ever passed)
    2. Environment variables (GITHUB_TOKEN, HF_GITHUB_TOKEN, etc.)
    """
    return resolve_github_token(token)


# ---------------------------------------------------------------------
# Tool 1: Prepare
# ---------------------------------------------------------------------

def tool_prepare(
    pr_url: str = "",
    original_path: str = "",
    translated_path: str = "",
) -> Dict[str, object]:
    """
    Tool 1: Fetch files from GitHub PR and build translation review prompts.

    MCP-safe:
    - Does NOT accept github_token as an argument
    - Token is resolved internally from environment / secrets
    """
    return prepare_translation_context(
        github_token=github_token_or_env(""),
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
    )


# ---------------------------------------------------------------------
# Tool 2: Review + Emit Payload
# ---------------------------------------------------------------------

def tool_review_and_emit(
    pr_url: str = "",
    translated_path: str = "",
    translated: str = "",
    raw_review_response: str = "",
) -> Dict[str, object]:
    """
    Tool 2: Parse LLM review response and emit GitHub review payload.

    No GitHub access required here.
    """
    return review_and_emit_payload(
        pr_url=pr_url,
        translated_path=translated_path,
        translated=translated,
        raw_review_response=raw_review_response,
    )


# ---------------------------------------------------------------------
# Tool 3: Submit Review
# ---------------------------------------------------------------------

def tool_submit_review(
    pr_url: str = "",
    translated_path: str = "",
    payload_or_review: Dict[str, object] | None = None,
    allow_self_request_changes: bool = True,
) -> Dict[str, object]:
    """
    Tool 3: Submit review payload to GitHub PR.

    MCP-safe:
    - Token resolved internally
    """
    if payload_or_review is None:
        raise ValueError("payload_or_review is required")

    return submit_review_to_github(
        github_token=github_token_or_env(""),
        pr_url=pr_url,
        translated_path=translated_path,
        payload_or_review=payload_or_review,
        allow_self_request_changes=allow_self_request_changes,
    )


# ---------------------------------------------------------------------
# Tool 4: End-to-End
# ---------------------------------------------------------------------

def tool_end_to_end(
    pr_url: str = "",
    original_path: str = "",
    translated_path: str = "",
    save_review: bool = False,
    save_path: str = "review.json",
    submit_review_flag: bool = False,
    raw_review_response: str = "",
) -> Dict[str, object]:
    """
    Tool 4: End-to-end execution:
    - fetch files
    - build prompts
    - parse review
    - optionally save and/or submit to GitHub

    MCP-safe:
    - Token resolved internally
    """
    return run_end_to_end(
        github_token=github_token_or_env(""),
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
        save_review=save_review,
        save_path=save_path,
        submit_review_flag=submit_review_flag,
        raw_review_response=raw_review_response,
    )
