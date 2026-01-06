from __future__ import annotations

from typing import Dict, Any, Optional

from setting import SETTINGS
from services import (
    prepare_translation_context,
    review_and_emit_payload,
    submit_review_to_github,
    run_end_to_end,
)
from adapters import resolve_github_token


# ---------------------------------------------------------
# Tool 1: Prepare
# ---------------------------------------------------------

def tool_prepare(
    pr_url: str,
    original_path: str,
    translated_path: str,
) -> Dict[str, Any]:
    """
    Fetch files from GitHub PR and build prompts for translation review.
    MCP-safe: GitHub token is resolved internally.
    """
    github_token = resolve_github_token(None)

    return prepare_translation_context(
        github_token=github_token,
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
    )


# ---------------------------------------------------------
# Tool 2: Review + Emit
# ---------------------------------------------------------

def tool_review_and_emit(
    pr_url: str,
    translated_path: str,
    translated: str,
    raw_review_response: str,
) -> Dict[str, Any]:
    """
    Parse LLM review response and emit GitHub review payload.
    """
    raw_review_response = (raw_review_response or "").strip()
    if not raw_review_response:
        raise ValueError("raw_review_response is required for review_and_emit")

    return review_and_emit_payload(
        pr_url=pr_url,
        translated_path=translated_path,
        translated=translated,
        raw_review_response=raw_review_response,
    )


# ---------------------------------------------------------
# Tool 3: Submit Review
# ---------------------------------------------------------

def tool_submit_review(
    pr_url: str,
    translated_path: str,
    payload_or_review: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Submit review payload to GitHub PR.
    """
    github_token = resolve_github_token(None)

    return submit_review_to_github(
        github_token=github_token,
        pr_url=pr_url,
        translated_path=translated_path,
        payload_or_review=payload_or_review,
    )


# ---------------------------------------------------------
# Tool 4: End-to-End
# ---------------------------------------------------------

def tool_end_to_end(
    pr_url: str,
    original_path: str,
    translated_path: str,
    save_review: bool = False,
    save_path: Optional[str] = None,
    submit_review_flag: bool = True,
    raw_review_response: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end translation review:
    - fetch files
    - (optional) generate or parse review
    - (optional) save review
    - (optional) submit review to GitHub
    """
    github_token = resolve_github_token(None)

    return run_end_to_end(
        github_token=github_token,
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
        save_review=save_review,
        save_path=save_path or "review.json",
        submit_review_flag=submit_review_flag,
        raw_review_response=raw_review_response or "",
    )
