from __future__ import annotations

from typing import Dict

from services import (
    prepare_translation_context,
    review_and_emit_payload,
    submit_review_to_github,
    run_end_to_end,
)


def tool_prepare(
    github_token: str = "",
    pr_url: str = "",
    original_path: str = "",
    translated_path: str = "",
) -> Dict[str, object]:
    """
    Tool 1: Fetch Files + Build Prompts
    """
    return prepare_translation_context(
        github_token=github_token,
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
    )


def tool_review_and_emit(
    provider: str,
    provider_token: str = "",
    model_name: str = "",
    pr_url: str = "",
    translated_path: str = "",
    original: str = "",
    translated: str = "",
) -> Dict[str, object]:
    """
    Tool 2: LLM Review + Emit Payload
    """
    return review_and_emit_payload(
        provider=provider,
        provider_token=provider_token,
        model_name=model_name,
        pr_url=pr_url,
        translated_path=translated_path,
        original=original,
        translated=translated,
    )


def tool_submit_review(
    github_token: str = "",
    pr_url: str = "",
    translated_path: str = "",
    payload_or_review: Dict[str, object] = None,  # type: ignore[assignment]
    allow_self_request_changes: bool = True,
) -> Dict[str, object]:
    """
    Tool 3: Submit Review
    """
    if payload_or_review is None:
        raise ValueError("payload_or_review is required")

    return submit_review_to_github(
        github_token=github_token,
        pr_url=pr_url,
        translated_path=translated_path,
        payload_or_review=payload_or_review,
        allow_self_request_changes=allow_self_request_changes,
    )


def tool_end_to_end(
    provider: str,
    provider_token: str = "",
    model_name: str = "",
    github_token: str = "",
    pr_url: str = "",
    original_path: str = "",
    translated_path: str = "",
    save_review: bool = False,
    save_path: str = "review.json",
    submit_review_flag: bool = False,
) -> Dict[str, object]:
    """
    Tool 4: End-to-End
    """
    return run_end_to_end(
        provider=provider,
        provider_token=provider_token,
        model_name=model_name,
        github_token=github_token,
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
        save_review=save_review,
        save_path=save_path,
        submit_review_flag=submit_review_flag,
    )
