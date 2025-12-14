from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from setting import SETTINGS

from adapters import github_request, fetch_file_from_pr, dispatch_review, resolve_github_token

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a meticulous bilingual reviewer checking a translation PR.

    PR number: {pr_number}
    PR URL: {pr_url}

    Review the translated text against the original and focus on:
    1. Are there any typos or spelling mistakes?
    2. Are any sentences difficult to understand?
    3. Is the overall content hard to comprehend?

    Always respond with strict JSON using this schema:
    {{
      "verdict": "request_changes" | "comment" | "approve",
      "summary": "<High-level Markdown summary of the review findings>",
      "comments": [
        {{
          "line": <1-based line number in the translated file>,
          "issue": "<Short Markdown description of the problem>",
          "suggested_edit": "<Replacement text for the entire translated line>",
          "context": "<Exact current text of that line for grounding>"
        }},
        ...
      ]
    }}

    Guidelines:
    - Only include comments for issues that warrant direct feedback.
    - When a concrete rewrite is possible, populate "suggested_edit" with the full replacement line exactly as it should appear after fixing the issue.
    - Keep edits scoped to the referenced line; do not span multiple lines.
    - Always copy the current text of that line verbatim into "context".
    - Omit the "suggested_edit" field or set it to an empty string if no suggestion is available.
    - Use "request_changes" when the identified problems must be fixed before merging.
    - Use "approve" only when the translation is correct and clear with no changes needed.
    - For optional improvements or general observations, use "comment".
    - Keep suggestions tightly scoped so they can be applied as GitHub suggestions.
    - Do not output partial fragments in "suggested_edit"; always provide the entire replacement line including unchanged portions.
    - Use the line numbers from the "TRANSLATED TEXT WITH LINE NUMBERS" section.
    """
).strip()


# --------------------- Core helpers ------------------

def parse_pr_url(pr_url: str) -> Tuple[str, int]:
    """Extract repo (owner/name) and PR number from a GitHub PR URL."""
    if not pr_url:
        raise ValueError("PR URL is required")

    parsed = urlparse(pr_url)
    parts = [p for p in parsed.path.split("/") if p]
    # Expect: [owner, repo, 'pull', pr_number, ...]
    if len(parts) < 4 or parts[2] != "pull":
        raise ValueError(f"Not a valid GitHub PR URL: {pr_url}")

    owner, repo, _, num = parts[0], parts[1], parts[2], parts[3]
    if not num.isdigit():
        raise ValueError(f"PR number not found in URL: {pr_url}")

    return f"{owner}/{repo}", int(num)


def add_line_numbers(text: str) -> str:
    return "\n".join(f"{i:04d}: {line}" for i, line in enumerate(text.splitlines(), 1))


def load_pr_files(
    github_token: str,
    pr_url: str,
    original_path: str,
    translated_path: str,
) -> Tuple[str, int, str, str]:
    repo_name, pr_number = parse_pr_url(pr_url)

    pr_api = f"{SETTINGS.github_api_base}/repos/{repo_name}/pulls/{pr_number}"
    pr_data = github_request(pr_api, github_token)

    head_sha = pr_data.get("head", {}).get("sha")
    if not head_sha:
        raise RuntimeError(
            f"Unable to determine head SHA for PR {pr_number} in {repo_name}."
        )

    original = fetch_file_from_pr(repo_name, pr_number, original_path, head_sha, github_token)
    translated = fetch_file_from_pr(repo_name, pr_number, translated_path, head_sha, github_token)
    return repo_name, pr_number, original, translated


def build_messages(
    original: str,
    translated: str,
    pr_number: int,
    pr_url: str,
) -> Tuple[str, str]:
    system_prompt = (
        "You are an expert translation reviewer ensuring clarity, accuracy, "
        "and readability of localized documentation."
    )

    user_prompt = (
        f"{PROMPT_TEMPLATE}\n\n"
        "----- ORIGINAL TEXT -----\n"
        f"{original}\n\n"
        "----- TRANSLATED TEXT -----\n"
        f"{translated}\n\n"
        "----- TRANSLATED TEXT WITH LINE NUMBERS -----\n"
        f"{add_line_numbers(translated)}"
    )

    return system_prompt, user_prompt


def normalize_summary_for_body(summary: str) -> str:
    """
    GitHub review body로 쓸 텍스트 정리.
    """
    s = (summary or "").strip()
    if not s:
        return "LLM translation review"

    if s.startswith("{") or s.startswith("["):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                inner = obj.get("summary")
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
        except Exception:
            return s

    return s


# ----------------------- Parsing & GitHub glue ----------------------

def _extract_json_candidates(raw_response: str) -> List[str]:
    candidates: List[str] = []

    for match in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL):
        snippet = match.group(1).strip()
        if snippet:
            candidates.append(snippet)

    stripped = raw_response.strip()
    if stripped:
        candidates.append(stripped)

    return candidates


def parse_review_response(raw_response: str) -> Tuple[str, str, List[Dict[str, object]]]:
    parsed: Optional[Dict[str, object]] = None

    for candidate in _extract_json_candidates(raw_response):
        try:
            parsed_candidate = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_candidate, dict):
            parsed = parsed_candidate
            break

    if parsed is None:
        return "comment", raw_response.strip(), []

    verdict = parsed.get("verdict", "comment")
    summary = str(parsed.get("summary", "")).strip()
    comments = parsed.get("comments", [])

    if not isinstance(verdict, str):
        verdict = "comment"
    verdict = verdict.lower()
    if verdict not in {"request_changes", "comment", "approve"}:
        verdict = "comment"

    if not summary:
        summary = raw_response.strip()

    if not isinstance(comments, list):
        comments = []

    normalized_comments: List[Dict[str, object]] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue

        line = comment.get("line")
        issue = str(comment.get("issue", "")).strip()
        suggested_edit = str(comment.get("suggested_edit", "")).strip()
        context = str(comment.get("context", "")).strip()

        if not isinstance(line, int) or line <= 0:
            continue
        if not issue:
            continue

        normalized_comments.append(
            {
                "line": line,
                "issue": issue,
                "suggested_edit": suggested_edit,
                "context": context,
            }
        )

    return verdict, summary, normalized_comments


def review_event_from_verdict(verdict: str) -> str:
    return {
        "request_changes": "REQUEST_CHANGES",
        "comment": "COMMENT",
        "approve": "APPROVE",
    }.get(verdict, "COMMENT")


def build_review_comments(
    translated_path: str,
    comments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    review_comments: List[Dict[str, object]] = []

    for comment in comments:
        line = int(comment["line"])
        issue = str(comment["issue"]).strip()

        raw_suggested = comment.get("suggested_edit", "")
        if isinstance(raw_suggested, str):
            suggested_edit = raw_suggested.rstrip("\r\n")
        else:
            suggested_edit = str(raw_suggested).rstrip("\r\n") if raw_suggested else ""

        context = str(comment.get("context", "")).rstrip("\n")
        full_line_suggestion = suggested_edit.rstrip("\n") if suggested_edit else ""

        body_parts = [issue]
        if context:
            body_parts.append(f"> _Current text_: {context}")
        if full_line_suggestion:
            body_parts.append("```suggestion\n" + full_line_suggestion + "\n```")

        body = "\n\n".join(body_parts).strip()

        review_comments.append(
            {
                "path": translated_path,
                "side": "RIGHT",
                "line": line,
                "body": body,
            }
        )

    return review_comments


def attach_translated_line_context(
    translated_text: str,
    comments: List[Dict[str, object]],
) -> None:
    if not comments:
        return

    lines = translated_text.splitlines()
    for comment in comments:
        line_idx = comment.get("line")
        if not isinstance(line_idx, int):
            continue

        list_index = line_idx - 1
        if list_index < 0 or list_index >= len(lines):
            continue

        current_line = lines[list_index].rstrip("\n")
        if not comment.get("context"):
            comment["context"] = current_line


def build_github_review_payload(
    body: str,
    event: str = "COMMENT",
    comments: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {"event": event, "body": body}
    if comments:
        payload["comments"] = comments
    return payload


def submit_pr_review(
    repo_name: str,
    pr_number: int,
    github_token: str,
    body: str,
    event: str,
    comments: Optional[List[Dict[str, object]]] = None,
    allow_self_request_changes: bool = True,
) -> Tuple[Dict, str]:
    """
    GitHub PR 리뷰 전송 (self-review REQUEST_CHANGES 우회 포함).
    """
    github_token = resolve_github_token(github_token)

    url = f"{SETTINGS.github_api_base}/repos/{repo_name}/pulls/{pr_number}/reviews"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }

    def _post(event_to_use: str, body_to_use: str) -> requests.Response:
        payload = build_github_review_payload(body=body_to_use, event=event_to_use, comments=comments)
        return requests.post(url, headers=headers, json=payload, timeout=30)

    # 1차 요청
    response = _post(event, body)

    if response.status_code == 401:
        raise PermissionError("GitHub token is invalid or lacks permission to submit a review.")

    # 본인 PR + REQUEST_CHANGES 케이스 처리
    if response.status_code == 422 and event == "REQUEST_CHANGES":
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"message": response.text}

        message = str(error_payload.get("message", ""))
        errors = " ".join(str(item) for item in error_payload.get("errors", []))
        combined_error = f"{message} {errors}".strip()

        if "own pull request" in combined_error.lower():
            if not allow_self_request_changes:
                raise RuntimeError(
                    "GitHub does not allow REQUEST_CHANGES on your own pull request: "
                    + combined_error
                )

            fallback_event = "COMMENT"
            fallback_body = "[REQUEST_CHANGES (self-review)]\n\n" + (body or "").strip()

            comment_response = _post(fallback_event, fallback_body)
            if comment_response.status_code >= 400:
                raise RuntimeError(
                    "Failed to submit fallback self-review comment: "
                    f"HTTP {comment_response.status_code} - {comment_response.text}"
                )
            return comment_response.json(), "REQUEST_CHANGES_SELF"

    if response.status_code >= 400:
        raise RuntimeError(
            "Failed to submit review: "
            f"HTTP {response.status_code} - {response.text}"
        )

    return response.json(), event


# --------------------- High-level domain services ------------------

def prepare_translation_context(
    github_token: str,
    pr_url: str,
    original_path: str,
    translated_path: str,
) -> Dict[str, object]:
    """
    PR에서 파일을 가져와 system/user prompt까지 구성.
    """
    repo_name, pr_number, original, translated = load_pr_files(
        github_token=github_token,
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
    )

    system_prompt, user_prompt = build_messages(
        original=original,
        translated=translated,
        pr_number=pr_number,
        pr_url=pr_url,
    )

    return {
        "repo": repo_name,
        "pr_number": pr_number,
        "original": original,
        "translated": translated,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def review_and_emit_payload(
    provider: str,
    provider_token: str,
    model_name: str,
    pr_url: str,
    translated_path: str,
    original: str,
    translated: str,
) -> Dict[str, object]:
    """
    LLM 리뷰 수행 후 verdict / summary / comments 및 GitHub payload 생성.
    """
    _, pr_number = parse_pr_url(pr_url)

    system_prompt, user_prompt = build_messages(
        original=original,
        translated=translated,
        pr_number=pr_number,
        pr_url=pr_url,
    )

    raw = dispatch_review(
        provider=provider,
        token=provider_token,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_name=model_name,
    )

    verdict, summary, comments = parse_review_response(raw)
    attach_translated_line_context(translated, comments)

    event = review_event_from_verdict(verdict)
    github_comments = build_review_comments(translated_path, comments)

    payload = build_github_review_payload(
        body=summary,
        event=event,
        comments=github_comments,
    )

    return {
        "verdict": verdict,
        "summary": summary,
        "comments": comments,
        "event": event,
        "payload": payload,
    }


def submit_review_to_github(
    github_token: str,
    pr_url: str,
    translated_path: str,
    payload_or_review: Dict[str, object],
    allow_self_request_changes: bool = True,
) -> Dict[str, object]:
    """
    payload JSON 또는 review JSON을 입력받아 GitHub 리뷰 제출.
    """
    repo, pr_number = parse_pr_url(pr_url)

    event = payload_or_review.get("event")
    body = payload_or_review.get("body")
    comments_obj = payload_or_review.get("comments")

    comments: Optional[List[Dict[str, object]]] = None

    if isinstance(event, str) and body:
        # 이미 GitHub payload 형식
        event_str = event
        if isinstance(comments_obj, list):
            comments = comments_obj
        body_str = str(body)
    else:
        # review 형식 (verdict/summary/comments)
        verdict = str(payload_or_review.get("verdict", "comment")).lower()
        summary = str(payload_or_review.get("summary", "")).strip()
        review_comments = payload_or_review.get("comments", [])
        if not isinstance(review_comments, list):
            review_comments = []

        event_str = review_event_from_verdict(verdict)
        body_str = summary if summary else "LLM translation review"
        comments = build_review_comments(translated_path, review_comments)

    if event_str == "REQUEST_CHANGES" and not body_str.strip() and not comments:
        raise ValueError(
            "REQUEST_CHANGES를 보내려면 review 본문 또는 코멘트가 하나 이상 필요합니다."
        )

    response, final_event = submit_pr_review(
        repo_name=repo,
        pr_number=pr_number,
        github_token=github_token,
        body=body_str,
        event=event_str,
        comments=comments,
        allow_self_request_changes=allow_self_request_changes,
    )

    return {
        "final_event": final_event,
        "response": response,
    }


def run_end_to_end(
    provider: str,
    provider_token: str,
    model_name: str,
    github_token: str,
    pr_url: str,
    original_path: str,
    translated_path: str,
    save_review: bool = False,
    save_path: str = "review.json",
    submit_review_flag: bool = False,
) -> Dict[str, object]:
    repo, pr_number, original, translated = load_pr_files(
        github_token=github_token,
        pr_url=pr_url,
        original_path=original_path,
        translated_path=translated_path,
    )

    review = review_and_emit_payload(
        provider=provider,
        provider_token=provider_token,
        model_name=model_name,
        pr_url=pr_url,
        translated_path=translated_path,
        original=original,
        translated=translated,
    )

    out: Dict[str, object] = {
        "repo": repo,
        "pr_number": pr_number,
        "review": review,
    }

    if save_review:
        Path(save_path).write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        out["saved_to"] = save_path

    if submit_review_flag:
        submission = submit_review_to_github(
            github_token=github_token,
            pr_url=pr_url,
            translated_path=translated_path,
            payload_or_review=review.get("payload") if isinstance(review.get("payload"), dict) else review,
        )
        out["submission"] = submission

    return out
