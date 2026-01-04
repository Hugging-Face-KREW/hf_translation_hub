#!/usr/bin/env python3
"""
Gradio + MCP server app for LLM translation review on GitHub PRs.

- UI는 입력/출력만 담당
- 실제 로직은 tools / services / adapters 에 위임
- MCP tool에서는 인증 인자를 받지 않는다
"""

from __future__ import annotations

import json
import gradio as gr

from setting import SETTINGS
from tools import (
    tool_prepare,
    tool_review_and_emit,
    tool_submit_review,
    tool_end_to_end,
)


def _error_payload(err: Exception) -> dict:
    """Return a structured error payload for Gradio / MCP JSON outputs."""
    return {"error": str(err), "type": err.__class__.__name__}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown(
            "# LLM Translation Reviewer for GitHub PRs (MCP-enabled)\n"
            "Fetch prompts + files here, run your own LLM client, then paste the response to build review payloads."
        )

        # ------------------------------------------------------------------
        # Common inputs (UI-only token input; NOT used by MCP tools)
        # ------------------------------------------------------------------
        with gr.Row():
            pr_url = gr.Textbox(
                label="PR URL",
                placeholder="https://github.com/owner/repo/pull/123",
                scale=2,
            )
            github_token = gr.Textbox(
                label="GitHub Token (UI only)",
                type="password",
            )

        with gr.Row():
            original_path = gr.Textbox(
                label="Original File Path (in repo)",
                placeholder="_posts/2025-09-29-building-hf-mcp.md",
            )
            translated_path = gr.Textbox(
                label="Translated File Path (in repo)",
                placeholder="_posts/2025-09-29-building-hf-mcp-ko.md",
            )

        gr.Markdown("---")

        # ------------------------------------------------------------------
        # Tool 1: Prepare (MCP-safe)
        # ------------------------------------------------------------------
        with gr.Accordion(
            "Tool 1: Prepare (Fetch Files + Build Prompts)", open=False
        ):
            prepare_toolcallid = gr.Textbox(
                label="Tool Call ID",
                value="",
                visible=False,
            )
            prepare_btn = gr.Button("tool_prepare")
            prepare_out = gr.JSON(label="Prepare result (files + prompts)")

            def _prepare_proxy(
                pr_url_: str,
                original_path_: str,
                translated_path_: str,
                toolCallId: str = "",
            ):
                """
                Fetch files from GitHub PR and build translation review prompts.

                MCP-safe:
                - Does NOT accept github_token as an argument
                - Token is resolved internally from environment / secrets

                Args:
                    pr_url_: The URL of the GitHub PR.
                    original_path_: The path of the original file.
                    translated_path_: The path of the translated file.
                    toolCallId: The ID of the tool call.

                Returns:
                    A dictionary containing the translation review results.

                Raises:
                    Exception: If the translation review cannot be prepared.
                """
                try:
                    return tool_prepare(
                        pr_url=pr_url_,
                        original_path=original_path_,
                        translated_path=translated_path_,
                        toolCallId=toolCallId,
                    )
                except Exception as err:
                    return _error_payload(err)

            prepare_btn.click(
                fn=_prepare_proxy,
                inputs=[pr_url, original_path, translated_path, prepare_toolcallid],
                outputs=[prepare_out],
                api_name="translation_prepare",
                api_description="Fetch files and build prompts for translation review",
            )

        # ------------------------------------------------------------------
        # Tool 2: Review + Emit Payload (token not required)
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 2: Review + Emit Payload", open=False):
            translated_text = gr.Textbox(
                label="Translated (for review)",
                lines=10,
            )
            raw_response = gr.Textbox(
                label="LLM review response (from your client)",
                lines=10,
                placeholder="Paste the JSON/text returned by your LLM here",
            )
            review_toolcallid = gr.Textbox(
                label="Tool Call ID",
                value="",
                visible=False,
            )
            review_btn = gr.Button("tool_review_and_emit")

            review_out = gr.JSON(
                label="Review result (verdict/summary/comments/event)"
            )
            payload_out = gr.JSON(label="Payload JSON (for GitHub)")

            def _review_emit_proxy(
                pr_url_: str,
                translated_path_: str,
                translated_text_: str,
                raw_response_: str,
                toolCallId: str = "",
            ):
                """
                Review and emit payload for translation review.
                - Takes the PR URL, path to translated file, the translated text, and the LLM's raw review response.
                - Calls the review tool to parse and analyze the translation.
                - Emits both the review result and the payload JSON for GitHub PR submission.

                Args:
                    pr_url_: The URL of the GitHub PR.
                    translated_path_: The path to the translated file.
                    translated_text_: The translated text content.
                    raw_response_: The raw response from the LLM reviewer (JSON/text).
                    toolCallId: Optional tool call ID for tracking.

                Returns:
                    Tuple containing:
                        - Review result (dict): Includes verdict, summary, comments, etc.
                        - Payload JSON (dict): Structured payload ready for GitHub PR API.
                """

                try:
                    result = tool_review_and_emit(
                        pr_url=pr_url_,
                        translated_path=translated_path_,
                        translated=translated_text_,
                        raw_review_response=raw_response_,
                        toolCallId=toolCallId,
                    )
                    return result, result.get("payload", {})
                except Exception as err:
                    error = _error_payload(err)
                    return error, error

            review_btn.click(
                fn=_review_emit_proxy,
                inputs=[
                    pr_url,
                    translated_path,
                    translated_text,
                    raw_response,
                    review_toolcallid,
                ],
                outputs=[review_out, payload_out],
                api_name="translation_review_and_emit",
                api_description="Review the translated content and emit the payload for GitHub PR",
            )

        # ------------------------------------------------------------------
        # Tool 3: Submit Review (MCP-safe)
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 3: Submit Review", open=False):
            payload_in = gr.Textbox(
                label="Payload or Review JSON (from Tool 2)",
                lines=6,
            )
            submit_toolcallid = gr.Textbox(
                label="Tool Call ID",
                value="",
                visible=False,
            )
            submit_btn = gr.Button("tool_submit_review")
            submit_out = gr.JSON(label="Submission result")

            def _submit_proxy(
                pr_url_: str,
                translated_path_: str,
                payload_json_: str,
                toolCallId: str = "",
            ):
                """
                Submit review payload to GitHub PR.

                MCP-safe:
                - Token resolved internally

                Args:
                    pr_url_: The URL of the GitHub PR.
                    translated_path_: The path of the translated file.
                    payload_json_: The payload or review to submit (as a JSON string).
                    toolCallId: The ID of the tool call.

                Returns:
                    A dictionary containing the review submission results.

                Raises:
                    Exception: If the review cannot be submitted.
                """
                try:
                    payload_obj = json.loads(payload_json_) if payload_json_ else {}
                except Exception as e:
                    return _error_payload(ValueError(f"Invalid JSON: {e}"))

                try:
                    return tool_submit_review(
                        pr_url=pr_url_,
                        translated_path=translated_path_,
                        payload_or_review=payload_obj,
                        toolCallId=toolCallId,
                    )
                except Exception as err:
                    return _error_payload(err)

            submit_btn.click(
                fn=_submit_proxy,
                inputs=[pr_url, translated_path, payload_in, submit_toolcallid],
                outputs=[submit_out],
                api_name="translation_submit_review",
                api_description="Submit the review payload to GitHub PR",
            )

        gr.Markdown("---")

        # ------------------------------------------------------------------
        # Tool 4: End-to-End (MCP-safe)
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 4: End-to-End", open=True):
            save_review = gr.Checkbox(
                label="Save review JSON to file", value=True
            )
            save_path = gr.Textbox(
                label="Save path", value="review.json"
            )
            submit_flag = gr.Checkbox(
                label="Submit to GitHub", value=False
            )
            e2e_raw_response = gr.Textbox(
                label="LLM review response (optional)",
                lines=6,
            )
            e2e_toolcallid = gr.Textbox(
                label="Tool Call ID",
                value="",
                visible=False,
            )
            e2e_btn = gr.Button("tool_end_to_end")
            e2e_out = gr.JSON(label="E2E result")

            def _e2e_proxy(
                pr_url_: str,
                original_path_: str,
                translated_path_: str,
                save_review_: bool,
                save_path_: str,
                submit_flag_: bool,
                e2e_raw_response_: str,
                toolCallId: str = "",
            ):
                """
                Runs the end-to-end tool: fetches files, builds prompts, parses LLM response, and optionally saves and/or submits the review.

                Args:
                    pr_url_: The URL of the GitHub PR.
                    original_path_: The path of the original file.
                    translated_path_: The path of the translated file.
                    save_review_: Whether to save the review as a JSON file.
                    save_path_: The file path to save the review JSON.
                    submit_flag_: Whether to submit the review to GitHub.
                    e2e_raw_response_: The raw LLM review response (optional).
                    toolCallId: The ID of the tool call (optional).

                Returns:
                    A dictionary containing the end-to-end execution results.
                """
                try:
                    return tool_end_to_end(
                        pr_url=pr_url_,
                        original_path=original_path_,
                        translated_path=translated_path_,
                        save_review=save_review_,
                        save_path=save_path_,
                        submit_review_flag=submit_flag_,
                        raw_review_response=e2e_raw_response_,
                        toolCallId=toolCallId,
                    )
                except Exception as err:
                    return _error_payload(err)

            e2e_btn.click(
                fn=_e2e_proxy,
                inputs=[
                    pr_url,
                    original_path,
                    translated_path,
                    save_review,
                    save_path,
                    submit_flag,
                    e2e_raw_response,
                    e2e_toolcallid,
                ],
                outputs=[e2e_out],
                api_name="translation_end_to_end",
                api_description="End-to-end translation review and submission",
            )

        gr.Markdown(
            """
            **Notes**
            - MCP-exposed tools do NOT accept authentication parameters.
            - GitHub credentials are resolved internally via environment variables / Space secrets.
            - UI token input is kept for local or UI-only flows, not for MCP.
            """
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=SETTINGS.ui_share,
        mcp_server=SETTINGS.ui_launch_mcp_server,
        ssr_mode=False,
    )