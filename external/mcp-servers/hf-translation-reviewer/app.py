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
    return {"error": str(err), "type": err.__class__.__name__}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown(
            "# LLM Translation Reviewer for GitHub PRs (MCP-enabled)\n"
            "You can use tools sequentially or use the end-to-end tool depending on your workflow."
        )

        # ------------------------------------------------------------------
        # Common inputs
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
                label="Original File Path",
                placeholder="docs/source/en/example.md",
            )
            translated_path = gr.Textbox(
                label="Translated File Path",
                placeholder="docs/source/ko/example.md",
            )

        gr.Markdown("---")

        # ------------------------------------------------------------------
        # Tool 1: Prepare
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 1: Prepare", open=False):
            prepare_toolcallid = gr.Textbox(visible=False)
            prepare_btn = gr.Button("translation_prepare")
            prepare_out = gr.JSON(label="Prepare result")

            def _prepare_proxy(
                pr_url_: str,
                original_path_: str,
                translated_path_: str,
                toolCallId: str = "",
            ):
                try:
                    return tool_prepare(
                        pr_url=pr_url_,
                        original_path=original_path_,
                        translated_path=translated_path_,
                    )
                except Exception as err:
                    return _error_payload(err)

            prepare_btn.click(
                fn=_prepare_proxy,
                inputs=[pr_url, original_path, translated_path, prepare_toolcallid],
                outputs=[prepare_out],
                api_name="translation_prepare",
            )

        # ------------------------------------------------------------------
        # Tool 2: Review + Emit
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 2: Review + Emit", open=False):
            translated_text = gr.Textbox(
                label="Translated text (full file content)",
                lines=10,
            )
            raw_response = gr.Textbox(
                label="LLM review JSON",
                lines=10,
            )
            review_toolcallid = gr.Textbox(visible=False)
            review_btn = gr.Button("translation_review_and_emit")

            review_out = gr.JSON(label="Review result")
            payload_out = gr.JSON(label="GitHub payload")

            def _review_emit_proxy(
                pr_url_: str,
                translated_path_: str,
                translated: str,
                raw_response_: str,
                toolCallId: str = "",
            ):
                try:
                    result = tool_review_and_emit(
                        pr_url=pr_url_,
                        translated_path=translated_path_,
                        translated=translated,
                        raw_review_response=raw_response_,
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
            )

        # ------------------------------------------------------------------
        # Tool 3: Submit Review
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 3: Submit Review", open=False):
            payload_in = gr.Textbox(
                label="Payload JSON",
                lines=6,
            )
            submit_toolcallid = gr.Textbox(visible=False)
            submit_btn = gr.Button("translation_submit_review")
            submit_out = gr.JSON(label="Submission result")

            def _submit_proxy(
                pr_url_: str,
                translated_path_: str,
                payload_json_: str,
                toolCallId: str = "",
            ):
                try:
                    payload_obj = json.loads(payload_json_) if payload_json_ else {}
                except Exception as e:
                    return _error_payload(ValueError(f"Invalid JSON: {e}"))

                try:
                    return tool_submit_review(
                        pr_url=pr_url_,
                        translated_path=translated_path_,
                        payload_or_review=payload_obj,
                    )
                except Exception as err:
                    return _error_payload(err)

            submit_btn.click(
                fn=_submit_proxy,
                inputs=[pr_url, translated_path, payload_in, submit_toolcallid],
                outputs=[submit_out],
                api_name="translation_submit_review",
            )

        # ------------------------------------------------------------------
        # Tool 4: End-to-End (ENABLED)
        # ------------------------------------------------------------------
        with gr.Accordion("Tool 4: End-to-End", open=True):
            save_review = gr.Checkbox(
                label="Save review JSON to file",
                value=False,
            )
            save_path = gr.Textbox(
                label="Save path",
                value="review.json",
            )
            submit_flag = gr.Checkbox(
                label="Submit to GitHub",
                value=False,
            )
            e2e_raw_response = gr.Textbox(
                label="LLM review JSON (optional)",
                lines=6,
                placeholder="If empty, e2e will only prepare context and prompts.",
            )
            e2e_toolcallid = gr.Textbox(visible=False)
            e2e_btn = gr.Button("translation_end_to_end")
            e2e_out = gr.JSON(label="E2E result")

            def _e2e_proxy(
                pr_url_: str,
                original_path_: str,
                translated_path_: str,
                save_review_: bool,
                save_path_: str,
                submit_flag_: bool,
                raw_review_response_: str,
                toolCallId: str = "",
            ):
                try:
                    return tool_end_to_end(
                        pr_url=pr_url_,
                        original_path=original_path_,
                        translated_path=translated_path_,
                        save_review=save_review_,
                        save_path=save_path_,
                        submit_review_flag=submit_flag_,
                        raw_review_response=raw_review_response_,
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
            )

        gr.Markdown(
            """
            **Notes**
            - End-to-end(e2e) tool is enabled.
            - For n8n workflows, prefer:
              `translation_prepare → LLM → translation_review_and_emit → translation_submit_review`.
            - e2e is best used as a convenience wrapper or inspection tool.
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
