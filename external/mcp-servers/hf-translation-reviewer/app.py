#!/usr/bin/env python3
"""
Gradio + MCP server app for LLM translation review on GitHub PRs.

- UI만 담당하고, 실제 로직은 tools/services/adapters 로 분리.
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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown(
            "# LLM Translation Reviewer for GitHub PRs (MCP-enabled)\n"
            "Only **PR URL** + fields below are required. Repo/PR number are parsed."
        )

        # 공통 입력 영역
        with gr.Row():
            pr_url = gr.Textbox(
                label="PR URL",
                placeholder="https://github.com/owner/repo/pull/123",
                scale=2,
            )
            provider = gr.Dropdown(
                label="Provider",
                choices=["openai", "anthropic", "gemini"],
                value=SETTINGS.default_provider,
            )
            model_name = gr.Textbox(
                label="Model name",
                value=SETTINGS.default_model,
                placeholder=(
                    "e.g., gpt-5 / gpt-4o / claude-3-5-sonnet-20240620 / gemini-1.5-pro"
                ),
            )
        with gr.Row():
            provider_token = gr.Textbox(
                label="Provider API Token",
                type="password",
            )
            github_token = gr.Textbox(
                label="GitHub Token",
                type="password",
            )
        with gr.Row():
            original_path = gr.Textbox(
                label="Original File Path (in repo)",
                placeholder="docs/source/en/xxx.md",
            )
            translated_path = gr.Textbox(
                label="Translated File Path (in repo)",
                placeholder="docs/source/ko/xxx.md",
            )

        gr.Markdown("---")

        # Tool 1: Prepare
        with gr.Accordion(
            "Tool 1: Prepare (Fetch Files + Build Prompts)", open=False
        ):
            prepare_btn = gr.Button("tool_prepare")
            prepare_out = gr.JSON(label="Prepare result (files + prompts)")

            prepare_btn.click(
                fn=tool_prepare,
                inputs=[github_token, pr_url, original_path, translated_path],
                outputs=[prepare_out],
            )

        # Tool 2: Review + Emit Payload
        with gr.Accordion("Tool 2: Review + Emit Payload", open=False):
            review_btn = gr.Button("tool_review_and_emit")

            original_text = gr.Textbox(
                label="Original (for review)",
                lines=6,
            )
            translated_text = gr.Textbox(
                label="Translated (for review)",
                lines=10,
            )

            review_out = gr.JSON(
                label="Review result (verdict/summary/comments/event)"
            )
            payload_out = gr.JSON(label="Payload JSON (for GitHub)")

            def _review_emit_proxy(
                provider_: str,
                provider_token_: str,
                model_name_: str,
                pr_url_: str,
                translated_path_: str,
                original_text_: str,
                translated_text_: str,
            ):
                result = tool_review_and_emit(
                    provider=provider_,
                    provider_token=provider_token_,
                    model_name=model_name_,
                    pr_url=pr_url_,
                    translated_path=translated_path_,
                    original=original_text_,
                    translated=translated_text_,
                )
                return result, result.get("payload", {})

            review_btn.click(
                fn=_review_emit_proxy,
                inputs=[
                    provider,
                    provider_token,
                    model_name,
                    pr_url,
                    translated_path,
                    original_text,
                    translated_text,
                ],
                outputs=[review_out, payload_out],
            )

        # Tool 3: Submit Review
        with gr.Accordion("Tool 3: Submit Review", open=False):
            submit_btn = gr.Button("tool_submit_review")
            payload_in = gr.Textbox(
                label="Payload or Review JSON (from Tool 2)",
                lines=6,
            )
            submit_out = gr.JSON(label="Submission result")

            def _submit_proxy(
                github_token_: str,
                pr_url_: str,
                translated_path_: str,
                payload_json_: str,
            ):
                try:
                    payload_obj = json.loads(payload_json_) if payload_json_ else {}
                except Exception as e:
                    raise ValueError(f"Invalid JSON: {e}")
                return tool_submit_review(
                    github_token=github_token_,
                    pr_url=pr_url_,
                    translated_path=translated_path_,
                    payload_or_review=payload_obj,
                    allow_self_request_changes=True,
                )

            submit_btn.click(
                fn=_submit_proxy,
                inputs=[github_token, pr_url, translated_path, payload_in],
                outputs=[submit_out],
            )

        gr.Markdown("---")

        # Tool 4: End-to-End
        with gr.Accordion("Tool 4: End-to-End", open=True):
            e2e_btn = gr.Button("tool_end_to_end")
            save_review = gr.Checkbox(
                label="Save review JSON to file", value=True
            )
            save_path = gr.Textbox(
                label="Save path", value="review.json"
            )
            submit_flag = gr.Checkbox(
                label="Submit to GitHub", value=False
            )
            e2e_out = gr.JSON(label="E2E result")

            e2e_btn.click(
                fn=tool_end_to_end,
                inputs=[
                    provider,
                    provider_token,
                    model_name,
                    github_token,
                    pr_url,
                    original_path,
                    translated_path,
                    save_review,
                    save_path,
                    submit_flag,
                ],
                outputs=[e2e_out],
            )

        gr.Markdown(
            """
            **Notes**
            - Tool 1: PR에서 파일을 읽고 프롬프트까지 준비합니다.
            - Tool 2: LLM으로 리뷰한 뒤, GitHub 리뷰 payload까지 생성합니다.
            - Tool 3: Tool 2에서 만든 payload JSON을 그대로 넣고 GitHub에 전송합니다.
            - Tool 4: 파일 로드부터 리뷰/저장/제출까지 한 번에 처리하는 end-to-end 툴입니다.
            - `launch(mcp_server=True)` 이므로 각 `tool_*` 버튼은 MCP 툴로도 사용 가능합니다.
            """
        )
    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        share=SETTINGS.ui_share,
        mcp_server=SETTINGS.ui_launch_mcp_server,
    )
