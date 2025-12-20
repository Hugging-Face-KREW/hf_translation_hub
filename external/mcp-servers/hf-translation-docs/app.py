"""HuggingFace Translation Documentation MCP Server."""

from __future__ import annotations

import os
import gradio as gr

from services import get_supported_projects
from tools import get_project_config, search_translation_files, get_file_content, generate_translation_prompt, validate_translation, save_translation_result
from setting import SETTINGS, LANGUAGE_CHOICES


def ensure_mcp_support() -> None:
    """Verify that `gradio[mcp]` is installed and enable the MCP server flag."""
    try:
        import gradio.mcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install gradio[mcp] before launching this module.") from exc

    os.environ.setdefault("GRADIO_MCP_SERVER", "true")
    os.environ.setdefault("GRADIO_SHOW_API", "true")


def build_ui() -> gr.Blocks:
    """Create a lightweight Gradio Blocks UI for testing MCP tools."""
    projects = get_supported_projects()
    languages = [lang[1] for lang in LANGUAGE_CHOICES]  # Extract language codes

    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown("# HuggingFace Translation Documentation MCP Server\nTest the MCP tools below.")

        # --- 1) Get Project Config ---
        with gr.Tab("Project Config"):
            project_input = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            config_output = gr.JSON(label="Project Configuration")
            gr.Button("Get Config").click(
                fn=get_project_config,
                inputs=[project_input],
                outputs=config_output,
                api_name="translation_get_project_config",
            )

        # --- 2) Search Translation Files ---
        with gr.Tab("Search Files"):
            search_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            search_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            search_limit = gr.Number(
                label="Max Files",
                value=SETTINGS.default_limit,
                minimum=1,
                maximum=100,
            )
            search_output = gr.JSON(label="Search Results")
            gr.Button("Search Files").click(
                fn=search_translation_files,
                inputs=[search_project, search_language, search_limit],
                outputs=search_output,
                api_name="translation_search_files",
            )

        # --- 3) Get File Content ---
        with gr.Tab("Get File Content"):
            content_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            content_file_path = gr.Textbox(
                label="File Path",
                placeholder="docs/source/en/model_doc/bert.md",
            )
            content_include_metadata = gr.Checkbox(
                label="Include Metadata",
                value=True,
            )
            content_output = gr.JSON(label="File Content")
            gr.Button("Get Content").click(
                fn=get_file_content,
                inputs=[content_project, content_file_path, content_include_metadata],
                outputs=content_output,
                api_name="translation_get_file_content",
            )

        # --- 4) Generate Translation Prompt ---
        with gr.Tab("Generate Prompt"):
            prompt_target_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            prompt_content = gr.Textbox(
                label="Content to Translate",
                placeholder="Enter markdown content...",
                lines=5,
            )
            prompt_additional = gr.Textbox(
                label="Additional Instructions",
                placeholder="Optional additional instructions...",
                lines=2,
            )
            prompt_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            prompt_file_path = gr.Textbox(
                label="File Path (optional)",
                placeholder="docs/source/en/model_doc/bert.md",
            )
            prompt_output = gr.JSON(label="Translation Prompt")
            gr.Button("Generate Prompt").click(
                fn=generate_translation_prompt,
                inputs=[prompt_target_language, prompt_content, prompt_additional, prompt_project, prompt_file_path],
                outputs=prompt_output,
                api_name="translation_generate_prompt",
            )

        # --- 5) Validate Translation ---
        with gr.Tab("Validate Translation"):
            validate_original = gr.Textbox(
                label="Original Content",
                placeholder="Enter original content...",
                lines=5,
            )
            validate_translated = gr.Textbox(
                label="Translated Content",
                placeholder="Enter translated content...",
                lines=5,
            )
            validate_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            validate_file_path = gr.Textbox(
                label="File Path (optional)",
                placeholder="docs/source/en/model_doc/bert.md",
            )
            validate_output = gr.JSON(label="Validation Results")
            gr.Button("Validate").click(
                fn=validate_translation,
                inputs=[validate_original, validate_translated, validate_language, validate_file_path],
                outputs=validate_output,
                api_name="translation_validate",
            )

        # --- 6) Save Translation Result ---
        with gr.Tab("Save Result"):
            save_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            save_original_path = gr.Textbox(
                label="Original File Path",
                placeholder="docs/source/en/model_doc/bert.md",
            )
            save_content = gr.Textbox(
                label="Translated Content",
                placeholder="Enter translated content to save...",
                lines=8,
            )
            save_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            save_output = gr.JSON(label="Save Results")
            gr.Button("Save Translation").click(
                fn=save_translation_result,
                inputs=[save_project, save_original_path, save_content, save_language],
                outputs=save_output,
                api_name="translation_save_result",
            )

    return demo


def main():
    """Main entry point for the MCP server."""
    ensure_mcp_support()
    
    ui = build_ui()
    
    # Launch with MCP server enabled
    ui.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
        mcp_server=True
    )


if __name__ == "__main__":
    main()