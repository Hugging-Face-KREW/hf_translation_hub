"""HuggingFace Translation PR Generator MCP Server."""

from __future__ import annotations

import os
import gradio as gr

from services import get_supported_projects, validate_pr_config_data
from tools import validate_pr_config, search_reference_pr, analyze_translation, generate_pr_draft, create_github_pr
from setting import SETTINGS, LANGUAGE_CHOICES


def ensure_mcp_support() -> None:
    """Verify that `gradio[mcp]` is installed and enable the MCP server flag."""
    try:
        import gradio.mcp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install gradio[mcp] before launching this module.") from exc

    os.environ.setdefault("GRADIO_MCP_SERVER", "true")
    os.environ.setdefault("GRADIO_SHOW_API", "true")


def validate_pr_config_ui(github_token: str, owner: str, repo_name: str, project: str):
    """UI wrapper for validate_pr_config that handles token input."""
    # If token provided via UI, temporarily set it in environment
    original_token = None
    if github_token.strip():
        original_token = os.environ.get("GITHUB_TOKEN")
        os.environ["GITHUB_TOKEN"] = github_token.strip()
    
    try:
        # Call the MCP endpoint function (which reads from environment)
        result = validate_pr_config(owner, repo_name, project)
        return result
    finally:
        # Restore original token if we changed it
        if original_token is not None:
            os.environ["GITHUB_TOKEN"] = original_token
        elif github_token.strip():
            # Remove the temporary token
            os.environ.pop("GITHUB_TOKEN", None)


def build_ui() -> gr.Blocks:
    """Create a lightweight Gradio Blocks UI for testing MCP tools."""
    projects = get_supported_projects()
    languages = [lang[1] for lang in LANGUAGE_CHOICES]

    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown("# HuggingFace Translation PR Generator MCP Server\nTest the MCP tools below.")

        # --- 1) Validate PR Config ---
        with gr.Tab("Validate PR Config"):
            gr.Markdown("""
            **For MCP Client**: Set `GITHUB_TOKEN` environment variable  
            **For UI Testing**: Enter token below (will be used temporarily)
            """)
            config_github_token = gr.Textbox(
                label="GitHub Token (UI Testing Only)", 
                type="password",
                placeholder="Leave empty to use GITHUB_TOKEN env var"
            )
            config_owner = gr.Textbox(label="GitHub Owner")
            config_repo = gr.Textbox(label="Repository Name")
            config_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            config_output = gr.JSON(label="Validation Results")
            gr.Button("Validate Config").click(
                fn=validate_pr_config_ui,
                inputs=[config_github_token, config_owner, config_repo, config_project],
                outputs=config_output,
                api_name="pr_validate_config",
                api_description="Validate the GitHub PR configuration and settings",
            )

        # --- 2) Search Reference PR ---
        with gr.Tab("Search Reference PR"):
            search_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            search_context = gr.Textbox(
                label="Context",
                value="documentation translation",
                placeholder="Context for PR search..."
            )
            search_output = gr.JSON(label="Search Results")
            gr.Button("Search Reference PR").click(
                fn=search_reference_pr,
                inputs=[search_language, search_context],
                outputs=search_output,
                api_name="pr_search_reference",
                api_description="Search for reference PRs for translation",
            )

        # --- 3) Analyze Translation ---
        with gr.Tab("Analyze Translation"):
            analyze_filepath = gr.Textbox(label="File Path", placeholder="docs/source/en/model_doc/bert.md")
            analyze_content = gr.Textbox(
                label="Translated Content",
                lines=8,
                placeholder="Enter translated content to analyze..."
            )
            analyze_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            analyze_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            analyze_output = gr.JSON(label="Analysis Results")
            gr.Button("Analyze Translation").click(
                fn=analyze_translation,
                inputs=[analyze_filepath, analyze_content, analyze_language, analyze_project],
                outputs=analyze_output,
                api_name="pr_analyze_translation",
                api_description="Analyze the translated content for quality and formatting",
            )

        # --- 4) Generate PR Draft ---
        with gr.Tab("Generate PR Draft"):
            draft_filepath = gr.Textbox(label="File Path", placeholder="docs/source/en/model_doc/bert.md")
            draft_content = gr.Textbox(
                label="Translated Content",
                lines=8,
                placeholder="Enter translated content..."
            )
            draft_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            draft_reference_pr = gr.Textbox(
                label="Reference PR URL",
                placeholder="https://github.com/..."
            )
            draft_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            draft_output = gr.JSON(label="PR Draft")
            gr.Button("Generate Draft").click(
                fn=generate_pr_draft,
                inputs=[draft_filepath, draft_content, draft_language, draft_reference_pr, draft_project],
                outputs=draft_output,
                api_name="pr_generate_draft",
                api_description="Generate a PR draft for the translated content",
            )

        # --- 5) Create GitHub PR ---
        with gr.Tab("Create GitHub PR"):
            create_github_token = gr.Textbox(label="GitHub Token", type="password")
            create_owner = gr.Textbox(label="GitHub Owner")
            create_repo = gr.Textbox(label="Repository Name")
            create_filepath = gr.Textbox(label="File Path", placeholder="docs/source/en/model_doc/bert.md")
            create_content = gr.Textbox(
                label="Translated Content",
                lines=8,
                placeholder="Enter translated content..."
            )
            create_language = gr.Dropdown(
                choices=languages,
                label="Target Language",
                value=SETTINGS.default_language,
            )
            create_reference_pr = gr.Textbox(
                label="Reference PR URL",
                placeholder="https://github.com/..."
            )
            create_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=SETTINGS.default_project,
            )
            create_output = gr.JSON(label="PR Creation Results")
            gr.Button("Create PR").click(
                fn=create_github_pr,
                inputs=[
                    create_github_token, create_owner, create_repo,
                    create_filepath, create_content, create_language,
                    create_reference_pr, create_project
                ],
                outputs=create_output,
                api_name="pr_create_github_pr",
                api_description="Create a GitHub PR with the translated content",
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