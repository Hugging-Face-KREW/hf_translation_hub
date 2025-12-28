from __future__ import annotations

import os
import gradio as gr

from services import get_available_projects, LANGUAGE_CHOICES
from tools import (
    list_projects,
    search_files,
    list_missing_files,
    list_outdated_files,
)
from setting import SETTINGS


def ensure_mcp_support() -> None:
    try:
        import gradio.mcp  # noqa
    except ImportError as exc:
        raise RuntimeError("Install gradio[mcp] before launching this module.") from exc

    os.environ.setdefault("GRADIO_MCP_SERVER", "true")
    os.environ.setdefault("GRADIO_SHOW_API", "true")


def build_ui() -> gr.Blocks:
    projects = get_available_projects()
    languages = LANGUAGE_CHOICES[:]

    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown("# Translation MCP Server\nExtended with outdated detection.")

        # --- 1) Project catalog ---
        with gr.Tab("Project catalog"):
            catalog_output = gr.JSON(label="catalog")
            gr.Button("Fetch").click(
                fn=list_projects,
                inputs=[],
                outputs=catalog_output,
                api_name="translation_project_catalog",
                api_description="Fetch the list of projects from the Hugging Face Translation Documentation MCP Server",
            )

        # --- 2) File search ---
        with gr.Tab("File search"):
            project_input = gr.Dropdown(
                choices=projects,
                label="Project",
                value=projects[0] if projects else "",
            )
            lang_input = gr.Dropdown(
                choices=languages,
                label="Language",
                value=SETTINGS.default_language,
            )
            limit_input = gr.Number(
                label="Limit",
                value=SETTINGS.default_limit,
                minimum=1,
            )
            include_report = gr.Checkbox(
                label="Include status report",
                value=True,
            )

            search_output = gr.JSON(label="search result")
            gr.Button("Search").click(
                fn=search_files,
                inputs=[project_input, lang_input, limit_input, include_report],
                outputs=search_output,
                api_name="translation_file_search",
                api_description="Search for files that need translation in a Hugging Face project",
            )

        # --- 3) Missing docs ---
        with gr.Tab("Missing docs"):
            missing_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=projects[0] if projects else "",
            )
            missing_lang = gr.Dropdown(
                choices=languages,
                label="Language",
                value=SETTINGS.default_language,
            )
            missing_limit = gr.Number(
                label="Limit",
                value=max(SETTINGS.default_limit, 20),
                minimum=1,
            )

            missing_output = gr.JSON(label="missing files")
            gr.Button("List missing").click(
                fn=list_missing_files,
                inputs=[missing_project, missing_lang, missing_limit],
                outputs=missing_output,
                api_name="translation_missing_list",
                api_description="List the missing files in a Hugging Face project",
            )

        # --- 4) Outdated docs (NEW) ---
        with gr.Tab("Outdated docs"):
            outdated_project = gr.Dropdown(
                choices=projects,
                label="Project",
                value=projects[0] if projects else "",
            )
            outdated_lang = gr.Dropdown(
                choices=languages,
                label="Language",
                value=SETTINGS.default_language,
            )
            outdated_limit = gr.Number(
                label="Limit",
                value=max(SETTINGS.default_limit, 20),
                minimum=1,
            )

            outdated_output = gr.JSON(label="outdated files")
            gr.Button("List outdated").click(
                fn=list_outdated_files,
                inputs=[outdated_project, outdated_lang, outdated_limit],
                outputs=outdated_output,
                api_name="translation_outdated_list",
                api_description="List the outdated files in a Hugging Face project",
            )

    return demo


ensure_mcp_support()

ui = build_ui()
ui.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", "7860")),
    share=False,
    show_api=True,
    mcp_server=True,
)
