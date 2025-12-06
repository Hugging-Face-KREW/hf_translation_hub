from __future__ import annotations

import argparse
import os

import gradio as gr

from services import get_available_projects, LANGUAGE_CHOICES
from tools import list_projects, search_files, list_missing_files
from setting import SETTINGS


def ensure_mcp_support() -> None:
    """Verify that ``gradio[mcp]`` is installed and enable the MCP server flag."""
    try:
        import gradio.mcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("Install gradio[mcp] before launching this module.") from exc
    os.environ.setdefault("GRADIO_MCP_SERVER", "true")


def build_demo() -> gr.Blocks:
    """Create a lightweight Gradio Blocks UI for exercising the MCP tools."""
    projects = get_available_projects()
    languages = LANGUAGE_CHOICES[:]

    with gr.Blocks(title=SETTINGS.ui_title) as demo:
        gr.Markdown("# Translation MCP Server\nTry the MCP tools exposed below.")

        # --- 1) Project catalog ---
        with gr.Tab("Project catalog"):
            catalog_output = gr.JSON(label="catalog")
            gr.Button("Fetch").click(
                fn=list_projects,
                inputs=[],  # 인자 없음
                outputs=catalog_output,
                api_name="translation_project_catalog",
            )

        # --- 2) File search (report + candidates) ---
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
            )

        # --- 3) Missing docs only ---
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
            )

    return demo


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments used for local or Space deployments."""
    parser = argparse.ArgumentParser(description="Launch the translation MCP demo.")

    parser.add_argument(
        "--as-space",
        action="store_true",
        help="Use Hugging Face Space defaults.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link.",
    )
    parser.add_argument(
        "--no-queue",
        dest="queue",
        action="store_false",
        help="Disable the request queue.",
    )
    parser.set_defaults(queue=True)

    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Launch the Gradio app with MCP server support enabled."""
    args = _parse_args(argv)

    ensure_mcp_support()

    launch_kwargs = {"mcp_server": True}

    if args.as_space or os.environ.get("SPACE_ID"):
        launch_kwargs.update(
            {
                "server_name": "0.0.0.0",
                "server_port": int(os.environ.get("PORT", "7860")),
                "show_api": False,
            }
        )
    else:
        launch_kwargs["show_api"] = True

    if args.share:
        launch_kwargs["share"] = True

    demo = build_demo()
    app = demo.queue() if args.queue else demo
    app.launch(**launch_kwargs)


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()
