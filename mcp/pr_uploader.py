import gradio as gr
import os
import json
import re
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

from pr_generator.agent import GitHubPRAgent
from pr_generator.searcher import find_reference_pr_simple_stream

# Initialize GitHubPRAgent
# These should be set as environment variables
USER_OWNER = os.environ.get("GH_USER_OWNER", "your_github_username")
USER_REPO = os.environ.get("GH_USER_REPO", "your_forked_repo_name")
BASE_OWNER = os.environ.get("GH_BASE_OWNER", "huggingface")
BASE_REPO = os.environ.get("GH_BASE_REPO", "transformers")

pr_agent = GitHubPRAgent(
    user_owner=USER_OWNER,
    user_repo=USER_REPO,
    base_owner=BASE_OWNER,
    base_repo=BASE_REPO
)

def start_pr_generation_mcp(
    reference_pr_url: str,
    target_language: str,
    filepath: str,
    translated_doc_content: str,
    base_branch: str = "main",
):
    # This function will call the GitHubPRAgent's workflow
    # and return the results for display in Gradio.
    # The actual implementation will involve calling pr_agent.run_translation_pr_workflow
    # and handling its output.
    
    # Placeholder for actual PR generation logic
    print(f"Starting PR generation with:")
    print(f"  Reference PR URL: {reference_pr_url}")
    print(f"  Target Language: {target_language}")
    print(f"  Filepath: {filepath}")
    print(f"  Translated Content Length: {len(translated_doc_content)} bytes")
    print(f"  Base Branch: {base_branch}")

    try:
        result = pr_agent.run_translation_pr_workflow(
            reference_pr_url=reference_pr_url,
            target_language=target_language,
            filepath=filepath,
            translated_doc=translated_doc_content,
            base_branch=base_branch,
        )
        
        if result["status"] == "success":
            message = f"✅ PR created successfully: {result['pr_url']}"
            return gr.Textbox(value=message), gr.Json(value=result)
        elif result["status"] == "partial_success":
            message = f"⚠️ Partial success: {result['message']}"
            return gr.Textbox(value=message), gr.Json(value=result)
        else:
            message = f"❌ Error during PR generation: {result['message']}"
            return gr.Textbox(value=message), gr.Json(value=result)

    except Exception as e:
        error_message = f"❌ Unexpected error during PR generation: {str(e)}"
        return gr.Textbox(value=error_message), gr.Json(value={"error": error_message})

def search_reference_pr_mcp(target_language: str, context: str):
    # This function will call the searcher agent and return the best PR URL.
    # It will also stream the progress messages.
    
    search_generator = find_reference_pr_simple_stream(target_language=target_language, context=context)
    
    # Collect all messages and the final result
    messages = []
    final_result = None
    try:
        while True:
            message = next(search_generator)
            messages.append(message)
            print(message) # Print to console for real-time feedback
    except StopIteration as e:
        final_result = e.value

    if final_result and final_result.get("status") == "success":
        pr_url = final_result.get("result", "").replace("Recommended PR URL: ", "")
        return gr.Textbox(value="\n".join(messages)), gr.Textbox(value=pr_url)
    else:
        error_message = final_result.get("result", "Unknown error during PR search.") if final_result else "No result from PR search."
        return gr.Textbox(value="\n".join(messages) + f"\n❌ {error_message}"), gr.Textbox(value="")


def create_pr_agent_interface():
    with gr.Blocks(css="""
        .markdown-scrollable {
            overflow-y: auto;
        }
    """) as demo:
        gr.Markdown("## PR Agent Module MCP Server")
        
        pr_status_display = gr.Textbox(label="PR Generation Status", interactive=False, value="Idle")
        start_pr_btn = gr.Button("Start PR Generation (MCP)", elem_classes="action-button")

        with gr.TabItem("PR Generation Inputs", id=0):
            gr.Markdown("### 🔍 Reference PR Search")
            with gr.Row():
                search_target_language = gr.Textbox(label="Target Language (for search)", value="korean")
                search_context = gr.Textbox(label="Context (for search)", value="docs")
                search_pr_btn = gr.Button("Search Reference PR")
            search_output = gr.Textbox(label="Search Progress", interactive=False, lines=5)
            recommended_pr_url = gr.Textbox(label="Recommended Reference PR URL", interactive=True)

            gr.Markdown("### 📝 PR Generation Details")
            reference_pr_url_input = gr.Textbox(
                label="🔗 Reference PR URL",
                value="https://github.com/huggingface/transformers/pull/24968",
                placeholder="e.g., https://github.com/huggingface/transformers/pull/24968",
            )
            target_language_input = gr.Textbox(
                label="🌐 Target Language",
                value="ko",
                placeholder="e.g., ko, ja, fr",
            )
            filepath_input = gr.Textbox(
                label="📁 Original File Path (e.g., docs/source/en/accelerator_selection.md)",
                value="docs/source/en/accelerator_selection.md",
                placeholder="e.g., docs/source/en/accelerator_selection.md",
            )
            translated_doc_content_input = gr.Textbox(
                label="📄 Translated Document Content",
                value="# Translated Accelerator Selection\n\nThis is the translated content.",
                lines=10,
                interactive=True,
            )
            base_branch_input = gr.Textbox(
                label="🌿 Base Branch (e.g., main)",
                value="main",
            )
            
            pr_json_output = gr.Json(
                label="PR Generation Raw JSON Output",
                value={},
            )

        search_pr_btn.click(
            fn=search_reference_pr_mcp,
            inputs=[search_target_language, search_context],
            outputs=[search_output, recommended_pr_url],
        )

        start_pr_btn.click(
            fn=start_pr_generation_mcp,
            inputs=[
                reference_pr_url_input,
                target_language_input,
                filepath_input,
                translated_doc_content_input,
                base_branch_input,
            ],
            outputs=[pr_status_display, pr_json_output],
        )
    return demo

if __name__ == "__main__":
    demo = create_pr_agent_interface()
    demo.launch()
