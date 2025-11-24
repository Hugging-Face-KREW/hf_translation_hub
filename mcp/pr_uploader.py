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
    translated_filepath: str,
    base_branch: str = "main",
    preview_mode: bool = False,
):
    # This function will call the GitHubPRAgent's workflow
    # and return the results for display in Gradio.
    
    print(f"Starting PR generation with:")
    print(f"  Reference PR URL: {reference_pr_url}")
    print(f"  Target Language: {target_language}")
    print(f"  Filepath: {filepath}")
    print(f"  Translated Filepath: {translated_filepath}") # Pass the filepath directly
    print(f"  Base Branch: {base_branch}")
    print(f"  Preview Mode: {preview_mode}") # Log preview mode status

    try:
        result = pr_agent.run_translation_pr_workflow(
            reference_pr_url=reference_pr_url,
            target_language=target_language,
            filepath=filepath,
            translated_filepath=translated_filepath, # Pass the filepath directly
            base_branch=base_branch,
            preview_mode=preview_mode, # Pass preview_mode to the agent
        )
        
        if result["status"] == "preview":
            message = "✨ PR Preview Generated Successfully!"
            # Return preview data, and also enable the checkbox and button
            return gr.Textbox(value=message), gr.Json(value=result["data"]), result["data"], gr.update(interactive=True), gr.update(interactive=True)
        elif result["status"] == "success":
            message = f"✅ PR created successfully: {result['pr_url']}"
            # On success, reset checkbox and button
            return gr.Textbox(value=message), gr.Json(value=result), None, gr.update(value=False, interactive=False), gr.update(interactive=False)
        elif result["status"] == "partial_success":
            message = f"⚠️ Partial success: {result['message']}"
            return gr.Textbox(value=message), gr.Json(value=result), None, gr.update(value=False, interactive=False), gr.update(interactive=False)
        else:
            message = f"❌ Error during PR generation: {result['message']}"
            return gr.Textbox(value=message), gr.Json(value=result), None, gr.update(value=False, interactive=False), gr.update(interactive=False)

    except Exception as e:
        error_message = f"❌ Unexpected error during PR generation: {str(e)}"
        return gr.Textbox(value=error_message), gr.Json(value={"error": error_message}), None, gr.update(value=False, interactive=False), gr.update(interactive=False)
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

def handle_pr_confirmation_mcp(preview_data: dict, approved: bool):
    if not approved:
        message = "❌ PR creation cancelled by user."
        return gr.Textbox(value=message), gr.Json(value=preview_data), gr.update(value=False, interactive=False), gr.update(interactive=False)

    if not preview_data:
        message = "❌ No preview data available to create PR."
        return gr.Textbox(value=message), gr.Json(value={}), gr.update(value=False, interactive=False), gr.update(interactive=False)

    try:
        # Extract necessary parameters from preview_data
        reference_pr_url = preview_data["reference_pr_url"]
        target_language = preview_data["target_language"]
        filepath = preview_data["filepath"]
        translated_filepath = preview_data["target_filepath"] # Get the filepath
        base_branch = preview_data["base_branch_for_pr"].split(":")[-1]

        print(f"Executing PR creation for: {filepath} to {target_language}")
        result = pr_agent.run_translation_pr_workflow(
            reference_pr_url=reference_pr_url,
            target_language=target_language,
            filepath=filepath,
            translated_filepath=translated_filepath, # Pass the filepath directly
            base_branch=base_branch,
            preview_mode=False, # Actual creation mode
        )

        if result["status"] == "success":
            message = f"✅ PR created successfully: {result['pr_url']}"
            return gr.Textbox(value=message), gr.Json(value=result), gr.update(value=False, interactive=False), gr.update(interactive=False)
        elif result["status"] == "partial_success":
            message = f"⚠️ Partial success: {result['message']}"
            return gr.Textbox(value=message), gr.Json(value=result), gr.update(value=False, interactive=False), gr.update(interactive=False)
        else:
            message = f"❌ Error during PR creation: {result['message']}"
            return gr.Textbox(value=message), gr.Json(value=result), gr.update(value=False, interactive=False), gr.update(interactive=False)

    except Exception as e:
        error_message = f"❌ Unexpected error during PR creation: {str(e)}"
        return gr.Textbox(value=error_message), gr.Json(value={"error": error_message}), gr.update(value=False, interactive=False), gr.update(interactive=False)

def create_pr_agent_interface():
    with gr.Blocks(css="""
        .markdown-scrollable {
            overflow-y: auto;
        }
    """) as demo:
        gr.Markdown("## PR Agent Module MCP Server")
        
        pr_status_display = gr.Textbox(label="PR Generation Status", interactive=False, value="Idle")
        with gr.Row(): # Use gr.Row to place buttons side-by-side
            preview_pr_btn = gr.Button("Preview PR (JSON)", elem_classes="secondary-button") # New button for preview

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
            translated_filepath_input = gr.Textbox(
                label="📄 Translated Document File Path (e.g., path/to/translated_file.md)",
                value="translation_result/docs/source/ko/accelerator_selection.md",
                lines=1,
                interactive=True,
                placeholder="e.g., translation_result/docs/source/ko/accelerator_selection.md",
            )
            base_branch_input = gr.Textbox(
                label="🌿 Base Branch (e.g., main)",
                value="main",
            )
            
            pr_json_output = gr.Json(
                label="PR Generation Raw JSON Output",
                value={},
            )
            
            # New UI for human approval
            with gr.Row():
                confirmation_checkbox = gr.Checkbox(label="I approve this PR preview and wish to proceed with actual PR creation.", interactive=False)
                confirm_pr_btn = gr.Button("Confirm & Create PR", elem_classes="action-button", interactive=False)

        # Hidden state to store preview data
        pr_preview_state = gr.State(value=None)

        search_pr_btn.click(
            fn=search_reference_pr_mcp,
            inputs=[search_target_language, search_context],
            outputs=[search_output, recommended_pr_url],
        )

        preview_pr_btn.click( # Modified click event for preview button
            fn=start_pr_generation_mcp,
            inputs=[
                reference_pr_url_input,
                target_language_input,
                filepath_input,
                translated_filepath_input, # Changed to use the new filepath input
                base_branch_input,
                gr.State(True), # Pass True for preview_mode
            ],
            outputs=[pr_status_display, pr_json_output, pr_preview_state, confirmation_checkbox, confirm_pr_btn],
        ).success(
            fn=lambda x: [gr.update(interactive=True), gr.update(interactive=True)], # Enable checkbox and button
            inputs=pr_preview_state, # Use output from start_pr_generation_mcp to trigger
            outputs=[confirmation_checkbox, confirm_pr_btn],
            queue=False,
        )

        confirm_pr_btn.click( # New click event for confirm button
            fn=lambda preview_data, approved: handle_pr_confirmation_mcp(preview_data, approved),
            inputs=[pr_preview_state, confirmation_checkbox],
            outputs=[pr_status_display, pr_json_output, confirmation_checkbox, confirm_pr_btn], # Reset checkbox and button state
        )
    return demo

if __name__ == "__main__":
    demo = create_pr_agent_interface()
    demo.launch()
