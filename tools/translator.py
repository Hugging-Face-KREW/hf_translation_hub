import gradio as gr
import os
from dotenv import load_dotenv
import json
import re

load_dotenv() # Load environment variables from .env file
from translator.project_config import get_available_projects, get_project_config
from translator.content import get_content, preprocess_content, get_full_prompt, llm_translate, fill_scaffold
from translator.retriever import report
import os
from pathlib import Path


def start_translate_handler_mcp(json_input_str):
    file_to_translate = ""
    project = ""
    repo_url = ""
    additional_instruction = ""
    force_retranslate = False
    request_data = {} # Initialize request_data for error context

    try:
        request_data_from_json = json.loads(json_input_str)
        
        # Extract top-level fields for translation control
        additional_instruction = request_data_from_json.get("additional_instruction", "")
        force_retranslate = request_data_from_json.get("force_retranslate", False)
        target_language = request_data_from_json.get("request", {}).get("target_language", "ko")
        source_language = "en" # Assuming source language is always English for now

        # Extract file details from the 'files' array (assuming the first file is the target)
        files_list = request_data_from_json.get("files", [])
        if not files_list:
            raise ValueError("No files found in the JSON input for translation.")
        
        selected_file_data = files_list[0]
        docs_url = selected_file_data.get("repo_url") # This is the full blob URL
        project = selected_file_data.get("metadata", {}).get("project")
        docs_path = selected_file_data.get("metadata", {}).get("docs_path") # Extract docs_path

        # Extract file_to_translate from docs_url
        file_to_translate = ""
        if "/blob/main/" in docs_url:
            file_to_translate = docs_url.split("/blob/main/")[1]
        elif "/blob/" in docs_url: # Handle other branches if necessary
            parts = docs_url.split("/blob/")
            if len(parts) > 1:
                file_to_translate = parts[1].split("/", 1)[1] # Get path after branch name

        # additional_instruction is extracted from the top-level, force_retranslate is also extracted.
        # No need to re-initialize them here.

        # Construct request_data for the output JSON, using extracted values
        request_data = {
            "project": project,
            "target_language": target_language,
            "source_language": source_language,
            "files": [
                {
                    "repo_url": docs_url, # Use docs_url here
                    "file_path": file_to_translate
                }
            ]
        }

    except json.JSONDecodeError as e:
        error_message = f"❌ Invalid JSON input: {str(e)}"
        return gr.Textbox(value=error_message), gr.Markdown(value=""), gr.Json(value={"error": error_message})
    except ValueError as e:
        error_message = f"❌ Invalid JSON structure: {str(e)}"
        return gr.Textbox(value=error_message), gr.Markdown(value=""), gr.Json(value={"error": error_message})
    except Exception as e:
        error_message = f"❌ Error parsing JSON input: {str(e)}"
        return gr.Textbox(value=error_message), gr.Markdown(value=""), gr.Json(value={"error": error_message})

    print(f"Received request: file={file_to_translate}, project={project}, repo_url={repo_url}, instruction={additional_instruction}, force_retranslate={force_retranslate}")
    
    print(f"[DEBUG] Raw JSON input: {json_input_str}")
    print(f"[DEBUG] Extracted file_to_translate: {file_to_translate}")
    
    if not file_to_translate:
        response = "❌ Please provide a file path to translate in the JSON input."
        return gr.Textbox(value=f"Error: {response}"), gr.Markdown(value=""), gr.Json(value={"error": response})

    if not project:
        response = "❌ Please select a project in the JSON input."
        return gr.Textbox(value=f"Error: {response}"), gr.Markdown(value=""), gr.Json(value={"error": response})

    # Define paths for translated files dynamically
    base_output_dir = Path("translation_result") / Path(docs_path) / target_language
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct the path for the translated file
    # Example: docs/source/en/chat_response_parsing.md -> translation_result/docs/source/ko/chat_response_parsing.md
    translated_file_name = Path(file_to_translate).name
    translated_file_path = base_output_dir / translated_file_name
    print(f"[DEBUG] Constructed translated_file_path: {translated_file_path}")
    print(f"[DEBUG] Does translated_file_path exist? {translated_file_path.exists()}")

    translated_doc = ""
    response_message = ""
    final_json_output = {} # Initialize here

    try:
        result_entry = {
            "file_path": str(translated_file_path.relative_to(Path("translation_result"))),
            "translated_content": "",
            "status": "",
            "metadata": {
                "time_elapsed": 0.0, # Placeholder, actual implementation would measure this
                "model_used": ""
            }
        }

        if not force_retranslate and translated_file_path.exists():
            # Reuse existing translation
            with open(translated_file_path, "r", encoding="utf-8") as f:
                translated_doc = f.read()
            response_message = f"✅ Reused existing translation for {file_to_translate} (Project: {project})"
            
            result_entry["translated_content"] = translated_doc
            result_entry["status"] = "reused"
            result_entry["metadata"]["model_used"] = "cached"

            final_json_output = {
                "type": "translation.output.response",
                "request": request_data,
                "results": [result_entry],
                "error": None
            }
        else:
            # 1. Get content - now passing docs_url
            original_content = get_content(project, docs_url=docs_url)
            print(f"[DEBUG] Original content length: {len(original_content)}")
            
            # 2. Preprocess content
            to_translate = preprocess_content(original_content)
            print(f"[DEBUG] Preprocessed content length: {len(to_translate)}")
            
            # 3. Get full prompt
            full_prompt = get_full_prompt(target_language, to_translate, additional_instruction) # Use extracted target_language
            
            # 4. Translate
            cb, translated_content_raw = llm_translate(full_prompt)
            print(f"LLM Callback: {cb}")
            print(f"[DEBUG] Raw translated content length: {len(translated_content_raw)}")
            
            # Determine model used for metadata
            model_used = ""
            if os.environ.get("ANTHROPIC_API_KEY"):
                model_used = "claude-sonnet-4-20250514 (Anthropic API)"
            elif os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
                model_used = "claude-3-7-sonnet-20250219-v1 (AWS Bedrock)"

            # 5. Fill scaffold
            translated_doc = fill_scaffold(original_content, to_translate, translated_content_raw)
            
            # 6. Save the new translation
            with open(translated_file_path, "w", encoding="utf-8") as f:
                f.write(translated_doc)
            
            response_message = f"✅ Successfully translated and saved {file_to_translate} (Project: {project})"
            
            result_entry["translated_content"] = translated_doc
            result_entry["status"] = "success"
            result_entry["metadata"]["model_used"] = model_used

            final_json_output = {
                "type": "translation.output.response",
                "request": request_data,
                "results": [result_entry],
                "error": None
            }
        print(f"[DEBUG] Final translated_doc content:\n{translated_doc}")
        
        # Create a display version of translated_doc for the Markdown component
        # This version will have problematic custom syntax removed for better rendering.
        display_translated_doc = translated_doc
        
        # Remove XML-style comments for display
        display_translated_doc = re.sub(r"<!--.*?-->", "", display_translated_doc, flags=re.DOTALL)
        
        # Remove <hfoptions> and <hfoption> tags and their content for display
        display_translated_doc = re.sub(r"<hfoptions.*?>(.*?)</hfoptions>", "", display_translated_doc, flags=re.DOTALL)
        display_translated_doc = re.sub(r"<hfoption.*?>(.*?)</hfoption>", "", display_translated_doc, flags=re.DOTALL)

        return gr.Textbox(value=f"Translation Complete: {response_message}"), gr.Markdown(value=display_translated_doc), gr.Textbox(value=translated_doc), gr.Json(value=final_json_output)
    except Exception as e:
        error_message = f"Error during translation: {str(e)}"
        # Ensure request_data is defined even in case of early errors for context
        # If request_data was not successfully parsed, create a minimal one for error context
        if not request_data:
            request_data = {
                "project": project if project else "unknown",
                "target_language": "ko",
                "source_language": "en",
                "files": [
                    {
                        "repo_url": repo_url if repo_url else "unknown",
                        "file_path": file_to_translate if file_to_translate else "unknown"
                    }
                ]
            }
        error_json_output = {
            "type": "translation.output.response",
            "request": request_data,
            "results": [],
            "error": error_message
        }
        return gr.Textbox(value=error_message), gr.Markdown(value=""), gr.Json(value=error_json_output)

def update_status_mcp():
    return gr.Textbox(value="Ready")

def update_project_config_display(project):
    """Update the project config display when project selection changes."""
    if not project:
        return ""
    
    # Since project_config is no longer used for repo_url, we'll just display the project name.
    config_html = f"""
### 📋 Project Configuration: {project}

- **Name:** {project}
"""
    return config_html

def generate_json_request(project, docs_url, additional_instruction, force_retranslate):
    # Extract file_path and docs_path from docs_url
    file_to_translate = ""
    docs_path_extracted = ""
    if "/blob/main/" in docs_url:
        parts = docs_url.split("/blob/main/")
        if len(parts) > 1:
            file_to_translate = parts[1]
            # Assuming docs_path is the part before the language directory and file name
            docs_path_parts = file_to_translate.split("/")
            if len(docs_path_parts) > 2: # Ensure there are enough parts for docs/source/en/file.md
                docs_path_extracted = "/".join(docs_path_parts[:-2]) # Exclude language and filename
            elif len(docs_path_parts) > 1: # Fallback if only docs/source/file.md (no language dir)
                docs_path_extracted = "/".join(docs_path_parts[:-1]) # Exclude filename
            else:
                docs_path_extracted = "" # No valid docs_path found
    elif "/blob/" in docs_url: # Handle other branches if necessary
        parts = docs_url.split("/blob/")
        if len(parts) > 1:
            path_after_blob = parts[1]
            branch_and_filepath = path_after_blob.split("/", 1)
            if len(branch_and_filepath) > 1:
                file_to_translate = branch_and_filepath[1]
                docs_path_parts = file_to_translate.split("/")
                if len(docs_path_parts) > 2: # Ensure there are enough parts for docs/source/en/file.md
                    docs_path_extracted = "/".join(docs_path_parts[:-2]) # Exclude language and filename
                elif len(docs_path_parts) > 1: # Fallback if only docs/source/file.md (no language dir)
                    docs_path_extracted = "/".join(docs_path_parts[:-1]) # Exclude filename
                else:
                    docs_path_extracted = "" # No valid docs_path found

    request_data = {
        "files": [
            {
                "path": file_to_translate,
                "repo_url": docs_url, # Use user-provided docs_url
                "metadata": {
                    "project": project,
                    "docs_path": docs_path_extracted, # Include docs_path here
                }
            }
        ],
        "additional_instruction": additional_instruction,
        "force_retranslate": force_retranslate,
        "target_language": "ko", # Hardcoded target language for this server
        "source_language": "en", # Hardcoded source language for this server
    }
    return json.dumps(request_data, indent=2)

def create_mcp_interface():
    with gr.Blocks(css="""
        .markdown-scrollable {
            overflow-y: auto;
        }
    """) as demo:
        gr.Markdown("## Translation Module MCP Server")
        
        status_display = gr.Textbox(label="Status", interactive=False, value="Idle")
        start_translate_btn = gr.Button("Start Translation (MCP)", elem_classes="action-button")

        with gr.TabItem("Translate Inputs", id=0):
            project_dropdown = gr.Radio(
                choices=get_available_projects(),
                label="🎯 Select Project",
                value="transformers",
            )
            project_config_display = gr.Markdown(value=update_project_config_display("transformers"))
            docs_url_input = gr.Textbox(
                label="🔗 Documentation URL (Full blob URL)",
                value="https://github.com/huggingface/transformers/blob/main/docs/source/en/accelerator_selection.md",
                placeholder="e.g., https://github.com/huggingface/transformers/blob/main/docs/source/en/accelerator_selection.md",
            )
            additional_instruction = gr.Textbox(
                label="📝 Additional instructions (Optional)",
                placeholder="Example: Translate 'model' as '모델' consistently",
                lines=2,
            )
            force_retranslate = gr.Checkbox(
                label="🔄 Force Retranslate",
                value=False,
            )
            
            generate_json_btn = gr.Button("Generate JSON Request")
            json_request_textbox = gr.Textbox(
                label="JSON Request (for Translation)",
                value="",
                lines=10,
                interactive=True,
            )

            with gr.Row():
                translated_output = gr.Markdown(
                    label="Translated Content (Markdown)",
                    value="",
                    elem_classes="markdown-scrollable",
                    height=500, # Explicitly set height to enable scrolling
                )
                raw_text_output = gr.Textbox(
                    label="Translated Content (Raw Text)",
                    value="",
                    lines=20, # Give it a reasonable default height
                    interactive=False,
                    elem_classes="markdown-scrollable", # Reuse scrollable class
                )
            json_output = gr.Json(
                label="Raw JSON Output",
                value={},
            )

        # Update project config display when project selection changes
        project_dropdown.change(
            fn=update_project_config_display,
            inputs=[project_dropdown],
            outputs=[project_config_display],
        )
        
        # Connect generate_json_btn to generate_json_request function
        generate_json_btn.click(
            fn=generate_json_request,
            inputs=[
                project_dropdown,
                docs_url_input,
                additional_instruction,
                force_retranslate
            ],
            outputs=[json_request_textbox],
        )
        
        start_translate_btn.click(
            fn=start_translate_handler_mcp,
            inputs=[json_request_textbox],
            outputs=[status_display, translated_output, raw_text_output, json_output],
        )
    return demo

if __name__ == "__main__":
    demo = create_mcp_interface()
    demo.launch(mcp_server=True)
