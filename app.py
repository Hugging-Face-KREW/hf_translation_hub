"""Module for gradio chat-based translation agent interface."""

import base64
import os

import gradio as gr
from dotenv import load_dotenv

from agent.handler import (
    approve_handler,
    confirm_and_go_translate_handler,
    confirm_translation_and_go_upload_handler,
    get_welcome_message,
    process_file_search_handler,
    restart_handler,
    send_message,
    start_translate_handler,
    sync_language_displays,
    update_language_selection,
    update_project_selection,
    update_prompt_preview,
    update_status,
    update_github_config,
    update_persistent_config,
)
from translator.model import Languages
from translator.project_config import get_available_projects

load_dotenv()


css = """
.gradio-container {
    background: linear-gradient(135deg, #ffeda7 0%, #ffbebf 100%);
}
.chat-container {
    background: rgba(255, 255, 180, 0.25);
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    padding: 1.0em;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,180,0.25);
    width: 100%;
    height: 100%;
}
.control-panel {
    background: rgba(255, 255, 180, 0.25);
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    padding: 1.0em;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,180,0.25);
    width: 100%;
    overflow: visible !important;

}
.status-card {
    width: 100%
}
.action-button {
    background: linear-gradient(135deg, #ff8c8c 0%, #f9a889 100%) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    transition: all 0.3s ease-in-out !important;
}
.action-button:hover {
    background: linear-gradient(135deg, #f9a889 0%, #ff8c8c 100%) !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.2) !important;
    transform: translateY(-2px) !important;
}

.simple-tabs .tab-nav button {
    background: transparent !important;
    color: #4A5568 !important;
    box-shadow: none !important;
    transform: none !important;
    border: none !important;
    border-bottom: 2px solid #E2E8F0 !important;
    border-radius: 0 !important;
    font-weight: 600 !important;
}

.simple-tabs .tab-nav button.selected {
    color: #f97316 !important;
    border-bottom: 2px solid #f97316 !important;
}

.simple-tabs .tab-nav button:hover {
    background: #f3f4f6 !important;
    color: #f97316 !important;
    box-shadow: none !important;
    transform: none !important;
}
"""


# Create the main interface
with gr.Blocks(
    css=css, title=" 🌐 Hugging Face Transformers Docs i18n made easy"
) as demo:
    # Title
    with open("images/hfkr_logo.png", "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode()
    gr.Markdown(
        f'<img src="data:image/png;base64,{base64_img}" style="display: block; margin-left: auto; margin-right: auto; height: 15em;"/>'
    )
    gr.Markdown(
        '<h1 style="text-align: center;"> 🌐 Hugging Face Transformers Docs i18n made easy</h1>'
    )

    # Content
    with gr.Row():
        # Chat interface
        with gr.Column(scale=3, elem_classes=["chat-container"]):
            gr.Markdown("### 🌐 Hugging Face i18n Agent")

            chatbot = gr.Chatbot(
                value=[[None, get_welcome_message()]], scale=1, height=525,
                show_copy_button=True
            )
            
            # Chat input directly under main chat
            gr.Markdown("### 💬 Chat with agent")
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message here... (e.g. 'what', 'how', or 'help')",
                    container=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1, elem_classes="action-button")

        # Controller interface
        with gr.Column(scale=2):
            # Configuration Panel
            with gr.Column(elem_classes=["control-panel"]):
                gr.Markdown("### ⚙️ Configuration")
                
                with gr.Accordion("🔧 API & GitHub Settings", open=True):
                    api_provider_radio = gr.Radio(
                        ["Anthropic", "AWS Bedrock"],
                        label="Select API Provider",
                        value="Anthropic", # Default selection
                        interactive=True,
                    )
                    config_anthropic_key = gr.Textbox(
                        label="🔑 Anthropic API Key",
                        type="password",
                        placeholder="sk-ant-...",
                        visible=True, # Initially visible as Anthropic is default
                    )
                    config_aws_bearer_token_bedrock = gr.Textbox(
                        label="🔑 AWS Bearer Token for Bedrock",
                        type="password",
                        placeholder="AWS_BEARER_TOKEN_BEDROCK",
                        visible=False, # Initially hidden
                    )
                    config_github_token = gr.Textbox(
                        label="🔑 GitHub Token (Required for PR, Optional for file search)",
                        type="password", 
                        placeholder="ghp_...",
                    )
                    
                    with gr.Row():
                        config_github_owner = gr.Textbox(
                            label="👤 GitHub Owner",
                            placeholder="your-username",
                            scale=1,
                        )
                        config_github_repo = gr.Textbox(
                            label="📁 Repository Name", 
                            placeholder="your-repository",
                            scale=1,
                        )
                    
                    save_config_btn = gr.Button(
                        "💾 Save Configuration", elem_classes="action-button"
                    )
                    
            # Quick Controller
            with gr.Column(elem_classes=["control-panel"]):
                gr.Markdown("### 🛠️ Quick Controls")
                status_display = gr.HTML(update_status())

                with gr.Tabs(elem_classes="simple-tabs") as control_tabs:
                    with gr.TabItem("1. Find Files", id=0):
                        with gr.Group():
                            project_dropdown = gr.Radio(
                                choices=get_available_projects(),
                                label="🎯 Select Project",
                                value="transformers",
                            )
                            lang_dropdown = gr.Radio(
                                choices=[language.value for language in Languages],
                                label="🌍 Translate To",
                                value="ko",
                            )
                            k_input = gr.Number(
                                label="📊 First k missing translated docs",
                                value=10,
                                minimum=1,
                            )
                            find_btn = gr.Button(
                                "🔍 Find Files to Translate",
                                elem_classes="action-button",
                            )
                            
                            confirm_go_btn = gr.Button(
                                "✅ Confirm Selection & Go to Translate",
                                elem_classes="action-button",
                            )

                    with gr.TabItem("2. Translate", id=1):
                        with gr.Group():
                            files_to_translate = gr.Radio(
                                choices=[],
                                label="📄 Select a file to translate",
                                interactive=True,
                                value=None,
                            )
                            file_to_translate_input = gr.Textbox(
                                label="🌍 Select in the dropdown or write the file path to translate",
                                value="",
                            )

                            translate_lang_display = gr.Dropdown(
                                choices=[language.value for language in Languages],
                                label="🌍 Translation Language",
                                value="ko",
                                interactive=False,
                            )
                            additional_instruction = gr.Textbox(
                                label="📝 Additional instructions (Optional - e.g., custom glossary)",
                                placeholder="Example: Translate 'model' as '모델' consistently",
                                lines=2,
                            )
                            
                            force_retranslate = gr.Checkbox(
                                label="🔄 Force Retranslate (ignore existing translations)",
                                value=False,
                            )
                            
                            with gr.Accordion("🔍 Preview Translation Prompt", open=False):
                                prompt_preview = gr.Textbox(
                                    lines=8,
                                    interactive=False,
                                    placeholder="Select a file and language to see the prompt preview...",
                                    show_copy_button=True,
                                )
                            
                            start_translate_btn = gr.Button(
                                "🚀 Start Translation", elem_classes="action-button"
                            )
                            
                            confirm_upload_btn = gr.Button(
                                "✅ Confirm Translation & Upload PR",
                                elem_classes="action-button",
                                visible=False,
                            )

                    with gr.TabItem("3. Upload PR", id=2):
                        with gr.Group():
                            reference_pr_url = gr.Textbox(
                                label="🔗 Reference PR URL (Optional)",
                                placeholder="Auto-filled based on project selection",
                            )
                            approve_btn = gr.Button(
                                "✅ Generate GitHub PR", elem_classes="action-button"
                            )
                            restart_btn = gr.Button(
                                "🔄 Restart Translation", elem_classes="action-button"
                            )

    # Event Handlers

    find_btn.click(
        fn=process_file_search_handler,
        inputs=[project_dropdown, lang_dropdown, k_input, chatbot],
        outputs=[chatbot, msg_input, status_display, control_tabs, files_to_translate],
    )
    
    confirm_go_btn.click(
        fn=confirm_and_go_translate_handler,
        inputs=[chatbot],
        outputs=[chatbot, msg_input, status_display, control_tabs],
    )

    # Auto-save selections to state and update prompt preview
    project_dropdown.change(
        fn=update_project_selection,
        inputs=[project_dropdown, chatbot],
        outputs=[chatbot, msg_input, status_display],
    )
    
    # Update prompt preview when project changes
    project_dropdown.change(
        fn=update_prompt_preview,
        inputs=[translate_lang_display, file_to_translate_input, additional_instruction],
        outputs=[prompt_preview],
    )
    
    lang_dropdown.change(
        fn=update_language_selection,
        inputs=[lang_dropdown, chatbot],
        outputs=[chatbot, msg_input, status_display, translate_lang_display],
    )

    #
    files_to_translate.change(
        fn=lambda x: x,
        inputs=[files_to_translate],
        outputs=[file_to_translate_input],
    )

    # Button event handlers
    start_translate_btn.click(
        fn=start_translate_handler,
        inputs=[chatbot, file_to_translate_input, additional_instruction, force_retranslate],
        outputs=[chatbot, msg_input, status_display, control_tabs, start_translate_btn, confirm_upload_btn],
    )
    
    confirm_upload_btn.click(
        fn=confirm_translation_and_go_upload_handler,
        inputs=[chatbot],
        outputs=[chatbot, msg_input, status_display, control_tabs],
    )

    # Configuration Save
    save_config_btn.click(
        fn=update_persistent_config,
        inputs=[api_provider_radio, config_anthropic_key, config_aws_bearer_token_bedrock, config_github_token, config_github_owner, config_github_repo, reference_pr_url, chatbot],
        outputs=[chatbot, msg_input, status_display],
    )

    # API Provider selection handler
    api_provider_radio.change(
        fn=lambda provider: (
            gr.update(visible=True) if provider == "Anthropic" else gr.update(visible=False),
            gr.update(visible=True) if provider == "AWS Bedrock" else gr.update(visible=False),
        ),
        inputs=[api_provider_radio],
        outputs=[config_anthropic_key, config_aws_bearer_token_bedrock],
    )

    approve_btn.click(
        fn=approve_handler,
        inputs=[chatbot, config_github_owner, config_github_repo, reference_pr_url],
        outputs=[chatbot, msg_input, status_display],
    )

    restart_btn.click(
        fn=restart_handler,
        inputs=[chatbot],
        outputs=[chatbot, msg_input, status_display, control_tabs],
    )

    send_btn.click(
        fn=send_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, status_display],
    )

    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, status_display],
    )

    # Update prompt preview when inputs change
    for input_component in [translate_lang_display, file_to_translate_input, additional_instruction]:
        input_component.change(
            fn=update_prompt_preview,
            inputs=[translate_lang_display, file_to_translate_input, additional_instruction],
            outputs=[prompt_preview],
        )

root_path = os.environ.get("GRADIO_ROOT_PATH")
demo.launch(root_path=root_path)
