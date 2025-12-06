"""Module for gradio chat-based translation agent interface."""

import os
import re
from pathlib import Path

import gradio as gr

from agent.workflow import (
    report_translation_target_files,
    translate_docs_interactive,
    generate_github_pr,
)
from pr_generator.searcher import find_reference_pr_simple_stream
from translator.content import get_full_prompt, get_content, preprocess_content
from translator.project_config import get_available_projects, get_project_config


# State management
class ChatState:
    def __init__(self):
        self.step = "welcome"  # welcome -> find_files -> translate -> create_github_pr
        
        # Transient state (reset on restart)
        self.selected_project = "transformers"
        self.target_language = "ko"
        self.k_files = 10
        self.files_to_translate = []
        self.additional_instruction = ""
        self.current_file_content = {"translated": ""}
        self.pr_result = None
        
        # Persistent settings (preserved across restarts)
        self.persistent_settings = {
            "anthropic_api_key": "",
            "aws_bearer_token_bedrock": "",
            "github_config": {
                "token": "",
                "owner": "",
                "repo_name": "",
                "reference_pr_url": "",
            }
        }
    
    def reset_transient_state(self):
        """Reset only the workflow state, keep persistent settings"""
        self.step = "welcome"
        self.selected_project = "transformers"
        self.target_language = "ko"
        self.k_files = 10
        self.files_to_translate = []
        self.additional_instruction = ""
        self.current_file_content = {"translated": ""}
        self.pr_result = None
    
    @property
    def github_config(self):
        return self.persistent_settings["github_config"]


state = ChatState()


def _extract_content_for_display(content: str) -> str:
    """Extract text from document for display."""
    # Remove Copyright header
    to_translate = re.sub(r"<!--.*?-->", "", content, count=1, flags=re.DOTALL)
    to_translate = to_translate.strip()
    ## remove code blocks from text
    to_translate = re.sub(r"```.*?```", "", to_translate, flags=re.DOTALL)
    ## remove markdown tables from text
    to_translate = re.sub(r"^\|.*\|$\n?", "", to_translate, flags=re.MULTILINE)
    ## remove empty lines from text
    to_translate = re.sub(r"\n\n+", "\n\n", to_translate)

    return to_translate


def get_welcome_message():
    """Initial welcome message with project selection"""
    return """**👋 Welcome to 🌐 Hugging Face i18n Translation Agent!**

I'll help you find files that need translation and translate them in a streamlined workflow.

**🎯 First, select which project you want to translate:**

Use the **`Quick Controls`** on the right to select a project, or **ask me `what`, `how`, or `help`** to get started.
"""


def process_file_search_handler(project: str, lang: str, k: int, history: list) -> tuple:
    """Process file search request and update Gradio UI components."""
    global state
    state.selected_project = project
    state.target_language = lang
    state.k_files = k
    state.step = "find_files"

    try:
        status_report, files_list = report_translation_target_files(project, lang, k)
    except Exception as e:
        if "rate limit" in str(e).lower():
            response = f"""❌ **GitHub API Rate Limit Exceeded**

{str(e)}

**💡 To fix this:**
1. Set GitHub Token in Configuration panel above
2. Click "💾 Save Configuration" 
3. Try "Find Files" again"""
            history.append(["File search request", response])
            return history, "", update_status(), gr.Tabs(selected=0), gr.update(choices=[]), gr.update(visible=False)
        else:
            raise  # Re-raise non-rate-limit errors
    state.files_to_translate = (
        [file[0] for file in files_list]
        if files_list
        else []
    )

    response = f"""**✅ File search completed!**

**Status Report:**
{status_report}

**📁 Found first {len(state.files_to_translate)} files to translate:**
"""

    if state.files_to_translate:
        config = get_project_config(state.selected_project)
        for i, file in enumerate(state.files_to_translate, 1):
            file_link = f"{config.repo_url}/blob/main/{file}"
            response += f"\n{i}. [`{file}`]({file_link})"

        # if len(state.files_to_translate) > 5:
        #     response += f"\n... and {len(state.files_to_translate) - 5} more files"

        response += "\n\n**🚀 Ready to start translation?**\nI can begin translating these files one by one. Would you like to proceed?"
    else:
        response += "\nNo files found that need translation."

    # Add to history
    history.append(["Please find files that need translation", response])
    cleared_input = ""

    # 드롭다운 choices로 쓸 파일 리스트 반환 추가
    return (
        history,
        cleared_input,
        update_status(),
        gr.Tabs(),  # Don't change tab
        update_dropdown_choices(state.files_to_translate),
    )


def update_dropdown_choices(file_list):
    return gr.update(choices=file_list, value=None)


def confirm_and_go_translate_handler(history):
    """Confirm selection and go to translate tab"""
    global state
    
    response = f"✅ **Selection confirmed!**\n\n🎯 **Project:** {state.selected_project}\n🌍 **Language:** {state.target_language}\n\n**➡️ Go to Tab 2 to start translation.**"
    history.append(["Confirm selection", response])
    return history, "", update_status(), gr.Tabs(selected=1)


def confirm_translation_and_go_upload_handler(history):
    """Confirm translation and go to upload PR tab"""
    global state
    
    if not state.current_file_content.get("translated"):
        response = "❌ No translation available. Please complete translation first."
        history.append(["Upload PR request", response])
        return history, "", update_status(), gr.Tabs()
    
    response = f"✅ **Translation confirmed!**\n\n📄 **File:** `{state.files_to_translate[0] if state.files_to_translate else 'Unknown'}`\n\n**➡️ Go to Tab 3 to upload PR.**"
    history.append(["Upload PR request", response])
    return history, "", update_status(), gr.Tabs(selected=2)


def start_translation_process(force_retranslate=False):
    """Start the translation process for the first file"""
    if not state.files_to_translate:
        return "❌ No files available for translation.", ""

    current_file = state.files_to_translate[0]

    # Call translation function (simplified for demo)
    try:
        status, translated = translate_docs_interactive(
            state.target_language, [[current_file]], state.additional_instruction, state.selected_project, force_retranslate
        )

        state.current_file_content = {"translated": translated}
        path = (
            Path(__file__).resolve().parent.parent
            / f"translation_result/{current_file}"
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(translated, encoding="utf-8")

        config = get_project_config(state.selected_project)
        original_file_link = f"{config.repo_url}/blob/main/{current_file}"
        print("Compeleted translation:\n")
        print(translated)
        print("----------------------------")
        
        # Different response format for existing vs new translation
        if isinstance(status, str) and "Existing translation loaded" in status:
            response = f"{status}\n**📄 Original Content Link:** {original_file_link}\n\n**🌐 Translated Content:**"
        else:
            response = (
                f"""🔄 Translation for: `{current_file}`\n"""
                f"**📄 Original Content Link:** {original_file_link}\n\n"
                f"{status}\n\n"
                "**🌐 Translated Content:**"
            )
        return response, translated


    except Exception as e:
        response = f"❌ Translation failed: {str(e)}"
        response += "\n**➡️ Please try from the beginning.**"
        return response, ""


def handle_general_message(message):
    """Handle general messages"""
    message_lower = message.lower()

    if any(word in message_lower for word in ["help", "what", "how"]):
        return """**🤖 I'm your Hugging Face i18n Translation Agent!**

I can help you:
1. **🔍 Find files** that need translation
2. **🌐 Translate documents** using AI
3. **📋 Review translations** for quality
4. **🚀 Create GitHub PR** for translation

Currently available actions with quick controls:
- "find files" - Search for files needing translation
- "translate" - Start translation process  
- "review" - Review current translation
- "github" - Create GitHub Pull Request
- "restart" - Start over"""

    elif "restart" in message_lower:
        global state
        state = ChatState()
        return get_welcome_message()

    else:
        return """I understand you want to work on translations! 

**Two ways to get started:**

1. **🔍 Find Files first** - Use Tab 1 to discover files that need translation
2. **🚀 Direct Translation** - Go to Tab 2 and enter a file path directly (e.g., `docs/source/en/model_doc/bert.md`)

Make sure to configure your API keys in the Configuration panel above.
"""


# Main handler
def handle_user_message(message, history):
    """Handle user messages and provide appropriate responses"""
    global state

    if not message.strip():
        return history, ""

    elif state.step == "find_files" and any(
        word in message.lower()
        for word in ["yes", "proceed", "start", "translate", "translation"]
    ):
        # User wants to start translation
        if state.files_to_translate:
            state.step = "translate"
            response, translated = start_translation_process()
            history.append([message, response])
            history.append(["", translated])
            return history, ""
        else:
            response = (
                "❌ No files available for translation. Please search for files first."
            )
    # Handle GitHub PR creation - This part is removed as approve_handler is the main entry point
    else:
        # General response
        response = handle_general_message(message)

    history.append([message, response])
    return history, ""


def update_status():
    if state.step == "welcome":
        return f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;padding: 10px; background: rgba(0, 0, 0, 0.25); border-radius: 8px;">
            <div><strong>🔄 Step:</strong> Welcome</div>
            <div><strong>🎯 Project:</strong> {state.selected_project}</div>
            <div><strong>📁 Files:</strong> 0</div>
            <div><strong>🌍 Language:</strong> {state.target_language}</div>
        </div>
        """

    step_map = {
        "welcome": "Welcome",
        "find_files": "Finding Files",
        "translate": "Translating",
        "review": "Reviewing",
        "create_github_pr": "Creating PR",
    }

    progress_map = {
        "welcome": "Ready to start",
        "find_files": "Files found",
        "translate": f"{len(state.files_to_translate)} remaining",
        "review": "Review complete",
        "create_github_pr": "PR generation in progress",
    }

    # Check GitHub configuration status
    github_status = "❌ Not configured"
    if all(
        [
            state.github_config["token"],
            state.github_config["owner"],
            state.github_config["repo_name"],
        ]
    ):
        github_status = (
            f"✅ {state.github_config['owner']}/{state.github_config['repo_name']}"
        )

    status_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; background: rgba(0, 0, 0, 0.25); border-radius: 8px;">
        <div><strong>🔄 Step:</strong> {step_map.get(state.step, state.step)}</div>
        <div><strong>🎯 Project:</strong> {state.selected_project}</div>
        <div><strong>📁 Files:</strong> {len(state.files_to_translate)}</div>
        <div><strong>🌍 Language:</strong> {state.target_language}</div>
        <div><strong>⏳ Progress:</strong> {progress_map.get(state.step, 'In progress')}</div>
        <div><strong>🔧 GitHub:</strong> {github_status}</div>
    </div>
    """

    return status_html


# Event handlers


def sync_language_displays(lang):
    return lang


def update_project_selection(project, history):
    """Update state when project is selected"""
    global state
    state.selected_project = project
    response = f"Selection confirmed: 🎯 Project → **{project}**"
    history.append(["Project selection", response])
    return history, "", update_status()


def update_language_selection(lang, history):
    """Update state when language is selected"""
    global state
    state.target_language = lang
    response = f"Selection confirmed: 🌍 Language → **{lang}**"
    history.append(["Language selection", response])
    return history, "", update_status(), lang


def update_persistent_config(api_provider, anthropic_key, aws_bearer_token_bedrock, github_token, github_owner, github_repo, reference_pr_url, history):
    """Update persistent configuration settings."""
    global state
    
    # Update API keys based on provider selection
    if api_provider == "Anthropic":
        state.persistent_settings["anthropic_api_key"] = anthropic_key
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        # Clear AWS Bedrock token if Anthropic is selected
        state.persistent_settings["aws_bearer_token_bedrock"] = ""
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    elif api_provider == "AWS Bedrock":
        state.persistent_settings["aws_bearer_token_bedrock"] = aws_bearer_token_bedrock
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_bearer_token_bedrock
        # Clear Anthropic key if AWS Bedrock is selected
        state.persistent_settings["anthropic_api_key"] = ""
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        # If no provider is selected or unknown, clear both
        state.persistent_settings["anthropic_api_key"] = ""
        os.environ.pop("ANTHROPIC_API_KEY", None)
        state.persistent_settings["aws_bearer_token_bedrock"] = ""
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    
    if github_token:
        os.environ["GITHUB_TOKEN"] = github_token

    # Get default reference PR URL from project config if not provided
    if not reference_pr_url and state.selected_project:
        try:
            config = get_project_config(state.selected_project)
            reference_pr_url = config.reference_pr_url
        except:
            pass

    # Save GitHub configuration to persistent settings
    state.persistent_settings["github_config"].update({
        "token": github_token or "",
        "owner": github_owner or "",
        "repo_name": github_repo or "",
        "reference_pr_url": reference_pr_url or "",
    })

    # Build response message based on what was configured
    response = "✅ Configuration saved!"
    if github_owner and github_repo:
        response += f" GitHub: {github_owner}/{github_repo}"
    
    if api_provider == "Anthropic" and anthropic_key:
        response += " Anthropic API key updated."
    elif api_provider == "AWS Bedrock" and aws_bearer_token_bedrock:
        response += " AWS Bedrock Bearer Token updated."
    
    history.append(["Configuration update", response])
    return history, "", update_status()


def update_github_config(token, owner, repo, reference_pr_url):
    """Legacy function for backward compatibility."""
    return update_persistent_config("", token, owner, repo, reference_pr_url)


def update_prompt_preview(language, file_path, additional_instruction):
    """Update prompt preview based on current settings"""
    if not file_path.strip():
        return "Select a file to see the prompt preview..."
    
    try:
        # Get language name
        if language == "ko":
            translation_lang = "Korean"
        else:
            translation_lang = language
        
        # Get sample content (first 500 characters)
        content = get_content(file_path, state.selected_project)
        to_translate = preprocess_content(content)
        
        # Truncate for preview
        sample_content = to_translate[:500] + ("..." if len(to_translate) > 500 else "")
        
        # Generate prompt
        prompt = get_full_prompt(translation_lang, sample_content, additional_instruction)
        
        return prompt
    except Exception as e:
        error_str = str(e)
        if "Failed to retrieve content from the URL" in error_str:
            return f"❌ **File not found:** `{file_path}`\n\n💡 **Please check:**\n1. Is this file in the **{state.selected_project}** project?\n2. Use \"🔍 Find Files to Translate\" to see available files\n3. Verify the file path is correct"
        return f"Error generating prompt preview: {error_str}"


def send_message(message, history):
    new_history, cleared_input = handle_user_message(message, history)
    return new_history, cleared_input, update_status()


# Button handlers with tab switching
def start_translate_handler(history, file_to_translate, additional_instruction="", force_retranslate=False):
    # Use persistent anthropic key
    anthropic_key = state.persistent_settings["anthropic_api_key"]
    aws_bearer_token_bedrock = state.persistent_settings["aws_bearer_token_bedrock"]

    if not anthropic_key and not aws_bearer_token_bedrock:
        response = "❌ Please set either Anthropic API key or AWS Bearer Token for Bedrock in Configuration panel first."
        history.append(["Translation request", response])
        return history, "", update_status(), gr.Tabs(), gr.update(), gr.update()
    
    # Set the active API key to environment variable for translator.content.py
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None) # Ensure only one is active
    elif aws_bearer_token_bedrock:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_bearer_token_bedrock
        os.environ.pop("ANTHROPIC_API_KEY", None) # Ensure only one is active
    
    # Check if file path is provided
    if not file_to_translate or not file_to_translate.strip():
        response = "❌ Please select a file from the dropdown or enter a file path to translate."
        history.append(["Translation request", response])
        return history, "", update_status(), gr.Tabs(), gr.update(), gr.update()
    
    state.additional_instruction = additional_instruction
    state.files_to_translate = [file_to_translate]
    state.step = "translate"
    
    # Start translation directly
    if force_retranslate:
        history.append(["Translation request", "🔄 **Force retranslation started...**"])
    response, translated = start_translation_process(force_retranslate)
    history.append(["", response])
    if translated:
        history.append(["", translated])
    
    # Update button text and show confirm button after translation
    start_btn_text = "🔄 Retranslation" if state.current_file_content["translated"] else "🚀 Start Translation"
    confirm_btn_visible = bool(state.current_file_content["translated"])
    
    return history, "", update_status(), gr.Tabs(), gr.update(value=start_btn_text), gr.update(visible=confirm_btn_visible)


def approve_handler(history, owner, repo, reference_pr_url):
    """Handles the request to generate a GitHub PR."""
    global state
    state.step = "create_github_pr"

    # Check all required GitHub configuration at once
    github_config = state.persistent_settings["github_config"]
    missing_config = []
    
    if not github_config.get("token"):
        missing_config.append("GitHub Token")
    if not owner:
        missing_config.append("GitHub Owner")
    if not repo:
        missing_config.append("Repository Name")
    
    if missing_config:
        config = get_project_config(state.selected_project)
        repo_name = config.repo_url.split('/')[-1]  # Extract repo name from URL
        response = f"❌ Please set the following in Configuration panel first: {', '.join(missing_config)}\n\n💡 **Note:** GitHub Owner/Repository should be your fork of [`{repo_name}`]({config.repo_url}) (e.g., Owner: `your-username`, Repository: `{repo_name}`)"
        history.append(["GitHub PR creation request", response])
        return history, "", update_status()

    # Update reference PR URL (can be set per PR)
    if reference_pr_url:
        state.persistent_settings["github_config"]["reference_pr_url"] = reference_pr_url

    # Use persistent settings
    github_config = state.persistent_settings["github_config"]

    # Initialize response variable
    response = ""
    
    # If reference PR is not provided, use the agent to find one
    if not github_config.get("reference_pr_url"):
        response = "🤖 **Reference PR URL not found. The agent will now search for a suitable one...**"
        try:
            # This part is simplified to avoid streaming logic in a non-generator function
            stream_gen = find_reference_pr_simple_stream(
                target_language=state.target_language,
                context="documentation translation",
            )
            # We will just get the final result from the generator
            final_result = None
            try:
                while True:
                    # We are not interested in the streamed messages here, just the final result.
                    next(stream_gen)
            except StopIteration as e:
                final_result = e.value

            if final_result and final_result.get("status") == "success":
                result_text = final_result.get("result", "")
                match = re.search(r"https://github.com/[^\s]+", result_text)
                if match:
                    found_url = match.group(0)
                    state.github_config["reference_pr_url"] = found_url
                    response += f"\n✅ **Agent found a reference PR:** {found_url}"
                else:
                    raise ValueError(
                        "Could not extract a valid PR URL from agent's response."
                    )
            else:
                error_message = final_result.get("message") or final_result.get(
                    "result", "Unknown error"
                )
                raise ValueError(f"Agent failed to find a PR. Reason: {error_message}")
        except Exception as e:
            response += f"\n❌ **Agent failed to find a reference PR.**\nReason: {e}\n\nPlease provide a reference PR URL manually in Tab 3 and try again."
            history.append(["Agent searching for PR", response])
            return history, "", update_status()

    # Proceed with PR generation
    if state.files_to_translate and state.current_file_content.get("translated"):
        current_file = state.files_to_translate[0]
        translated_content = state.current_file_content["translated"]
        response += "\n\n🚀 **Generating GitHub PR...**"

        # Extract title from file for toctree mapping
        file_name = current_file.split("/")[-1].replace(".md", "").replace("_", " ").title()
        print(file_name)
        
        pr_response = generate_github_pr(
            target_language=state.target_language,
            filepath=current_file,
            translated_content=translated_content,
            github_config=state.github_config,
            en_title=file_name,
            project=state.selected_project,
        )
        response += f"\n{pr_response}"
    else:
        response = "❌ No translated file available. Please complete the translation process first."

    history.append(["GitHub PR creation request", response])
    return history, "", update_status()


def restart_handler(history):
    """Resets the workflow state but preserves persistent settings."""
    global state
    # Backup persistent settings
    backup_settings = state.persistent_settings.copy()
    
    # Reset state
    state = ChatState()
    
    # Restore persistent settings
    state.persistent_settings = backup_settings
    
    # Restore environment variables
    if backup_settings["anthropic_api_key"]:
        os.environ["ANTHROPIC_API_KEY"] = backup_settings["anthropic_api_key"]
    if backup_settings["aws_bearer_token_bedrock"]:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = backup_settings["aws_bearer_token_bedrock"]
    if backup_settings["github_config"]["token"]:
        os.environ["GITHUB_TOKEN"] = backup_settings["github_config"]["token"]
    
    welcome_msg = get_welcome_message()
    new_hist = [[None, welcome_msg]]
    return new_hist, "", update_status(), gr.Tabs(selected=0)
