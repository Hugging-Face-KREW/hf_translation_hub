"""External API adapters for file content retrieval."""

import re
import string
import requests
from project_config import get_project_config
from prompt_glossary import PROMPT_WITH_GLOSSARY


def get_content(filepath: str, project: str = "transformers") -> str:
    """Get file content from GitHub raw URL."""
    if filepath == "":
        raise ValueError("No files selected for translation.")

    config = get_project_config(project)
    # Extract repo path from repo_url (e.g., "huggingface/transformers")
    repo_path = config.repo_url.replace("https://github.com/", "")
    
    url = f"https://raw.githubusercontent.com/{repo_path}/main/{filepath}"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        return content
    else:
        raise ValueError(f"Failed to retrieve content from the URL: {url}")


def preprocess_content(content: str) -> str:
    """Extract text to translate from document."""
    # ignore top license comment
    to_translate = content[content.find("#") :]
    # remove empty lines from text
    to_translate = re.sub(r"\n\n+", "\n\n", to_translate)
    return to_translate


def get_full_prompt(language: str, to_translate: str, additional_instruction: str = "") -> str:
    """Generate optimized translation prompt for the content."""
    base_prompt = string.Template(
        "What do these sentences about Hugging Face Transformers "
        "(a machine learning library) mean in $language? "
        "Please do not translate the word after a 🤗 emoji "
        "as it is a product name. Output the complete markdown file**, with prose translated and all other content intact"
        "No explanations or extras—only the translated markdown. Also translate all comments within code blocks as well."
    ).safe_substitute(language=language)
    
    base_prompt += "\n\n```md"
    
    full_prompt = "\n".join([base_prompt, to_translate.strip(), "```", PROMPT_WITH_GLOSSARY])
    
    if additional_instruction.strip():
        full_prompt += f"\n\n🗒️ Additional instructions: {additional_instruction.strip()}"
    
    return full_prompt


def get_language_name(language_code: str) -> str:
    """Convert language code to full language name."""
    language_map = {
        "ko": "Korean",
        "zh": "Chinese", 
        "ja": "Japanese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese"
    }
    return language_map.get(language_code, language_code.title())