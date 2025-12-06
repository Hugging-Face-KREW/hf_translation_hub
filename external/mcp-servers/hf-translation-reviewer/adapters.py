from __future__ import annotations

import base64
from typing import Dict, List, Optional

from urllib.parse import urlparse

import requests

from setting import SETTINGS

# Optional provider SDKs
try:
    import openai  # type: ignore
except Exception:
    openai = None

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None


# ---------------- GitHub HTTP adapters -----------------


def github_request(
    url: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
) -> Dict:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}",
    }
    response = requests.get(url, headers=headers, params=params, timeout=30)
    if response.status_code == 404:
        raise FileNotFoundError(f"GitHub resource not found: {url}")
    if response.status_code == 401:
        raise PermissionError("GitHub token is invalid or lacks necessary scopes.")
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API request failed with status {response.status_code}: {response.text}"
        )
    return response.json()


def fetch_file_from_pr(
    repo_name: str,
    pr_number: int,
    path: str,
    head_sha: str,
    github_token: str,
) -> str:
    url = f"{SETTINGS.github_api_base}/repos/{repo_name}/contents/{path}"
    data = github_request(url, github_token, params={"ref": head_sha})
    content = data.get("content")
    encoding = data.get("encoding")
    if content is None or encoding != "base64":
        raise ValueError(
            f"Unexpected content response for '{path}' (encoding={encoding!r})."
        )
    decoded = base64.b64decode(content)
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File '{path}' in PR {pr_number} is not valid UTF-8 text"
        ) from exc


# ---------------- LLM provider adapters -----------------


def call_openai(
    token: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gpt-5",
) -> str:
    if openai is None:
        raise RuntimeError("openai package not installed. Install with `pip install openai`.")
    client = openai.OpenAI(api_key=token)
    params = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # Some models (e.g., gpt-5) may not allow custom temperature.
    if model_name not in {"gpt-5"}:
        params["temperature"] = 0.2
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content.strip()


def call_anthropic(
    token: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "claude-3-5-sonnet-20240620",
) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package not installed. Install with `pip install anthropic`.")
    client = anthropic.Anthropic(api_key=token)
    response = client.messages.create(
        model=model_name,
        system=system_prompt,
        max_tokens=1500,
        temperature=0.2,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


def call_gemini(
    token: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "gemini-1.5-pro",
) -> str:
    if genai is None:
        raise RuntimeError("google-generativeai package not installed. Install with `pip install google-generativeai`.")
    genai.configure(api_key=token)
    model = genai.GenerativeModel(model_name)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    response = model.generate_content(prompt, generation_config={"temperature": 0.2})
    return response.text.strip()


PROVIDERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
}


def dispatch_review(
    provider: str,
    token: str,
    system_prompt: str,
    user_prompt: str,
    model_name: str,
) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDERS)}")
    return PROVIDERS[provider](token, system_prompt, user_prompt, model_name)
