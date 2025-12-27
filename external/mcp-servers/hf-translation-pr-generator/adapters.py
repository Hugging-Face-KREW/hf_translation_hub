"""External API adapters for GitHub operations."""

import os
import requests
from typing import Dict, Any, List, Optional


def get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers with authentication if available."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    return headers


def check_github_token_validity(token: str) -> Dict[str, Any]:
    """Check if GitHub token is valid and get user info."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        response = requests.get("https://api.github.com/user", headers=headers)
        if response.status_code == 200:
            user_data = response.json()
            return {
                "valid": True,
                "user": user_data.get("login"),
                "name": user_data.get("name"),
                "scopes": response.headers.get("X-OAuth-Scopes", "").split(", ") if response.headers.get("X-OAuth-Scopes") else []
            }
        else:
            return {
                "valid": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def get_repository_info(owner: str, repo: str, token: str = None) -> Dict[str, Any]:
    """Get repository information from GitHub API."""
    headers = get_github_headers()
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            repo_data = response.json()
            return {
                "exists": True,
                "fork": repo_data.get("fork", False),
                "private": repo_data.get("private", False),
                "permissions": repo_data.get("permissions", {}),
                "default_branch": repo_data.get("default_branch", "main"),
                "parent": repo_data.get("parent", {}).get("full_name") if repo_data.get("parent") else None
            }
        elif response.status_code == 404:
            return {
                "exists": False,
                "error": "Repository not found"
            }
        else:
            return {
                "exists": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }


def search_github_prs(
    query: str,
    sort: str = "updated",
    order: str = "desc",
    per_page: int = 30
) -> Dict[str, Any]:
    """Search GitHub PRs using the search API."""
    headers = get_github_headers()
    
    try:
        url = "https://api.github.com/search/issues"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page
        }
        
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            return {
                "success": True,
                "data": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_pr_details(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """Get detailed information about a specific PR."""
    headers = get_github_headers()
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            pr_data = response.json()
            
            # Also get files changed
            files_url = f"{url}/files"
            files_response = requests.get(files_url, headers=headers)
            files_changed = []
            
            if files_response.status_code == 200:
                files_data = files_response.json()
                files_changed = [f["filename"] for f in files_data]
            
            return {
                "success": True,
                "pr": {
                    "title": pr_data.get("title"),
                    "body": pr_data.get("body"),
                    "state": pr_data.get("state"),
                    "merged": pr_data.get("merged"),
                    "created_at": pr_data.get("created_at"),
                    "updated_at": pr_data.get("updated_at"),
                    "user": pr_data.get("user", {}).get("login"),
                    "base_branch": pr_data.get("base", {}).get("ref"),
                    "head_branch": pr_data.get("head", {}).get("ref"),
                    "files_changed": files_changed
                }
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def create_or_update_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str = "main",
    token: str = None
) -> Dict[str, Any]:
    """Create or update a file in GitHub repository."""
    headers = get_github_headers()
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        # First, try to get existing file to get SHA
        get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        get_params = {"ref": branch}
        get_response = requests.get(get_url, headers=headers, params=get_params)
        
        # Prepare content (base64 encoded)
        import base64
        encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        data = {
            "message": message,
            "content": encoded_content,
            "branch": branch
        }
        
        # If file exists, add SHA for update
        if get_response.status_code == 200:
            file_data = get_response.json()
            data["sha"] = file_data["sha"]
        
        # Create/update file
        put_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        response = requests.put(put_url, headers=headers, json=data)
        
        if response.status_code in [200, 201]:
            result_data = response.json()
            return {
                "success": True,
                "commit": {
                    "sha": result_data["commit"]["sha"],
                    "message": message,
                    "url": result_data["commit"]["html_url"]
                },
                "content": {
                    "path": path,
                    "sha": result_data["content"]["sha"]
                }
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def create_pull_request(
    owner: str,
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str = "main",
    token: str = None
) -> Dict[str, Any]:
    """Create a new pull request."""
    headers = get_github_headers()
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 201:
            pr_data = response.json()
            return {
                "success": True,
                "pr": {
                    "number": pr_data["number"],
                    "html_url": pr_data["html_url"],
                    "state": pr_data["state"]
                }
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def create_branch(
    owner: str,
    repo: str,
    branch_name: str,
    from_branch: str = "main",
    token: str = None
) -> Dict[str, Any]:
    """Create a new branch in GitHub repository."""
    headers = get_github_headers()
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        # Get SHA of the source branch
        ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{from_branch}"
        ref_response = requests.get(ref_url, headers=headers)
        
        if ref_response.status_code != 200:
            return {
                "success": False,
                "error": f"Could not get reference for branch {from_branch}"
            }
        
        ref_data = ref_response.json()
        sha = ref_data["object"]["sha"]
        
        # Create new branch
        create_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": sha
        }
        
        response = requests.post(create_url, headers=headers, json=data)
        
        if response.status_code == 201:
            return {
                "success": True,
                "branch": branch_name,
                "sha": sha
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }