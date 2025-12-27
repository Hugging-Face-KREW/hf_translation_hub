"""
GitHub Agent for MCP Server - copied and adapted from original GitHubPRAgent.
Removes LLM dependencies - PR title/description provided by MCP client.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

try:
    from github import Github, GithubException
    from github.GitRef import GitRef
    GITHUB_AVAILABLE = True
except ImportError:
    print("PyGithub not available. Install with: pip install PyGithub")
    GITHUB_AVAILABLE = False


class GitHubAgent:
    """GitHub Agent without LLM dependencies - adapted from original GitHubPRAgent."""

    def __init__(self, user_owner: str, user_repo: str, base_owner: str, base_repo: str):
        self.user_owner = user_owner
        self.user_repo = user_repo  
        self.base_owner = base_owner
        self.base_repo = base_repo
        self._github_client = None

    @property
    def github_client(self) -> Optional[Github]:
        """Return GitHub API client with lazy initialization."""
        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub not available")
        
        if self._github_client is None:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise ValueError("GITHUB_TOKEN environment variable required")
            self._github_client = Github(token)
        
        return self._github_client

    def run_translation_pr_workflow(
        self,
        reference_pr_url: str,
        target_language: str,
        filepath: str,
        translated_doc: str,
        pr_title: str,
        pr_description: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Execute translation document PR creation workflow (without LLM)."""
        if not GITHUB_AVAILABLE:        
            target_filepath = filepath.replace("/en/", f"/{target_language}/")
            file_name = filepath.split('/')[-1].replace('.md', '').replace('_', '-')
            branch_name = f"{target_language}-{file_name}"

            return {
                "status": "success",
                "simulation": True,
                "branch": branch_name,
                "file_path": target_filepath,
                "pr_url": f"https://github.com/{self.user_owner}/{self.user_repo}/pull/1234",
                "message": "PyGithub not installed - simulation mode"
                }
        try:
            # Generate file path and branch name
            target_filepath = filepath.replace("/en/", f"/{target_language}/")
            file_name = filepath.split('/')[-1].replace('.md', '').replace('_', '-')
            branch_name = f"{target_language}-{file_name}"

            print(f"📁 Target file: {target_filepath}")
            print(f"🌿 Branch name: {branch_name}")

            # 1. Create branch
            branch_result = self._create_branch_for_pr(branch_name, base_branch)
            if "ERROR" in branch_result:
                return {"status": "error", "message": branch_result}

            # 2. Create/update file
            file_result = self._create_or_update_file_for_pr(
                target_filepath, translated_doc, branch_name
            )
            if "ERROR" in file_result:
                return {"status": "error", "message": file_result}

            # 3. Create pull request
            pr_result = self._create_pull_request_for_translation(
                pr_title, pr_description, branch_name, base_branch
            )
            
            if "ERROR" in pr_result:
                return {
                    "status": "partial_success",
                    "branch": branch_name,
                    "file_path": target_filepath,
                    "message": f"File was saved and commit was successful.\nPR creation failed: {pr_result}",
                    "error_details": pr_result
                }
            
            # Extract PR URL from result
            pr_url = None
            if "PR creation successful:" in pr_result:
                pr_url = pr_result.split("PR creation successful: ")[-1]
            
            return {
                "status": "success",
                "pr_url": pr_url,
                "branch": branch_name,
                "file_path": target_filepath,
                "message": f"Successfully created translation PR: {pr_url}"
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

    def _create_branch_for_pr(self, branch_name: str, base_branch: str) -> str:
        """Create a new branch for PR."""
        try:
            user_repo = self.github_client.get_repo(f"{self.user_owner}/{self.user_repo}")
            
            # Get base branch SHA
            base_ref = user_repo.get_git_ref(f"heads/{base_branch}")
            source_sha = base_ref.object.sha
            
            # Create branch
            ref_name = f"refs/heads/{branch_name}"
            new_ref = user_repo.create_git_ref(ref=ref_name, sha=source_sha)
            
            if isinstance(new_ref, GitRef):
                return f"SUCCESS: Branch '{branch_name}' created successfully"
            return f"ERROR: Branch '{branch_name}' creation failed"
            
        # except GithubException as e:
        #     if e.status == 422 and "Reference already exists" in str(e.data):
        #         return f"WARNING: Branch '{branch_name}' already exists (continuing)"
        #     return f"ERROR: Branch creation failed: {str(e)}"
        except Exception as e:
            return f"ERROR: Branch creation error: {str(e)}"

    def _create_or_update_file_for_pr(self, file_path: str, content: str, branch_name: str) -> str:
        """Create or update file in the branch."""
        try:
            user_repo = self.github_client.get_repo(f"{self.user_owner}/{self.user_repo}")
            
            commit_message = f"Add {file_path.split('/')[-2]} translation for {file_path.split('/')[-1]}"
            
            # Try to create file first
            try:
                user_repo.create_file(
                    path=file_path,
                    message=commit_message,
                    content=content,
                    branch=branch_name
                )
                return f"SUCCESS: File created - {file_path}"
                
            except Exception as e:
                if e.status == 422:
                    # File exists, try to update
                    try:
                        existing_file = user_repo.get_contents(file_path, ref=branch_name)
                        user_repo.update_file(
                            path=file_path,
                            message=commit_message,
                            content=content,
                            sha=existing_file.sha,
                            branch=branch_name
                        )
                        return f"SUCCESS: File updated - {file_path}"
                    except Exception as update_e:
                        return f"ERROR: File update failed - {str(update_e)}"
                else:
                    raise e
                    
        except Exception as e:
            return f"ERROR: File processing failed - {str(e)}"

    def _create_pull_request_for_translation(self, title: str, body: str, head_branch: str, base_branch: str) -> str:
        """Create pull request for translation."""
        try:
            base_repo = self.github_client.get_repo(f"{self.base_owner}/{self.base_repo}")
            
            # Format head for cross-repo PR
            head = f"{self.user_owner}:{head_branch}"
            
            # Check for existing PR
            existing_pr = self._check_existing_pr(base_repo, head, base_branch)
            if existing_pr:
                return f"ERROR: {existing_pr}"
            
            # Create PR
            pr = base_repo.create_pull(
                title=title,
                body=body,
                head=head,
                base=base_branch
            )
            return f"PR creation successful: {pr.html_url}"
            
        # except GithubException as e:
        #     if e.status == 422:
        #         error_msg = e.data.get("message", "Unknown error")
        #         return f"ERROR: PR creation failed (422): {error_msg}"
        #     return f"ERROR: PR creation failed: {str(e)}"
        except Exception as e:
            return f"ERROR: PR creation error: {str(e)}"

    def _check_existing_pr(self, repo, head: str, base: str) -> Optional[str]:
        """Check if there's an existing PR."""
        try:
            pulls = repo.get_pulls(state="open", head=head, base=base)
            for pr in pulls:
                return f"Existing PR found: {pr.html_url}"
            return None
        except Exception as e:
            print(f"⚠️ Warning: Could not check existing PRs: {e}")
            return None