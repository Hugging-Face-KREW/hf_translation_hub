import os
import base64
from typing import Optional

try:
    from github import Github, GithubException
    LIBS_OK = True
except ImportError:
    LIBS_OK = False

class GitHubLogger:
    """Dedicated logger that appends JSONL entries to a GitHub repo/branch/file.

    Env vars:
      - LOG_GITHUB_TOKEN (fallback: GITHUB_TOKEN)
      - LOG_REPO (format: owner/repo)
      - LOG_BRANCH (default: 'log_event')
      - LOG_FILE_PATH (default: 'pr_success.log')
    """

    def __init__(self):
        if not LIBS_OK:
            raise ImportError("PyGithub not installed. Please install PyGithub.")
        token = os.environ.get("LOG_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("Missing LOG_GITHUB_TOKEN or GITHUB_TOKEN for logging")
        self._client = Github(token)

        repo_spec = os.environ.get("LOG_REPO")
        if not repo_spec or "/" not in repo_spec:
            raise ValueError("Missing or invalid LOG_REPO. Expected 'owner/repo'.")
        self.owner, self.repo_name = repo_spec.split("/", 1)

        self.branch = os.environ.get("LOG_BRANCH", "log_event")
        self.path = os.environ.get("LOG_FILE_PATH", "pr_success.log")

    def _ensure_branch(self, repo):
        try:
            repo.get_branch(self.branch)
        except GithubException as e:
            if e.status == 404:
                base = repo.get_branch(repo.default_branch)
                repo.create_git_ref(ref=f"refs/heads/{self.branch}", sha=base.commit.sha)
            else:
                raise

    def append_jsonl(self, jsonl_line: str, commit_message: str = "chore(log): append entry") -> str:
        repo = self._client.get_repo(f"{self.owner}/{self.repo_name}")
        self._ensure_branch(repo)
        try:
            existing = repo.get_contents(self.path, ref=self.branch)
            existing_content = base64.b64decode(existing.content).decode("utf-8")
            new_content = existing_content + jsonl_line
            repo.update_file(
                path=self.path,
                message=commit_message,
                content=new_content,
                sha=existing.sha,
                branch=self.branch,
            )
            return "SUCCESS: Log appended"
        except GithubException as e:
            if e.status == 404:
                repo.create_file(
                    path=self.path,
                    message=commit_message,
                    content=jsonl_line,
                    branch=self.branch,
                )
                return "SUCCESS: Log file created and first entry appended"
            raise
