"""
GitHub PR creation agent using Langchain.
This code integrates with the actual GitHub API using the PyGithub library.
Please set the GITHUB_TOKEN environment variable and install required libraries before running.
"""

import os
import re
import json
from typing import Optional, Dict, List, Tuple, Any

# Load environment variables from .env file
from dotenv import load_dotenv
from translator.content import llm_translate

load_dotenv()

# Constants definition
ANTHROPIC_MODEL_ID = "claude-sonnet-4-20250514"
DEFAULT_TEMPERATURE = 0.0

# Library imports and error handling
try:
    from github import Github, GithubException
    from github.GitRef import GitRef
    from langchain_anthropic import ChatAnthropic

    REQUIRED_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    print("Please run: pip install PyGithub boto3 langchain-anthropic")
    REQUIRED_LIBS_AVAILABLE = False


class GitHubPRAgent:
    """Agent class for GitHub PR creation"""

    def __init__(self, user_owner: str = None, user_repo: str = None, base_owner: str = None, base_repo: str = None):
        self._github_client = None
        self._llm = None
        self.user_owner = user_owner
        self.user_repo = user_repo
        self.base_owner = base_owner
        self.base_repo = base_repo

    @property
    def github_client(self) -> Optional[Github]:
        """Return GitHub API client with lazy initialization."""
        if not REQUIRED_LIBS_AVAILABLE:
            raise ImportError("Required libraries not found.")

        if self._github_client is None:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                print("Warning: GITHUB_TOKEN environment variable not set.")
                return Github()  # Limited access
            self._github_client = Github(token)

        return self._github_client

    @property
    def llm(self):
        """Return LLM client with lazy initialization."""
        if not REQUIRED_LIBS_AVAILABLE:
            raise ImportError("Required libraries not found.")

        if self._llm is None:
            self._llm = ChatAnthropic(
                model=ANTHROPIC_MODEL_ID,
                temperature=DEFAULT_TEMPERATURE,
            )
        return self._llm

    def _handle_github_error(self, e: Exception, operation: str) -> str:
        """Handle GitHub API errors consistently."""
        if isinstance(e, GithubException):
            return f"{operation} failed: {e.status} {e.data.get('message', e.data)}"
        return f"Unexpected error during {operation}: {str(e)}"

    def create_pull_request(
        self,
        owner: str,
        repo_name: str,
        title: str,
        head: str,
        base: str,
        body: str = "",
        draft: bool = False,
        maintainer_can_modify: bool = True,
    ) -> str:
        """Create a new Pull Request."""
        try:
            # 1. Check if head and base are the same
            if head == base:
                return f"ERROR: head branch ({head}) and base branch ({base}) are identical."

            # 2. Check for existing PR
            existing_pr = self.check_existing_pr(owner, repo_name, head, base)
            if existing_pr:
                return f"ERROR: {existing_pr}"

            # 3. Verify head and base branches exist
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            try:
                # For fork-to-upstream PR, head format is "fork_owner:branch_name"
                if ":" in head:
                    fork_owner, branch_name = head.split(":", 1)
                    fork_repo = self.github_client.get_repo(f"{fork_owner}/{repo_name}")
                    head_branch = fork_repo.get_branch(branch_name)
                else:
                    head_branch = repo.get_branch(head)

                base_branch = repo.get_branch(base)

                # 4. Check if head and base branches point to the same commit
                if head_branch.commit.sha == base_branch.commit.sha:
                    return f"ERROR: head branch ({head}) and base branch ({base}) point to the same commit. No changes to merge."

            except GithubException as e:
                if e.status == 404:
                    return f"ERROR: Branch not found. head: {head}, base: {base}"

            # 5. Create PR
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head,
                base=base,
                draft=draft,
                maintainer_can_modify=maintainer_can_modify,
            )
            return f"PR creation successful: {pr.html_url}"
        except GithubException as e:
            if e.status == 422:
                error_msg = e.data.get("message", "Unknown error")
                errors = e.data.get("errors", [])

                error_details = []
                for error in errors:
                    if "message" in error:
                        error_details.append(error["message"])

                detail_msg = " | ".join(error_details) if error_details else ""
                return f"ERROR: PR creation failed (422): {error_msg}. {detail_msg}"
            return self._handle_github_error(e, "PR creation")
        except Exception as e:
            return self._handle_github_error(e, "PR creation")

    def create_branch(
        self, owner: str, repo_name: str, branch_name: str, source_sha: str
    ) -> str:
        """Create a new branch."""
        try:
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            ref_name = f"refs/heads/{branch_name}"
            new_ref = repo.create_git_ref(ref=ref_name, sha=source_sha)

            if isinstance(new_ref, GitRef):
                return f"SUCCESS: Branch '{branch_name}' created successfully (ref: {new_ref.ref})"
            return f"ERROR: Branch '{branch_name}' creation failed. Please check API response."
        except GithubException as e:
            if e.status == 422 and "Reference already exists" in str(e.data):
                return f"WARNING: Branch '{branch_name}' already exists."
            return self._handle_github_error(e, "branch creation")
        except Exception as e:
            return self._handle_github_error(e, "branch creation")

    def check_existing_pr(
        self, owner: str, repo_name: str, head: str, base: str
    ) -> Optional[str]:
        """Check if there's an existing PR with the same head and base."""
        try:
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            # For head parameter, use exactly what was passed (could be "fork_owner:branch" or just "branch")
            search_head = head if ":" in head else f"{owner}:{head}"
            pulls = repo.get_pulls(state="open", head=search_head, base=base)
            for pr in pulls:
                return f"Existing PR found: {pr.html_url}"
            return None
        except Exception as e:
            print(f"⚠️ Error checking existing PR: {str(e)}")
            return None

    def create_or_update_file(
        self,
        owner: str,
        repo_name: str,
        path: str,
        message: str,
        content: str,
        branch_name: Optional[str] = None,
        sha_blob: Optional[str] = None,
    ) -> str:
        """Create or update a single file."""
        try:
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")

            args = {
                "path": path,
                "message": message,
                "content": content,
            }
            if branch_name:
                args["branch"] = branch_name

            # Try to update file
            if sha_blob:
                args["sha"] = sha_blob
                repo.update_file(**args)
                return f"SUCCESS: File updated - {path}"

            # Try to create file
            repo.create_file(**args)
            return f"SUCCESS: File created - {path}"

        except GithubException as e:
            # Try to update if file already exists
            if e.status == 422:
                try:
                    existing_file = repo.get_contents(
                        path, ref=branch_name or repo.default_branch
                    )
                    args["sha"] = existing_file.sha
                    repo.update_file(**args)
                    return f"SUCCESS: File updated - {path}"
                except:
                    pass
            return f"ERROR: File processing failed - {path}"
        except Exception:
            return f"ERROR: File processing failed - {path}"

    def analyze_reference_pr(self, pr_url: str) -> Dict[str, Any]:
        """Analyze reference PR to extract style information."""
        try:
            # Parse PR URL
            match = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
            if not match:
                return {"error": f"Invalid PR URL format: {pr_url}"}

            owner, repo_name, pr_number = match.groups()
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            pr = repo.get_pull(int(pr_number))

            return {
                "title": pr.title,
                "body": pr.body,
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
                "files_changed": [f.filename for f in pr.get_files()],
                "commits": [
                    {"message": c.commit.message, "sha": c.sha}
                    for c in pr.get_commits()
                ],
            }
        except Exception as e:
            return {"error": f"Error occurred during PR analysis: {str(e)}"}

    def _generate_with_llm(
        self, prompt: str, fallback_value: str, operation: str
    ) -> str:
        """Generate text using LLM."""
        try:
            _usage_info, generated = llm_translate(prompt)
            generated = generated.strip()
            print(f"LLM generated {operation}: {generated}")
            return generated
        except Exception as e:
            print(f"❌ Error generating {operation} with LLM: {e}")
            print(f"Using fallback value: {fallback_value}")
            return fallback_value

    def generate_branch_name_from_reference(
        self, reference_branch_name: str, target_language: str, file_name: str
    ) -> str:
        """Generate branch name using simple template."""
        # Keep .md extension and make branch-safe
        branch_safe_name = file_name.replace('_', '-')
        return f"{target_language}-{branch_safe_name}"

    def generate_pr_content_from_reference(
        self,
        reference_title: str,
        reference_body: str,
        target_language: str,
        filepath: str,
        target_filepath: str,
        file_name: str,
    ) -> Tuple[str, str]:
        """Use LLM to analyze reference PR title and body and generate appropriate PR content."""
        prompt = f"""Here is the reference PR information:

Reference PR title: {reference_title}

Reference PR body:
{reference_body}

Now I need to generate PR title and body for a new translation task:
- Target language: {target_language}
- Original file: {filepath}
- Translation file: {target_filepath}
- File name: {file_name}

Please analyze the style and format of the reference PR to generate consistent new PR title and body.

Requirements:
1. Follow the title format and pattern of the reference PR
2. Maintain the body style, markdown format, indentation, and line breaks of the reference PR
3. Appropriately reflect the target language ({target_language}) and file paths
4. If there are user mentions (@username), change them to general text instead of actual mentions
5. Adjust the content to fit the translation task

Response format:
Title: [PR title here]
Body: [PR body here, maintaining the exact markdown format and structure of the original]"""

        try:
            _usage_info, generated_content = llm_translate(prompt)
            generated_content = generated_content.strip()

            # Separate title and body from response
            lines = generated_content.split("\n")
            title_line = ""
            body_lines = []
            parsing_body = False

            for line in lines:
                if line.startswith("Title:"):
                    title_line = line.replace("Title:", "").strip()
                elif line.startswith("Body:"):
                    parsing_body = True
                    body_content = line.replace("Body:", "").strip()
                    if body_content:
                        body_lines.append(body_content)
                elif parsing_body:
                    body_lines.append(line)

            generated_title = title_line if title_line else reference_title
            generated_body = (
                "\n".join(body_lines)
                if body_lines
                else f"Add {target_language} translation for `{filepath}`."
            )

            print(f"LLM generated PR title: {generated_title}")
            print(f"LLM generated PR body (first 100 chars): {generated_body[:100]}...")

            return generated_title, generated_body

        except Exception as e:
            print(f"❌ Error generating PR content with LLM: {e}")
            return self._generate_default_pr_content(
                target_language, filepath, target_filepath, file_name
            )

    def _generate_default_pr_content(
        self, target_language: str, filepath: str, target_filepath: str, file_name: str
    ) -> Tuple[str, str]:
        """Generate default PR content."""
        title = f"🌐 [i18n-{target_language}] Translated `{file_name}` to {target_language}"
        body = f"""# What does this PR do?

Translated the `{filepath}` file of the documentation to {target_language} 😄 
Thank you in advance for your review!

Part of https://github.com/huggingface/transformers/issues/20179

## Before reviewing
- [x] Check for missing / redundant translations (번역 누락/중복 검사)
- [x] Grammar Check (맞춤법 검사)
- [x] Review or Add new terms to glossary (용어 확인 및 추가)
- [x] Check Inline TOC (e.g. `[[lowercased-header]]`)
- [x] Check live-preview for gotchas (live-preview로 정상작동 확인)

## Who can review? (Initial)
{target_language} translation reviewers

## Before submitting
- [x] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#start-contributing-pull-requests),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?

## Who can review? (Final)
 May you please review this PR?
Documentation maintainers
"""
        return title, body

    def generate_commit_message_from_reference(
        self, commit_messages: List[str], target_language: str, file_name: str
    ) -> str:
        """Generate simple commit message using template."""
        return f"docs: {target_language}: {file_name}"

    def get_branch_info(self, owner: str, repo_name: str, branch_name: str) -> str:
        """Get information about an existing branch."""
        try:
            repo = self.github_client.get_repo(f"{owner}/{repo_name}")
            branch = repo.get_branch(branch_name)
            commit = branch.commit
            commit_info = commit.commit

            return f"""
📋 Existing branch information:
  - Branch name: {branch_name}
  - Latest commit: {commit.sha[:8]}
  - Commit message: {commit_info.message.split(chr(10))[0][:80]}...
  - Author: {commit_info.author.name}
  - Date: {commit_info.author.date.strftime('%Y-%m-%d %H:%M:%S')}
            """
        except Exception as e:
            return f"Failed to retrieve branch information: {str(e)}"

    def run_translation_pr_workflow(
        self,
        reference_pr_url: str,
        target_language: str,
        filepath: str,
        translated_doc: str,
        base_branch: str = "main",
    ) -> Dict[str, Any]:
        """Execute translation document PR creation workflow."""
        try:
            # 1. Analyze reference PR
            print(f"🔍 Analyzing reference PR: {reference_pr_url}")
            pr_analysis = self.analyze_reference_pr(reference_pr_url)

            if "error" in pr_analysis:
                return {"status": "error", "message": pr_analysis["error"]}

            print("Reference PR analysis completed")

            # 2. Generate translation file path and branch name
            target_filepath = filepath.replace("/en/", f"/{target_language}/")
            file_name = filepath.split("/")[-1]  # Keep .md extension

            print(f"🌿 Generating branch name...")
            branch_name = self.generate_branch_name_from_reference(
                pr_analysis["head_branch"], target_language, file_name
            )

            # 3. Get main branch SHA from upstream and create branch in fork
            upstream_repo = self.github_client.get_repo(f"{self.base_owner}/{self.base_repo}")
            main_branch = upstream_repo.get_branch(base_branch)
            main_sha = main_branch.commit.sha

            print(f"🌿 Creating branch: {branch_name} in fork repository")
            branch_result = self.create_branch(self.user_owner, self.user_repo, branch_name, main_sha)

            # Check branch creation result
            if branch_result.startswith("ERROR"):
                return {
                    "status": "error",
                    "message": f"Branch creation failed: {branch_result}\n\nTarget: {self.user_owner}/{self.user_repo}\nBranch: {branch_name}\nBase SHA: {main_sha[:8]}",
                    "branch": branch_name,
                    "error_details": branch_result,
                }
            elif branch_result.startswith("WARNING"):
                print(f"⚠️ {branch_result}")
                # Continue if branch already exists
            elif branch_result.startswith("SUCCESS"):
                print(f"✅ {branch_result}")
            else:
                print(f"⚠️ Unexpected branch creation result: {branch_result}")
                # Continue anyway, might still work

            # 4. Generate commit message and save file
            commit_messages = [commit["message"] for commit in pr_analysis["commits"]]
            commit_message = self.generate_commit_message_from_reference(
                commit_messages, target_language, file_name
            )

            print(f"📄 Saving file: {target_filepath}")
            file_result = self.create_or_update_file(
                self.user_owner,
                self.user_repo,
                target_filepath,
                commit_message,
                translated_doc,
                branch_name,
            )

            if not file_result.startswith("SUCCESS"):
                return {
                    "status": "error",
                    "message": f"File save failed: {file_result}\n\n🎯 Target: {self.user_owner}/{self.user_repo} (expected: {target_language} fork of {self.base_owner}/{self.base_repo})\n🌿 Branch: {branch_name}\n📁 File: {target_filepath}",
                    "branch": branch_name,
                    "file_path": target_filepath,
                    "error_details": file_result,
                }

            print(f"{file_result}")

            # 5. Create PR
            pr_title, pr_body = self.generate_pr_content_from_reference(
                pr_analysis["title"],
                pr_analysis["body"],
                target_language,
                filepath,
                target_filepath,
                file_name,
            )

            print(f"🔄 Creating PR: {pr_title}")
            print(f"   Head: {self.user_owner}:{branch_name} → Base: {self.base_owner}:{base_branch}")

            # Create PR from fork to upstream repository
            pr_result = self.create_pull_request(
                self.base_owner, self.base_repo, pr_title, f"{self.user_owner}:{branch_name}", base_branch, pr_body, draft=True
            )

            if pr_result.startswith("ERROR"):
                print(f"❌ {pr_result}")
                return {
                    "status": "partial_success",
                    "branch": branch_name,
                    "file_path": target_filepath,
                    "message": f"File was saved and commit was successful.\nPR creation failed: {pr_result}",
                    "error_details": pr_result,
                }
            elif "successful" in pr_result and "http" in pr_result:
                print(f"{pr_result}")
                return {
                    "status": "success",
                    "branch": branch_name,
                    "file_path": target_filepath,
                    "pr_url": pr_result.split(": ")[-1],
                    "message": "Translation document PR created successfully!",
                }
            else:
                return {
                    "status": "partial_success",
                    "branch": branch_name,
                    "file_path": target_filepath,
                    "message": "File was saved but PR creation failed.",
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Workflow execution failed: {str(e)}\n\nConfig: {self.user_owner}/{self.user_repo} → {self.base_owner}/{self.base_repo}\nFile: {filepath if 'filepath' in locals() else 'Unknown'}",
                "error_details": str(e),
            }


# Backward compatibility functions (maintain compatibility with existing code)
_agent = GitHubPRAgent()


def get_github_client():
    return _agent.github_client


def create_pull_request_func(*args, **kwargs):
    return _agent.create_pull_request(*args, **kwargs)


def create_branch_func(*args, **kwargs):
    return _agent.create_branch(*args, **kwargs)


def create_or_update_file_func(*args, **kwargs):
    return _agent.create_or_update_file(*args, **kwargs)


def analyze_reference_pr_func(*args, **kwargs):
    return _agent.analyze_reference_pr(*args, **kwargs)


def generate_branch_name_from_reference(*args, **kwargs):
    return _agent.generate_branch_name_from_reference(*args, **kwargs)


def generate_pr_content_from_reference(*args, **kwargs):
    return _agent.generate_pr_content_from_reference(*args, **kwargs)


def generate_default_pr_content(*args, **kwargs):
    return _agent._generate_default_pr_content(*args, **kwargs)


def generate_commit_message_from_reference(*args, **kwargs):
    return _agent.generate_commit_message_from_reference(*args, **kwargs)


def get_branch_info(*args, **kwargs):
    return _agent.get_branch_info(*args, **kwargs)


def run_translation_pr_agent_simple(*args, **kwargs):
    return _agent.run_translation_pr_workflow(*args, **kwargs)
