"""
GitHub PR Search Agent
An agent that finds a suitable reference PR when a reference PR URL is not provided.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Langchain imports
try:
    from langchain_anthropic import ChatAnthropic
    from langchain.tools import StructuredTool
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    from github import Github

    REQUIRED_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"Required libraries are not installed: {e}")
    REQUIRED_LIBS_AVAILABLE = False

# Constants
ANTHROPIC_MODEL_ID = "claude-sonnet-4-20250514"
DEFAULT_TEMPERATURE = 0.0
# Fallback PR URL to ensure a PR is always returned
DEFAULT_FALLBACK_PR_URL = "https://github.com/huggingface/transformers/pull/24968"


class GitHubPRSearcher:
    """GitHub PR Searcher - now using a LangChain agent."""

    def _search_github_prs(self, query: str) -> List[Dict[str, Any]]:
        """
        Searches GitHub for pull requests matching the query and returns the top 5 results.
        The query should be a valid GitHub search query.
        """
        logger.info(f"Executing GitHub search with query: {query}")
        try:
            issues = self.github_client.search_issues(query=query)
            # Take top 5 to keep context small for the agent
            top_issues = issues.get_page(0)[:5]

            if not top_issues:
                return []

            return [
                {"title": issue.title, "url": issue.html_url, "number": issue.number}
                for issue in top_issues
            ]
        except Exception as e:
            logger.error(f"Error during GitHub search: {e}", exc_info=True)
            # Return an error message that the agent can understand
            return [{"error": f"An error occurred during search: {e}"}]

    def __init__(self):
        if not REQUIRED_LIBS_AVAILABLE:
            raise ImportError("Required libraries for agent could not be found.")

        self._github_client = None
        self.llm = ChatAnthropic(
            model=ANTHROPIC_MODEL_ID,
            temperature=DEFAULT_TEMPERATURE,
        )

        search_tool = StructuredTool.from_function(
            func=self._search_github_prs,
            name="search_github_prs",
            description="Searches GitHub for pull requests matching the query and returns the top 5 results. The query should be a valid GitHub search query.",
        )
        tools = [search_tool]

        prompt_string = """You are a GitHub expert. Your mission is to find the best reference pull request (PR) for a given task.

You need to find a merged PR in the repository: {owner}/{repo_name}.
The PR should be for a documentation translation into **{target_language}**.
The context for the translation is: **{context}**.

Use the tools at your disposal to search for relevant PRs.
Analyze the search results and select the one that best matches the request. A good PR is usually one that has "translation", "docs", "i18n", and the target language in its title.

Here is an example of a good search query you could use:
`repo:{owner}/{repo_name} is:pr is:merged "{target_language}" "{context}" i18n translation docs`

After your analysis, you MUST output **only the final URL** of the best PR you have chosen. Do not include any other text in your final response."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_string),
                (
                    "human",
                    "Find the best reference PR for translating docs to {target_language} about {context} in the {owner}/{repo_name} repository.",
                ),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    @property
    def github_client(self) -> Optional[Github]:
        """Lazy initialization of the GitHub API client."""
        if not REQUIRED_LIBS_AVAILABLE:
            raise ImportError("Required libraries could not be found.")

        if self._github_client is None:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                print("Warning: GITHUB_TOKEN environment variable is not set.")
                self._github_client = Github()  # Limited access
            else:
                self._github_client = Github(token)
        return self._github_client

    def find_best_reference_pr(
        self, owner: str, repo_name: str, target_language: str, context: str
    ):
        """
        Finds the best reference PR using a LangChain agent.
        Yields progress and returns the final PR URL.
        """
        message = "🤖 Agent is searching for the best reference PR..."
        logger.info(message)
        yield message

        try:
            agent_input = {
                "owner": owner,
                "repo_name": repo_name,
                "target_language": target_language,
                "context": context,
            }

            agent_output = None
            for event in self.agent_executor.stream(agent_input):
                if "actions" in event and event["actions"]:
                    action = event["actions"][0]
                    tool_query = action.tool_input.get("query", str(action.tool_input))
                    message = f"🔍 Agent is using tool `{action.tool}` with query:\n`{tool_query}`"
                    logger.info(message)
                    yield message
                elif "steps" in event and event["steps"]:
                    message = "📊 Agent is analyzing the results from the tool..."
                    logger.info(message)
                    yield message
                elif "output" in event and event["output"]:
                    agent_output = event["output"]

            if not agent_output:
                message = "⚠️ Agent failed to find a suitable PR. Using default PR."
                logger.warning(message)
                yield message
                return DEFAULT_FALLBACK_PR_URL

            # The agent's final output can be a string, a list of tool results,
            # or a list of content blocks from the LLM. We'll find the URL
            # by searching for it in the string representation of the output.
            output_text = str(agent_output)
            urls = re.findall(r"https?://github.com/[^/]+/[^/]+/pull/\d+", output_text)

            final_url = ""
            if urls:
                final_url = urls[-1]  # Take the last URL found

            if not final_url:
                message = f"⚠️ Agent returned unparsable output: {agent_output}. Using default PR."
                logger.warning(message)
                yield message
                return DEFAULT_FALLBACK_PR_URL

            message = f"✅ Selected the best PR:\n`{final_url}`"
            logger.info(f"Selected the best PR: {final_url}")
            yield message
            return final_url

        except Exception as e:
            message = f"❌ Error during agent execution: {e}\nUsing default PR."
            logger.error(message, exc_info=True)
            yield message
            return DEFAULT_FALLBACK_PR_URL


def find_reference_pr_simple_stream(target_language: str = "", context: str = ""):
    """
    A simple function to find a reference PR, streaming progress.
    This function always searches in the 'huggingface/transformers' repository.
    """
    searcher = GitHubPRSearcher()
    stream_generator = searcher.find_best_reference_pr(
        "huggingface", "transformers", target_language, context
    )
    # The handler will receive the final URL from the generator's return statement
    final_url = yield from stream_generator

    # Format the final result as expected by the handler
    return {
        "status": "success",
        "result": f"Recommended PR URL: {final_url}",
        "repository": "huggingface/transformers",
        "target_language": target_language,
    }


# Example usage
if __name__ == "__main__":
    # Example execution for streaming
    # In a real application, a generator consumer (like the one in handler.py)
    # would process the yielded values. This script simulates that.
    print("--- Running Streaming Search Simulation ---")

    def run_simulation():
        """Simulates the consumption of the streaming generator."""
        test_gen = find_reference_pr_simple_stream(
            target_language="korean", context="docs"
        )
        try:
            while True:
                # This will print progress messages
                print(next(test_gen))
        except StopIteration as e:
            # When the generator is exhausted, the final result is in e.value
            print("\n--- FINAL RESULT ---")
            print(e.value)

    run_simulation()
