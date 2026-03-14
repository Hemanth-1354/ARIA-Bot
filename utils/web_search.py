"""
utils/web_search.py
Real-time web search via Tavily API using LangChain integration.
"""

import sys
import os
import logging
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 4) -> Optional[str]:
    """
    Perform a real-time web search using Tavily and return formatted results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 4).

    Returns:
        Formatted string of web search results, or None if search fails.

    Raises:
        RuntimeError: If the Tavily API key is missing or the search errors out.
    """
    try:
        if not TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY is not set. "
                "Add it to your environment variables or Streamlit secrets."
            )

        
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )

        if not response:
            logger.warning("Tavily returned an empty response for query: %s", query)
            return None

        parts = []

       
        if response.get("answer"):
            parts.append(f"**Summary:** {response['answer']}\n")

        
        results = response.get("results", [])
        if results:
            parts.append("**Sources:**")
            for i, result in enumerate(results, 1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                content = result.get("content", "").strip()
                snippet = content[:300] + "..." if len(content) > 300 else content
                parts.append(f"{i}. **{title}**\n   {snippet}\n   🔗 {url}")

        if not parts:
            return "No relevant web results found."

        formatted = "\n\n".join(parts)
        logger.info("Web search completed for query: '%s' — %d results.", query, len(results))
        return formatted

    except ValueError as ve:
        logger.error("Web search config error: %s", ve)
        raise RuntimeError(str(ve)) from ve

    except ImportError:
        logger.error("tavily-python package not installed.")
        raise RuntimeError(
            "The 'tavily-python' package is required. Run: pip install tavily-python"
        )

    except Exception as e:
        logger.error("Web search failed for query '%s': %s", query, e)
        raise RuntimeError(f"Web search failed: {e}") from e
