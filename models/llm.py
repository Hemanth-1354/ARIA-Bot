import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL, CONCISE_MAX_TOKENS, DETAILED_MAX_TOKENS

logger = logging.getLogger(__name__)


def get_chatgroq_model(mode: str = "detailed") -> ChatGroq:
    """
    Initialize and return the Groq chat model.

    Args:
        mode: 'concise' for short replies, 'detailed' for in-depth responses.

    Returns:
        ChatGroq instance configured for the chosen mode.

    Raises:
        RuntimeError: If the Groq model cannot be initialized.
    """
    try:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your environment variables or Streamlit secrets."
            )

        max_tokens = CONCISE_MAX_TOKENS if mode == "concise" else DETAILED_MAX_TOKENS

        model = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        logger.info("Groq model '%s' initialised in '%s' mode.", GROQ_MODEL, mode)
        return model

    except ValueError as ve:
        logger.error("Configuration error: %s", ve)
        raise RuntimeError(str(ve)) from ve

    except Exception as e:
        logger.error("Failed to initialise Groq model: %s", e)
        raise RuntimeError(f"Failed to initialise Groq model: {e}") from e
