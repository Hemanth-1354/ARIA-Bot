"""
utils/followup.py
Generates 3 smart follow-up question suggestions after every assistant reply.
Uses the same Groq model — lightweight, fast, concise system prompt.
"""

import sys
import os
import logging
import json
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.llm import get_chatgroq_model

logger = logging.getLogger(__name__)

_FOLLOWUP_PROMPT = """You are a research assistant helping a user explore a topic deeper.
Given the user's last question and the assistant's reply, generate exactly 3 short follow-up questions the user might want to ask next.

Rules:
- Each question must be on a NEW LINE, no numbering, no bullets, no extra text
- Keep each question under 12 words
- Make them specific and directly related to the answer
- Do NOT repeat or rephrase the original question
- Output ONLY the 3 questions, nothing else
"""


def generate_followups(user_question: str, assistant_answer: str) -> List[str]:
    """
    Generate 3 follow-up question suggestions based on the last Q&A pair.

    Args:
        user_question:    The user's last question.
        assistant_answer: The assistant's response to that question.

    Returns:
        List of up to 3 follow-up question strings.
        Returns empty list silently on failure (non-critical feature).
    """
    try:
        model = get_chatgroq_model(mode="concise")

        user_content = (
            f"User asked: {user_question}\n\n"
            f"Assistant answered: {assistant_answer[:1200]}\n\n"
            "Generate 3 follow-up questions:"
        )

        from langchain_core.messages import SystemMessage, HumanMessage
        response = model.invoke([
            SystemMessage(content=_FOLLOWUP_PROMPT),
            HumanMessage(content=user_content),
        ])

        raw = response.content.strip()
        questions = [
            q.strip().lstrip("•-123456789. ")
            for q in raw.split("\n")
            if q.strip() and len(q.strip()) > 5
        ]

        #  3 follow-ups
        valid = [q for q in questions if len(q) > 5][:3]
        logger.info("Generated %d follow-up suggestions.", len(valid))
        return valid

    except Exception as e:
        # Follow-ups
        logger.warning("Follow-up generation failed (non-critical): %s", e)
        return []
