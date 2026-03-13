"""
utils/prompt_builder.py
Builds system prompts for the AI Research Paper Assistant
based on the selected response mode and available context.
"""

import logging

logger = logging.getLogger(__name__)


BASE_PERSONA = """You are ARIA — AI Research Intelligence Assistant. \
You are an expert research assistant specialising in analysing academic papers, \
technical reports, and research documents. You help researchers, students, and \
professionals deeply understand complex research content.

Your strengths:
- Extracting key findings, methodologies, and contributions from papers
- Explaining complex technical concepts clearly
- Comparing and synthesising multiple sources
- Identifying research gaps and future directions
- Answering follow-up questions about uploaded documents
"""

CONCISE_INSTRUCTION = """
RESPONSE MODE: CONCISE
Provide SHORT, DIRECT answers only.
- Maximum 3-4 sentences or bullet points
- Lead with the core answer immediately
- No lengthy explanations unless absolutely necessary
- Use bullet points when listing items
"""

DETAILED_INSTRUCTION = """
RESPONSE MODE: DETAILED
Provide COMPREHENSIVE, IN-DEPTH responses.
- Cover all relevant aspects thoroughly
- Include context, reasoning, and examples where helpful
- Structure your response with clear sections if needed
- Cite specific parts of the document when referencing it
- Explain technical concepts step by step
"""

RAG_CONTEXT_TEMPLATE = """
DOCUMENT CONTEXT (retrieved from the uploaded research paper):
────────────────────────────────────────────────────────────
{context}
────────────────────────────────────────────────────────────
Answer the user's question using the document context above as your primary source. \
If the answer is not in the document, clearly state that and optionally use your general knowledge.
"""

WEB_CONTEXT_TEMPLATE = """
WEB SEARCH RESULTS (real-time information retrieved from the internet):
────────────────────────────────────────────────────────────
{web_results}
────────────────────────────────────────────────────────────
Use the above web search results to supplement your answer. \
Cite sources where relevant.
"""


def build_system_prompt(
    mode: str = "detailed",
    rag_context: str = "",
    web_context: str = "",
) -> str:
    """
    Construct the full system prompt for the LLM.

    Args:
        mode: 'concise' or 'detailed' response mode.
        rag_context: Retrieved document chunks from FAISS (may be empty).
        web_context: Results from Tavily web search (may be empty).

    Returns:
        Complete system prompt string.
    """
    try:
        parts = [BASE_PERSONA]

        # Response mode instruction
        if mode == "concise":
            parts.append(CONCISE_INSTRUCTION)
        else:
            parts.append(DETAILED_INSTRUCTION)

        # Inject RAG context if available
        if rag_context and rag_context.strip():
            parts.append(RAG_CONTEXT_TEMPLATE.format(context=rag_context))

        # Inject web search context if available
        if web_context and web_context.strip():
            parts.append(WEB_CONTEXT_TEMPLATE.format(web_results=web_context))

        # If no context at all
        if not rag_context and not web_context:
            parts.append(
                "\nNo document has been uploaded yet. "
                "Answer from your general knowledge about AI and research. "
                "Encourage the user to upload a research paper for document-specific Q&A."
            )

        prompt = "\n".join(parts)
        logger.info(
            "System prompt built | mode=%s | has_rag=%s | has_web=%s",
            mode,
            bool(rag_context),
            bool(web_context),
        )
        return prompt

    except Exception as e:
        logger.error("Failed to build system prompt: %s", e)
        # Fallback to a safe minimal prompt
        return BASE_PERSONA
