# 🔬 ARIA — AI Research Intelligence Assistant

> Built for NeoStats AI Engineer Case Study

## What is ARIA?

ARIA is an intelligent research assistant that lets you:
- **Ask questions about any uploaded research paper** (RAG over PDFs)
- **Get real-time web search results** on any AI/research topic
- **Switch between Concise and Detailed** response modes instantly

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq · Llama 3.3 70B Versatile |
| Embeddings | HuggingFace · all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| Web Search | Tavily API |
| Framework | LangChain + Streamlit |

## Quick Start

```bash
pip install -r requirements.txt
GROQ_API_KEY=your_key TAVILY_API_KEY=your_key streamlit run app.py
```

## Project Structure

```
project/
├── config/config.py        ← API keys & settings
├── models/llm.py           ← Groq model
├── models/embeddings.py    ← HuggingFace embeddings
├── utils/rag.py            ← RAG pipeline
├── utils/web_search.py     ← Tavily search
├── utils/prompt_builder.py ← System prompt builder
├── app.py                  ← Streamlit UI
└── requirements.txt
```
