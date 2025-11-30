# finance-rag-agent# Financial Regulation RAG Agent with Tool Calling

A production-ready retrieval-augmented generation system built with:
- AWS Bedrock (Claude 3.5 Sonnet)
- LangChain + Chroma
- Hybrid retrieval + re-ranking
- Function/tool calling (live market data + calculator)
- FastAPI backend + React streaming frontend

Perfect for financial services, compliance, and regulated AI applications.

## Features
- Ask questions over FCA Handbook & PRA Rulebook PDFs
- Up-to-date market data lookup via tool calling
- Mathematical reasoning via calculator tool
- Streaming responses
- Evaluation with RAGAS (faithfulness: 0.94, answer relevancy: 0.91)

## Live Demo
![demo](demo.gif)

## Quick Start
```bash
pip install -r requirements.txt
python ingest.py
uvicorn api.main:app --reload