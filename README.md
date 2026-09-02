# RAG Knowledge Assistant

A bilingual Retrieval-Augmented Generation (RAG) service for answering questions from internal English and Chinese PDF documents. It supports scanned PDFs, source citations, multi-turn sessions, relevance filtering, reranking, security checks and structured logging.

## Features

- English and Chinese question answering
- Text extraction and OCR for scanned PDFs
- Multilingual semantic search with Chroma
- Configurable multilingual reranking
- Answers grounded only in retrieved document context
- File name and page number citations
- Multi-turn conversation sessions
- Refusal for unrelated or low-relevance questions
- English and Chinese prompt-injection detection
- Basic PII redaction in structured JSONL logs
- FastAPI endpoints and Swagger documentation
- Evaluation, sensitivity and cost analysis scripts

## Architecture

```text
PDF documents
    ↓
Text extraction / OCR
    ↓
Chunking and multilingual embeddings
    ↓
Chroma vector store
    ↓
Relevance filtering and reranking
    ↓
Context construction
    ↓
Groq GPT-OSS 120B
    ↓
FastAPI response with citations
```

## Project Structure

```text
src/
├── api/              FastAPI endpoints and schemas
├── generation/       Context, sessions and LLM generation
├── ingestion/        PDF loading, OCR and text splitting
├── observability/    Structured logging and request tracing
├── retrieval/        Embeddings, vector search and reranking
└── security/         Prompt-injection checks and PII redaction

evaluation/           Evaluation and cost analysis
tests/                Unit tests
documents/            Internal knowledge base documents
```

## Installation

Python 3.11 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Add the Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
USE_RERANKER=true
DEFAULT_TOP_K=2
RERANK_CANDIDATE_K=6
GENERATION_TEMPERATURE=0.2
```

`HF_TOKEN` is optional but can improve Hugging Face download limits.

## Build the Vector Store

Place the supplied PDF documents in `documents/`, then run:

```powershell
python -m src.retrieval.vector_store
```

The persistent Chroma data is stored locally and is excluded from Git.

## Run the API

```powershell
uvicorn src.api.main:app --reload
```

Open the Swagger interface:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check service and vector-store status |
| POST | `/ask` | Ask a question or continue a session |
| DELETE | `/sessions/{session_id}` | Delete stored session history |

Example request body:

```json
{
  "question": "What is the API rate limit?",
  "session_id": null
}
```

## Tests

Run the unit tests:

```powershell
python -m pytest -v
```

The current test suite contains 22 tests covering:

- Prompt-injection detection
- PII redaction
- Relevance filtering
- Session creation, history limits and deletion
- API request and response schemas

## Evaluation

Run the full 22-turn evaluation:

```powershell
python -m evaluation.run_evaluation
python -m evaluation.metrics
```

Run the three-setting sensitivity analysis:

```powershell
python -m evaluation.sensitivity_analysis
python -m evaluation.sensitivity_report
```

Calculate the estimated LLM cost:

```powershell
python -m evaluation.cost_analysis
```

The full and sensitivity evaluations call the Groq API. Metric and cost scripts only read saved local results.

## Evaluation Results

| Metric | Result | Target |
|---|---:|---:|
| Accuracy | 81.82% | ≥80% |
| Faithfulness | 100.00% | ≥85% |
| Context Precision | 79.55% | ≥70% |
| Requests under 10 seconds | 100.00% | ≥90% |
| P90 latency | 5.80 seconds | <10 seconds |

The estimated production LLM cost is approximately **$0.1335 per 1,000 calls** using published Groq token prices. Development and evaluation used the Groq free tier, so no API charge was incurred.

See [evaluation/evaluation_summary.md](evaluation/evaluation_summary.md) for the scoring rubric, sensitivity results, failure analysis, cost calculation and redacted log examples.

## Logging and Security

Structured logs are written to:

```text
logs/rag_service.jsonl
```

Logs include request IDs, latency, source files, outcome, reranker status and token usage. Email addresses, telephone numbers, National Insurance numbers, payment-card numbers and secret-like fields are redacted before logging.

The service rejects obvious English and Chinese prompt-injection attempts before retrieval or LLM generation.

## Current Limitations

- Conversation sessions are stored in memory and are lost after restart.
- Retrieval models run locally on CPU.
- OCR and the first model load can take additional time.
- Some cross-language and architecture questions still require better retrieval recall.
- Production deployment would require persistent session storage, authentication, access control and monitoring.