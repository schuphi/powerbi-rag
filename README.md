# powerbi-rag

V1 of an internal retrieval and QA system I developed for querying Power BI reports with natural language.

## What it does

Extracts metadata from PBIX files, builds searchable artifacts, and retrieves relevant tables, measures, pages, and visuals using lexical, dense, or hybrid retrieval.

This repository reflects the first working version of that approach. A newer internal version calls the Tabular Editor 2 (TE2) API instead of relying primarily on PBIX/layout-derived extraction. That makes the metadata calls much more reliable, but also more costly operationally.

High reliability is tricky in this problem space because large Power BI reports often turn into very large JSON payloads in Power BI project files, which creates real tradeoffs around parsing robustness, latency, and cost.

## Tech Stack

- FastAPI backend
- BM25-style lexical retrieval
- ChromaDB for dense vector search
- Sentence-transformers or OpenAI embeddings
- Anthropic Claude for answer generation
- Gradio web interface
- Retrieval evaluation benchmarks and metrics

## Setup

```bash
pip install -e .[dev]
cp .env.example .env
# Add API keys only if you want LLM-powered answers

# Test with Adventure Works sample
python scripts/simple_test.py

# Run retrieval benchmark
python scripts/run_retrieval_evals.py --mode lexical

# Start web interface
python -m powerbi_rag.cli start-ui
```

## Usage

Upload a PBIX file and ask questions like:
- "What tables are in this report?"
- "How is Total Sales calculated?"
- "Show me all the relationships"

Retrieval modes:
- `lexical` for self-contained keyword/BM25-style search
- `dense` for vector search with ChromaDB
- `hybrid` to fuse lexical and dense rankings

## API

Start the API server:
```bash
python -m powerbi_rag.cli start-api
```

Key endpoints:
- `POST /ask` - Ask questions about your reports
- `POST /upload` - Upload PBIX files
- `GET /health` - System status

## Evaluation

The repo now includes a benchmarkable retrieval setup in [evals/adventure_works_retrieval.jsonl](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/evals/adventure_works_retrieval.jsonl) with metrics such as `Recall@k`, `MRR`, `nDCG`, and hit rate.

```bash
python scripts/run_retrieval_evals.py --mode lexical
python scripts/run_retrieval_evals.py --mode hybrid
```

## Project Structure

```
src/powerbi_rag/   # Core extraction, retrieval, API, and UI code
scripts/           # Smoke tests, demos, and evaluation runners
tests/             # Test suite
evals/             # Benchmark inputs
results/evals/     # Generated evaluation outputs
docs/              # Short architecture notes
```

## Development

```bash
pytest
python -m powerbi_rag.cli --help
```

## Notes

- Extraction and API health checks work without API keys
- Anthropic is only required for answer generation
- Lexical retrieval and evaluation work without vector DB dependencies
- Adventure Works sample included for testing
- See [DEMO.md](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/DEMO.md) for verification steps
- See [architecture.md](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/docs/architecture.md) for a short system overview
