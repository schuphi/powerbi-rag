# Demo

Minimal verification steps for the repository.

## Quick test

```bash
python scripts/demo_working_features.py
```

Shows the current extraction, retrieval, and API surface.

## With sample data

```bash
python scripts/simple_test.py
```

Runs the extraction smoke test against the bundled sample.

## Retrieval evaluation

```bash
python scripts/run_retrieval_evals.py --mode lexical
```

## Test suite

```bash
pytest
```

## Dependencies

Core extraction and lexical retrieval work without API keys. Dense retrieval and answer generation need optional dependencies:
- Anthropic for answer generation
- ChromaDB for vector storage
- OpenAI or sentence-transformers for embeddings

See [architecture.md](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/docs/architecture.md) for a concise system overview.
