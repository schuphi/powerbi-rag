# Architecture

This repository is the first version of an internal Power BI retrieval and QA workflow.

## Flow

1. Extraction
- [pbix_extractor.py](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/src/powerbi_rag/extraction/pbix_extractor.py) reads PBIX files and derives tables, columns, measures, pages, and visuals.
- For many PBIX files, extraction is layout-derived rather than a full semantic-model parse.

2. Indexing
- Extracted report elements are converted into `PowerBIArtifact` records.
- These artifacts can be indexed for lexical retrieval, dense retrieval, or both.

3. Retrieval
- [lexical_store.py](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/src/powerbi_rag/retrieval/lexical_store.py) provides BM25-style lexical search.
- [vector_store.py](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/src/powerbi_rag/retrieval/vector_store.py) provides dense retrieval through ChromaDB.
- [hybrid_retriever.py](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/src/powerbi_rag/retrieval/hybrid_retriever.py) combines lexical and dense rankings.

4. Generation
- [rag_pipeline.py](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/src/powerbi_rag/retrieval/rag_pipeline.py) retrieves context and optionally generates answers with an LLM.

5. Evaluation
- [adventure_works_retrieval.jsonl](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/evals/adventure_works_retrieval.jsonl) contains the retrieval benchmark.
- [run_retrieval_evals.py](/Users/philipp/Desktop/Github/pbi-rag/powerbi-rag/scripts/run_retrieval_evals.py) computes `Recall@k`, `MRR`, `nDCG`, and hit rate.

## Limitations

- Extraction is not yet a full-fidelity parser for arbitrary PBIX semantic models.
- Dense retrieval depends on optional vector and embedding dependencies.
- Answer generation currently depends on Anthropic.
