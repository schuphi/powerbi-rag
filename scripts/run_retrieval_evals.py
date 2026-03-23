#!/usr/bin/env python3
"""Run retrieval evaluations against the Adventure Works sample."""

import argparse
import json
import math
import sys
import tempfile
from datetime import datetime, UTC
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from powerbi_rag.extraction.pbix_extractor import PBIXExtractor
from powerbi_rag.retrieval.hybrid_retriever import HybridRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evaluations for powerbi-rag.")
    parser.add_argument(
        "--pbix",
        default="data/raw/Adventure Works Sales Sample.pbix",
        help="PBIX file to evaluate against.",
    )
    parser.add_argument(
        "--benchmark",
        default="evals/adventure_works_retrieval.jsonl",
        help="Benchmark JSONL file.",
    )
    parser.add_argument(
        "--mode",
        choices=["lexical", "dense", "hybrid"],
        default="hybrid",
        help="Retrieval mode to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_retriever(artifacts, mode: str):
    retriever = None
    vector_available = False

    if mode in {"dense", "hybrid"}:
        try:
            from powerbi_rag.retrieval.vector_store import ChromaVectorStore

            tempdir = tempfile.TemporaryDirectory()
            vector_store = ChromaVectorStore(
                persist_directory=tempdir.name,
                collection_name="eval_collection",
                embedding_function="sentence_transformers",
            )
            vector_store.reset_collection()
            vector_store.add_artifacts(artifacts)
            retriever = HybridRetriever(vector_store=vector_store, retrieval_mode=mode)
            retriever.index_artifacts(artifacts)
            vector_available = True
            return retriever, vector_available, tempdir
        except Exception as exc:
            print(f"[warn] Dense retrieval unavailable, falling back to lexical only: {exc}")

    retriever = HybridRetriever(vector_store=None, retrieval_mode="lexical")
    retriever.index_artifacts(artifacts)
    return retriever, vector_available, None


def evaluate(retriever, benchmark: list[dict], top_k: int) -> dict:
    per_query = []
    recall_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    hit_count = 0

    for entry in benchmark:
        results = retriever.search(
            entry["question"],
            n_results=top_k,
            artifact_type=entry.get("filter_by_type"),
        )
        expected_names = {name for name in entry.get("expected_names", [])}
        retrieved_names = [
            result.get("metadata", {}).get("name", "")
            for result in results
        ]
        matched_names = set()
        relevant_ranks = []
        for rank, name in enumerate(retrieved_names, start=1):
            if name in expected_names and name not in matched_names:
                matched_names.add(name)
                relevant_ranks.append(rank)

        recall = len(relevant_ranks) / len(expected_names) if expected_names else 0.0
        mrr = 1 / relevant_ranks[0] if relevant_ranks else 0.0
        ndcg = compute_ndcg(relevant_ranks, len(expected_names), top_k)
        hit = bool(relevant_ranks)

        recall_sum += recall
        mrr_sum += mrr
        ndcg_sum += ndcg
        hit_count += int(hit)

        per_query.append(
            {
                "question": entry["question"],
                "filter_by_type": entry.get("filter_by_type"),
                "expected_names": sorted(expected_names),
                "retrieved_names": retrieved_names,
                "recall_at_k": round(recall, 4),
                "mrr": round(mrr, 4),
                "ndcg_at_k": round(ndcg, 4),
                "hit": hit,
            }
        )

    total = len(benchmark) or 1
    return {
        "summary": {
            "queries": len(benchmark),
            "recall_at_k": round(recall_sum / total, 4),
            "mrr": round(mrr_sum / total, 4),
            "ndcg_at_k": round(ndcg_sum / total, 4),
            "hit_rate": round(hit_count / total, 4),
        },
        "per_query": per_query,
    }


def compute_ndcg(relevant_ranks: list[int], relevant_count: int, top_k: int) -> float:
    if not relevant_ranks or relevant_count == 0:
        return 0.0

    dcg = sum(1 / math.log2(rank + 1) for rank in relevant_ranks if rank <= top_k)
    ideal_count = min(relevant_count, top_k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def save_results(mode: str, payload: dict) -> Path:
    results_dir = Path("results/evals")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = results_dir / f"retrieval_eval_{mode}_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    pbix_path = Path(args.pbix)
    benchmark_path = Path(args.benchmark)

    if not pbix_path.exists():
        print(f"[error] PBIX file not found: {pbix_path}")
        return 1
    if not benchmark_path.exists():
        print(f"[error] Benchmark file not found: {benchmark_path}")
        return 1

    extractor = PBIXExtractor()
    report = extractor.extract_report(pbix_path)
    artifacts = extractor.extract_artifacts(report)
    benchmark = load_benchmark(benchmark_path)

    retriever, vector_available, tempdir = build_retriever(artifacts, args.mode)
    results = evaluate(retriever, benchmark, args.top_k)
    results["config"] = {
        "requested_mode": args.mode,
        "vector_available": vector_available,
        "effective_mode": retriever.retrieval_mode,
        "top_k": args.top_k,
        "artifact_count": len(artifacts),
        "pbix": str(pbix_path),
        "benchmark": str(benchmark_path),
    }

    output_path = save_results(args.mode, results)
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved results to {output_path}")

    if tempdir is not None:
        tempdir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
