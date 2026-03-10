#!/usr/bin/env python
# coding: utf-8
"""
evaluate_hybrid_reranked.py
Evaluate hybrid retrieval + CrossEncoder reranking in one pass.
Retrieve → Rehydrate → Rerank → Evaluate
"""
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timezone
from sentence_transformers import CrossEncoder
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid

# -----------------------------
# Helpers
# -----------------------------
def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No ground truth files found in {gt_dir}")
    return files[0]

# -----------------------------
# Load config
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GT_DIR = PROJECT_ROOT / config["evaluation"]["ground_truth_dir"]
GT_PREFIX = config["evaluation"]["ground_truth_prefix"]
RETRIEVAL_TOP_K = 20  # retrieve more candidates for reranker to work with
RERANK_TOP_K = config["evaluation"].get("top_k", 5)  # final top-K after reranking

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROUND_TRUTH_PATH = get_latest_ground_truth(GT_DIR, GT_PREFIX)

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["output_dir"]
    / f"hybrid_reranked_eval_{timestamp}.json"
)

print(f"[INFO] Ground truth: {GROUND_TRUTH_PATH.name}")
print(f"[INFO] Retrieving top {RETRIEVAL_TOP_K}, reranking to top {RERANK_TOP_K}")
print(f"[INFO] Reranker: {RERANKER_MODEL}")
print(f"[INFO] Output: {OUTPUT_PATH}")

# -----------------------------
# Load ground truth
# -----------------------------
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# -----------------------------
# Load canonical docs for rehydration
# -----------------------------
CANONICAL_PATH = PROJECT_ROOT / "data" / "canonical" / "all_documents.json"
with open(CANONICAL_PATH) as f:
    canonical_docs = json.load(f)
doc_index = {doc["id"]: doc for doc in canonical_docs}
print(f"[INFO] Loaded {len(doc_index)} canonical docs for rehydration")

# -----------------------------
# Load reranker
# -----------------------------
print(f"[INFO] Loading CrossEncoder...")
reranker = CrossEncoder(RERANKER_MODEL, device="cpu")
print(f"[INFO] CrossEncoder loaded")

# -----------------------------
# Metrics
# -----------------------------
def recall_at_k(retrieved_ids, relevant_ids, k):
    return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))

def mrr(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved_ids, relevant_ids, k):
    relevant_count = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return relevant_count / k if k > 0 else 0.0

# -----------------------------
# Evaluation loop
# -----------------------------
results_all = []
print(f"[INFO] Evaluating HYBRID + RERANKER for {len(ground_truth)} queries...")

for item in tqdm(ground_truth, desc="Queries"):
    query = item["query"]
    relevant_ids = item.get("relevant_doc_ids", [])

    # Step 1: Retrieve
    hits = retrieve_hybrid(query, top_k=RETRIEVAL_TOP_K)
    retrieved_ids = [hit.payload.get("doc_id", str(hit.id)) for hit in hits]

    # Step 2: Rehydrate — get full text for each retrieved doc
    docs = [doc_index[doc_id] for doc_id in retrieved_ids if doc_id in doc_index]

    # Step 3: Rerank
    if docs:
        pairs = [[query, doc["text"]] for doc in docs]
        scores = reranker.predict(pairs)
        sorted_idx = np.argsort(scores)[::-1][:RERANK_TOP_K]
        reranked_ids = [docs[i]["id"] for i in sorted_idx]
    else:
        reranked_ids = retrieved_ids[:RERANK_TOP_K]

    # Step 4: Evaluate
    results_all.append({
        "query": query,
        "relevant_doc_ids": relevant_ids,
        "retrieved_doc_ids": retrieved_ids,      # pre-rerank
        "reranked_doc_ids": reranked_ids,         # post-rerank
        "recall_at_k": recall_at_k(reranked_ids, relevant_ids, k=RERANK_TOP_K),
        "mrr": mrr(reranked_ids, relevant_ids),
        "precision_at_k": precision_at_k(reranked_ids, relevant_ids, k=RERANK_TOP_K),
    })

# -----------------------------
# Summary
# -----------------------------
n = len(results_all)
avg_recall = sum(r["recall_at_k"] for r in results_all) / n
avg_mrr = sum(r["mrr"] for r in results_all) / n
avg_precision = sum(r["precision_at_k"] for r in results_all) / n

print(f"\n[RESULTS] Hybrid + Reranker ({n} queries)")
print(f"  Recall@{RERANK_TOP_K}:    {avg_recall:.4f}")
print(f"  MRR:         {avg_mrr:.4f}")
print(f"  Precision@{RERANK_TOP_K}: {avg_precision:.4f}")

# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2)

print(f"\n[OK] Saved to {OUTPUT_PATH}")