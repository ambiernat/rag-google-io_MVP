#!/usr/bin/env python
# coding: utf-8
"""
evaluate_sparse.py
Evaluate BM25 sparse retrieval using configuration from YAML.
"""
import yaml
import json
from pathlib import Path
from tqdm import tqdm
from retrieval.retrievers.retrieve_sparse import retrieve_sparse
from datetime import datetime

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
CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_DIR = PROJECT_ROOT / config["evaluation"]["ground_truth_dir"]
GT_PREFIX = config["evaluation"]["ground_truth_prefix"]
GROUND_TRUTH_PATH = get_latest_ground_truth(GT_DIR, GT_PREFIX)

timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = (
    PROJECT_ROOT
    / config["evaluation"]["output_dir"]
    / f"sparse_eval_{timestamp}.json"
)
TOP_K = config["evaluation"].get("top_k", 5)

print(f"[INFO] Using ground truth file: {GROUND_TRUTH_PATH.name}")

# -----------------------------
# Load ground truth
# -----------------------------
with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

# -----------------------------
# Metrics
# -----------------------------
def recall_at_k(retrieved_ids, relevant_ids, k=TOP_K):
    return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))

def mrr(retrieved_ids, relevant_ids):
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0

def precision_at_k(retrieved_ids, relevant_ids, k=TOP_K):
    relevant_count = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return relevant_count / k if k > 0 else 0.0

# -----------------------------
# Evaluation loop
# -----------------------------
results_all = []
print(f"[INFO] Evaluating SPARSE retriever for {len(ground_truth)} queries...")

for item in tqdm(ground_truth, desc="Queries"):
    query = item["query"]
    relevant_ids = item.get("relevant_doc_ids", [])

    # Retrieve top-K using Qdrant BM25
    points = retrieve_sparse(query, top_k=TOP_K)

    # Extract doc IDs from ScoredPoint payloads
    retrieved_ids = [point.payload.get("doc_id", str(point.id)) for point in points]

    results_all.append({
        "query": query,
        "relevant_doc_ids": relevant_ids,
        "retrieved_doc_ids": retrieved_ids,
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=TOP_K),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k=TOP_K)
    })

# -----------------------------
# Save results
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results_all, f, indent=2)

print(f"[OK] Sparse evaluation complete. Results saved to {OUTPUT_PATH}")



# #!/usr/bin/env python
# # coding: utf-8

# """
# evaluate_sparse.py
# Evaluate BM25 sparse retrieval using configuration from YAML.
# """

# import yaml
# import json
# from pathlib import Path
# from tqdm import tqdm
# from retrieval.retrievers.retrieve_sparse import retrieve_sparse  # BM25 retriever
# from datetime import datetime

# # -----------------------------
# # Helpers
# # -----------------------------
# def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
#     files = sorted(
#         gt_dir.glob(f"{prefix}_*.json"),
#         key=lambda p: p.stat().st_mtime,
#         reverse=True
#     )
#     if not files:
#         raise FileNotFoundError(f"No ground truth files found in {gt_dir}")
#     return files[0]


# # -----------------------------
# # Load config
# # -----------------------------

# CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)

# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# GT_DIR = PROJECT_ROOT / config["evaluation"]["ground_truth_dir"]
# GT_PREFIX = config["evaluation"]["ground_truth_prefix"]

# GROUND_TRUTH_PATH = get_latest_ground_truth(GT_DIR, GT_PREFIX)

# timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
# OUTPUT_PATH = (
#     PROJECT_ROOT
#     / config["evaluation"]["output_dir"]
#     / f"sparse_eval_{timestamp}.json"
# )


# TOP_K = config["evaluation"].get("top_k", 5)


# print(f"[INFO] Using ground truth file: {GROUND_TRUTH_PATH.name}")


# # -----------------------------
# # Load ground truth
# # -----------------------------
# with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
#     ground_truth = json.load(f)

# # -----------------------------
# # Metrics
# # -----------------------------
# def recall_at_k(retrieved_ids, relevant_ids, k=TOP_K):
#     return int(any(rid in relevant_ids for rid in retrieved_ids[:k]))

# def mrr(retrieved_ids, relevant_ids):
#     for i, rid in enumerate(retrieved_ids, start=1):
#         if rid in relevant_ids:
#             return 1.0 / i
#     return 0.0

# def precision_at_k(retrieved_ids, relevant_ids, k=TOP_K):
#     relevant_count = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
#     return relevant_count / k if k > 0 else 0.0

# # -----------------------------
# # Initialize retriever
# # -----------------------------
# retriever = SparseRetriever()

# # -----------------------------
# # Evaluation loop
# # -----------------------------
# results_all = []

# print(f"[INFO] Evaluating SPARSE retriever for {len(ground_truth)} queries...")
# for item in tqdm(ground_truth, desc="Queries"):
#     query = item["query"]
#     relevant_ids = item.get("relevant_doc_ids", [])

#     # Retrieve top-K from BM25
#     hits = retriever.retrieve(query, top_k=TOP_K)

#     # Extract retrieved doc IDs (fallback to index if doc_id missing)
    
#     retrieved_ids = [doc["id"] for _, doc in hits]
#     assert all("id" in doc for _, doc in hits), "Retrieved document missing 'id' field"


#     # Record metrics
#     results_all.append({
#         "query": query,
#         "relevant_doc_ids": relevant_ids,
#         "retrieved_doc_ids": retrieved_ids,
#         "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=TOP_K),
#         "mrr": mrr(retrieved_ids, relevant_ids),
#         "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k=TOP_K)
#     })

# # -----------------------------
# # Save results
# # -----------------------------
# OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
#     json.dump(results_all, f, indent=2)

# print(f"[OK] Sparse evaluation complete. Results saved to {OUTPUT_PATH}")
