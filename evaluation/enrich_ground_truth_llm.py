#!/usr/bin/env python
# coding: utf-8
"""
enrich_ground_truth.py
Enriches ground truth by using GPT as a judge to label
multi-doc relevance for each query.
"""
from openai import OpenAI
import json
import yaml
import getpass
from pathlib import Path
from datetime import datetime, UTC
from tqdm import tqdm
import sys
import os

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_DIR = PROJECT_ROOT / config["enrich_gt"]["ground_truth_dir"]
GT_PREFIX = config["enrich_gt"]["ground_truth_prefix"]
TOP_K = config["enrich_gt"].get("top_k", 5)

MODEL_NAME = "gpt-5-nano"
timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = GT_DIR / f"ground_truth_{MODEL_NAME}_{timestamp}.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Find latest ground truth
# -----------------------------
def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No ground truth files found in {gt_dir} with prefix '{prefix}'")
    return files[0]

INPUT_PATH = get_latest_ground_truth(GT_DIR, GT_PREFIX)

# -----------------------------
# OpenAI client
# -----------------------------
try:
    API_KEY = getpass.getpass("Enter OpenAI API key: ")
except Exception as e:
    print(f"ERROR: {e}")
    raise

client = OpenAI(api_key=API_KEY)

# -----------------------------
# Retriever
# -----------------------------
os.environ["QDRANT_URL"] = os.getenv("QDRANT_URL", "http://localhost:6333")
sys.path.insert(0, str(PROJECT_ROOT))
from retrieval.retrievers.retrieve_dense import retrieve_dense

# -----------------------------
# Judge function
# -----------------------------
def judge_relevance(query: str, chunk_text: str) -> bool:
    prompt = f"""You are a relevance judge for a retrieval system.

Given a user query and a text chunk from a Google I/O 2025 talk transcript, 
decide if the chunk contains information that is relevant to answering the query.

Respond with only "YES" or "NO".

Query: {query}

Chunk:
{chunk_text[:1000]}

Is this chunk relevant to the query?"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")

# -----------------------------
# Load ground truth
# -----------------------------
with open(INPUT_PATH) as f:
    ground_truth = json.load(f)

print(f"[INFO] Loaded {len(ground_truth)} queries from {INPUT_PATH.name}")
print(f"[INFO] Retrieving top {TOP_K} candidates per query and judging relevance...")

# -----------------------------
# Enrich ground truth
# -----------------------------
enriched = []
errors = 0

for item in tqdm(ground_truth, desc="Judging"):
    query = item["query"]
    original_relevant = item["relevant_doc_ids"]

    try:
        # Retrieve top-K candidates
        points = retrieve_dense(query, top_k=TOP_K)

        # Judge each candidate
        judged_relevant = set(original_relevant)  # always keep original
        for point in points:
            doc_id = point.payload.get("doc_id", str(point.id))
            chunk_text = point.payload.get("text", "")

            if doc_id in judged_relevant:
                continue  # already marked relevant

            try:
                if judge_relevance(query, chunk_text):
                    judged_relevant.add(doc_id)
            except Exception as e:
                print(f"[ERROR] Judge failed for {doc_id}: {e}")
                errors += 1

        enriched.append({
            **item,
            "relevant_doc_ids": list(judged_relevant),
            "original_relevant_doc_ids": original_relevant,
            "num_relevant": len(judged_relevant),
        })

    except Exception as e:
        print(f"[ERROR] Query failed: {query[:50]}...: {e}")
        enriched.append(item)
        errors += 1

# -----------------------------
# Stats
# -----------------------------
avg_relevant = sum(e["num_relevant"] for e in enriched if "num_relevant" in e) / len(enriched)
print(f"\n[INFO] Enriched {len(enriched)} queries ({errors} errors)")
print(f"[INFO] Average relevant docs per query: {avg_relevant:.2f} (was 1.00)")

# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(enriched, f, indent=2)

print(f"[OK] Saved to {OUTPUT_PATH}")