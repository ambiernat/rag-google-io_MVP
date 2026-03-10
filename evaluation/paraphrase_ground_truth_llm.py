#!/usr/bin/env python
# coding: utf-8
"""
paraphrase_ground_truth.py
Takes the latest ground truth file and paraphrases queries
to reduce vocabulary overlap with source chunks.
"""
from openai import OpenAI
import json
import yaml
import getpass
from pathlib import Path
from datetime import datetime, UTC
from tqdm import tqdm

# -----------------------------
# Load config
# -----------------------------
CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GT_DIR = PROJECT_ROOT / config["paraphrase_gt"]["ground_truth_dir"]
GT_PREFIX = config["paraphrase_gt"]["ground_truth_prefix"]

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
# Paraphrase function
# -----------------------------
def paraphrase_query(query: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"""Rewrite this search query as a casual user who has never seen the source material.
- Use different words and phrasing
- Be more conversational or colloquial
- Avoid technical jargon or specific product names where possible
- Keep the same intent and meaning
- Return only the rewritten query, nothing else

Original: {query}
Rewritten:"""}]
    )
    return response.choices[0].message.content.strip()

# -----------------------------
# Load ground truth
# -----------------------------
with open(INPUT_PATH) as f:
    ground_truth = json.load(f)

print(f"[INFO] Loaded {len(ground_truth)} queries from {INPUT_PATH.name}")

# -----------------------------
# Paraphrase
# -----------------------------
paraphrased = []
errors = 0

for item in tqdm(ground_truth, desc="Paraphrasing"):
    try:
        new_query = paraphrase_query(item["query"])
        paraphrased.append({
            **item,
            "query": new_query,
            "original_query": item["query"],
        })
    except Exception as e:
        print(f"[ERROR] {item['query'][:50]}...: {e}")
        paraphrased.append(item)  # keep original on error
        errors += 1

print(f"\n[INFO] Paraphrased {len(paraphrased)} queries ({errors} errors)")

# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(paraphrased, f, indent=2)

print(f"[OK] Saved to {OUTPUT_PATH}")

# -----------------------------
# Preview a few
# -----------------------------
print("\n--- Sample paraphrases ---")
for item in paraphrased[:3]:
    print(f"Original:    {item['original_query']}")
    print(f"Paraphrased: {item['query']}")
    print()