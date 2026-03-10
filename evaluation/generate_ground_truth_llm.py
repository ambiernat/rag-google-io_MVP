#!/usr/bin/env python
# coding: utf-8
"""
generate_ground_truth.py
Generate ground truth queries for retrieval evaluation.
Samples documents across all videos for broad coverage.
"""
from openai import OpenAI
import json
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import getpass

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "gpt-5-nano"  # cheaper, good enough for this task
QUESTIONS_PER_DOC = 2
DOCS_PER_VIDEO = 2  # sample 2 chunks per video → ~156 docs → ~312 queries
RANDOM_SEED = 42

timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
OUTPUT_PATH = Path("data/eval/ground_truth") / f"ground_truth_{MODEL_NAME}_{timestamp}.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load documents
# -----------------------------
with open("data/canonical/all_documents.json") as f:
    documents = json.load(f)

print(f"[INFO] Loaded {len(documents)} documents from {len(set(d['video_id'] for d in documents))} videos")

# -----------------------------
# Sample across all videos
# -----------------------------
random.seed(RANDOM_SEED)

# Group by video
from collections import defaultdict
by_video = defaultdict(list)
for doc in documents:
    by_video[doc["video_id"]].append(doc)

# Sample N chunks per video
sampled_docs = []
for video_id, docs in by_video.items():
    sampled_docs.extend(random.sample(docs, min(DOCS_PER_VIDEO, len(docs))))

print(f"[INFO] Sampled {len(sampled_docs)} documents across {len(by_video)} videos")

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
# Prompt
# -----------------------------
def build_prompt(text: str, n_questions: int) -> str:
    return f"""You are generating user search questions for a retrieval evaluation dataset.
The user has NOT seen the text below. They are searching for information contained in it.

Generate {n_questions} DISTINCT, realistic user questions that could retrieve this text.
- Questions should vary in wording and intent
- Do NOT quote the text
- Do NOT include answers
- Do NOT number the questions
- Each question must be on a separate line

TEXT:
{text}
"""

def generate_questions(text: str, n_questions: int = QUESTIONS_PER_DOC) -> list[str]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": build_prompt(text, n_questions)}]
    )
    raw = response.choices[0].message.content.strip()
    return [q.strip("- ").strip() for q in raw.split("\n") if q.strip()]

# -----------------------------
# Generate ground truth
# -----------------------------
ground_truth = []
errors = 0

for doc in tqdm(sampled_docs, desc="Generating queries"): #sampled_docs[:5] # start small,scale later
    try:
        questions = generate_questions(doc["text"], QUESTIONS_PER_DOC)
        for q in questions:
            ground_truth.append({
                "query": q,
                "relevant_doc_ids": [doc["id"]],
                "video_id": doc["video_id"],
                "title": doc.get("title", ""),
            })
    except Exception as e:
        print(f"[ERROR] {doc['id']}: {e}")
        errors += 1

print(f"\n[INFO] Generated {len(ground_truth)} queries ({errors} errors)")
print(f"[INFO] Covers {len(set(q['video_id'] for q in ground_truth))} videos")

# -----------------------------
# Save
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(ground_truth, f, indent=2)

print(f"[OK] Saved to {OUTPUT_PATH}")