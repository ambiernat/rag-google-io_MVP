#!/usr/bin/env python
# coding: utf-8

"""
CrossEncoder Reranker Script (Canonical Rehydration)

Reranks retrieval results using a CrossEncoder model.
Uses canonical documents to rehydrate retrieved_doc_ids.

Run:
    python -m retrieval.rerankers.crossencoder_reranker
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from sentence_transformers import CrossEncoder


# -------------------------------------------------
# Paths & Config
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "retrieval" / "config.yaml"
EVAL_RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"
CANONICAL_DOCS_PATH = PROJECT_ROOT / "data" / "canonical" / "all_documents.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

if not CANONICAL_DOCS_PATH.exists():
    raise FileNotFoundError(f"Canonical docs not found: {CANONICAL_DOCS_PATH}")


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_latest_eval_file(input_type: str) -> Path:
    """
    Example filename:
        hybrid_eval_20260125T151944.json
    """
    pattern = f"{input_type}_eval_*.json"
    files = sorted(EVAL_RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No eval files found for pattern: {pattern}")
    return files[-1]


def extract_text(doc: Dict) -> str:
    if "payload" in doc and "text" in doc["payload"]:
        return doc["payload"]["text"]
    if "text" in doc:
        return doc["text"]
    return ""

# -------------------------------------------------
# Programmatic Reranker Interface - Helper/for HPO
# -------------------------------------------------

# Add at the top-level, outside main():

_model_cache = {}

def crossencoder_rerank(
    query: str,
    documents: list,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 10,
    device: str = "cpu",
    batch_size: int = 16,
) -> list:
    if not documents:
        return []

    # Load model once and cache
    if model_name not in _model_cache:
        print(f"[INFO] Loading CrossEncoder: {model_name}") 
        _model_cache[model_name] = CrossEncoder(model_name, device=device)
    model = _model_cache[model_name]

    texts = [doc.get("text", str(doc)) for doc in documents]
    pairs = [[query, t] for t in texts]
    scores = model.predict(pairs, batch_size=batch_size)
    sorted_idx = np.argsort(scores)[::-1][:top_k]

    reranked = []
    for idx in sorted_idx:
        doc = documents[idx].copy()
        doc["rerank_score"] = float(scores[idx])
        reranked.append(doc)

    return reranked


# -------------------------------------------------
# Reranking
# -------------------------------------------------
def rerank_with_crossencoder(
    model: CrossEncoder,
    query: str,
    documents: List[Dict],
    top_k: int,
) -> List[Dict]:
    if not documents:
        return []

    texts = [extract_text(doc) for doc in documents]
    pairs = [[query, text] for text in texts]

    scores = model.predict(pairs)
    sorted_idx = np.argsort(scores)[::-1][:top_k]

    reranked = []
    for idx in sorted_idx:
        doc = documents[idx].copy()
        doc["rerank_score"] = float(scores[idx])
        reranked.append(doc)

    return reranked


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    config = load_config()

    reranker_cfg = config["rerankers"]["cross_encoder"]
    input_type = config["rerankers"]["input"]["type"]
    log_level = config.get("logging", {}).get("level", "INFO")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    if not reranker_cfg.get("enabled", False):
        logger.info("CrossEncoder reranker disabled. Exiting.")
        return

    logger.info(f"Reranker input type: {input_type}")
    logger.info(f"CrossEncoder model: {reranker_cfg['model_name']}")
    logger.info(f"Top-K after rerank: {reranker_cfg['top_k']}")

    # -------------------------------------------------
    # Load latest eval file
    # -------------------------------------------------
    eval_file = get_latest_eval_file(input_type)
    logger.info(f"Using latest eval file: {eval_file.name}")
    eval_data = load_json(eval_file)

    # -------------------------------------------------
    # Load canonical documents
    # -------------------------------------------------
    logger.info("Loading canonical documents...")
    canonical_docs = load_json(CANONICAL_DOCS_PATH)
    doc_index = {doc["id"]: doc for doc in canonical_docs}

    # -------------------------------------------------
    # Load CrossEncoder once
    # -------------------------------------------------
    logger.info("Loading CrossEncoder model...")
    model = CrossEncoder(reranker_cfg["model_name"])

    # -------------------------------------------------
    # Rerank
    # -------------------------------------------------
    results = []

    for item in eval_data:
        query = item["query"]
        retrieved_ids = item.get("retrieved_doc_ids", [])

        documents = [
            doc_index[doc_id]
            for doc_id in retrieved_ids
            if doc_id in doc_index
        ]

        reranked_docs = rerank_with_crossencoder(
            model=model,
            query=query,
            documents=documents,
            top_k=reranker_cfg["top_k"],
        )

        results.append(
            {
                "query": query,
                "reranked_docs": reranked_docs,
            }
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    output_path = PROJECT_ROOT / config["files"]["output_file"]
    save_json(results, output_path)

    logger.info(f"Reranked results saved to: {output_path}")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    main()
