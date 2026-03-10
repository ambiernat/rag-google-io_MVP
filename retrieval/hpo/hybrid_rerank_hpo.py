#!/usr/bin/env python
# coding: utf-8
"""
dense_rerank_hpo.py

Hyperparameter Optimization (HPO) for Hybrid Retrieval + CrossEncoder Rerank
using Optuna and MLflow.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm

import optuna
import mlflow
import yaml
import random

from sentence_transformers import CrossEncoder

# -----------------------------
# Project Root
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag-google-io/
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------
# HPO Config
# -----------------------------
CONFIG_PATH = PROJECT_ROOT / "retrieval" / "hpo" / "config_hybrid_rerank.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=cfg.get("logging", {}).get("level", "INFO"))
logger = logging.getLogger(__name__)

# -----------------------------
# Imports
# -----------------------------
from retrieval.retrievers.retrieve_hybrid import retrieve_hybrid
from retrieval.rerankers.crossencoder_reranker import crossencoder_rerank

# -----------------------------
# Ground Truth
# -----------------------------
def get_latest_ground_truth(gt_dir: Path, prefix: str) -> Path:
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No ground truth files found in {gt_dir} with prefix {prefix}")
    return files[0]

gt_dir = PROJECT_ROOT / cfg["data"]["ground_truth"]["dir"]
gt_prefix = cfg["data"]["ground_truth"]["prefix"]
gt_file = get_latest_ground_truth(gt_dir, gt_prefix)

with open(gt_file, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

logger.info(f"Loaded {len(ground_truth)} ground truth queries from {gt_file.name}")


# Subsample for HPO speed
HPO_SAMPLE_SIZE = 50
random.seed(42)
ground_truth = random.sample(ground_truth, min(HPO_SAMPLE_SIZE, len(ground_truth)))
logger.info(f"Subsampled to {len(ground_truth)} queries for HPO")
# -----------------------------
# Hybrid wrapper
# -----------------------------
def hybrid_wrapper(query: str, top_k: int) -> list:
    hits = retrieve_hybrid(query, top_k=top_k)
    return [
        {
            "id": hit.payload.get("doc_id", str(hit.id)),  # ← use string doc_id
            "text": hit.payload.get("text", ""),
            **hit.payload
        }
        for hit in hits
    ]

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(retrieved_ids: list, relevant_ids: list, k: int) -> dict:
    recall = int(any(rid in relevant_ids for rid in retrieved_ids[:k]))
    mrr = 0.0
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            mrr = 1.0 / i
            break
    precision = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids) / k
    return {"recall@k": recall, "mrr": mrr, "precision@k": precision}

# -----------------------------
# Optuna Objective
# -----------------------------
_model_cache = {}

def objective(trial):
    retrieve_k = trial.suggest_categorical("retrieve_k", cfg["retrieval"]["retrieve_k"]["choices"])
    rerank_k = trial.suggest_categorical("rerank_k", cfg["reranker"]["rerank_k"]["choices"])
    rerank_model = trial.suggest_categorical("rerank_model", cfg["reranker"]["models"])

    primary_metric = cfg["objective"].get("primary_metric", "recall@k")
    cost_penalty_cfg = cfg["objective"].get("cost_penalty", {})
    cost_penalty_enabled = cost_penalty_cfg.get("enabled", False)
    retrieve_k_weight = cost_penalty_cfg.get("retrieve_k_weight", 0.1)

    metrics_list = []

    with mlflow.start_run(nested=True): 
        mlflow.log_params({
            "retrieve_k": retrieve_k,
            "rerank_k": rerank_k,
            "rerank_model": rerank_model
        })

        for item in tqdm(ground_truth, desc=f"Trial {trial.number}", leave=False):
            query = item["query"]
            relevant_ids = item["relevant_doc_ids"]

            retrieved_docs = hybrid_wrapper(query, retrieve_k)

            if rerank_model not in _model_cache:
                print(f"[INFO] Loading CrossEncoder: {rerank_model}")
                _model_cache[rerank_model] = CrossEncoder(rerank_model, device=cfg["compute"].get("device", "cpu"))
            model = _model_cache[rerank_model]

            if retrieved_docs:
                pairs = [[query, doc["text"]] for doc in retrieved_docs]
                scores = model.predict(pairs, batch_size=cfg["compute"].get("batch_size", 16))
                sorted_idx = np.argsort(scores)[::-1][:rerank_k]
                retrieved_ids = [retrieved_docs[i]["id"] for i in sorted_idx]
            else:
                retrieved_ids = []

            metrics_list.append(compute_metrics(retrieved_ids, relevant_ids, rerank_k))

        metric_map = {"recall@k": "recall_k", "mrr": "mrr", "precision@k": "precision_k"}
        avg_metrics = {
            metric_map[k]: float(np.mean([m.get(k, 0) for m in metrics_list]))
            for k in metric_map
        }

        for name, value in avg_metrics.items():
            mlflow.log_metric(name, value)

        objective_value = avg_metrics.get(
            metric_map.get(primary_metric, primary_metric), 0.0
        )
        if cost_penalty_enabled:
            objective_value -= (retrieve_k / 100) * retrieve_k_weight

    return objective_value

# -----------------------------
# MLflow & Optuna Setup
# -----------------------------
mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
mlflow.set_experiment(cfg["experiment"]["name"])
logger.info(f"MLflow experiment: {cfg['experiment']['name']}")

# -----------------------------
# Run HPO
# -----------------------------
n_trials = cfg["experiment"].get("n_trials", 20)
logger.info(f"Starting Optuna study for {n_trials} trials...")

with mlflow.start_run(run_name="hpo_study"):
    study = optuna.create_study(direction=cfg["experiment"].get("direction", "maximize"))
    study.optimize(objective, n_trials=n_trials)

# -----------------------------
# Save artifacts
# -----------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
output_dir = PROJECT_ROOT / cfg["data"]["ground_truth"]["results_dir"]
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"\nBest trial: {study.best_params}")
logger.info(f"Best value: {study.best_value:.4f}")

# Best params
if cfg["artifacts"].get("save_best_params", True):
    best_params_file = output_dir / f"best_params_{timestamp}.json"
    with open(best_params_file, "w") as f:
        json.dump({**study.best_params, "best_value": study.best_value}, f, indent=2)
    logger.info(f"Saved best params to {best_params_file}")

# Best results
if cfg["artifacts"].get("save_best_results", True):
    best_params = study.best_trial.params
    best_results = []

    for item in tqdm(ground_truth, desc="Computing best trial results"):
        query = item["query"]
        relevant_ids = item["relevant_doc_ids"]

        retrieved_docs = hybrid_wrapper(query, best_params["retrieve_k"])
        reranked_docs = crossencoder_rerank(
            query=query,
            documents=retrieved_docs,
            model_name=best_params["rerank_model"],
            top_k=best_params["rerank_k"],
            device=cfg["compute"].get("device", "cpu"),
            batch_size=cfg["compute"].get("batch_size", 16),
        )

        retrieved_ids = [doc["id"] for doc in reranked_docs]
        metrics = compute_metrics(retrieved_ids, relevant_ids, best_params["rerank_k"])

        best_results.append({
            "query": query,
            "relevant_doc_ids": relevant_ids,
            "reranked_doc_ids": retrieved_ids,
            **metrics
        })

    best_results_file = output_dir / f"best_results_{timestamp}.json"
    with open(best_results_file, "w") as f:
        json.dump(best_results, f, indent=2)
    logger.info(f"Saved best results to {best_results_file}")

# Study summary
if cfg["artifacts"].get("save_study_summary", True):
    study_summary_file = output_dir / f"study_summary_{timestamp}.json"
    study.trials_dataframe().to_json(study_summary_file, orient="records", indent=2)
    logger.info(f"Saved study summary to {study_summary_file}")

logger.info("HPO completed successfully.")