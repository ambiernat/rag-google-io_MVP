# retrieval/retrievers/retrieve_sparse.py

"""
retrieve_sparse.py
Function-based sparse retriever using Qdrant-native BM25.
Mirrors retrieve_dense.py structure and behavior.
"""
from pathlib import Path
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Document
import os

# -------------------------------------------------
# Load config
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # retrieval/
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Allow environment variable to override
QDRANT_URL = os.getenv("QDRANT_URL", config["qdrant"]["url"])
SPARSE_CFG = config["sparse"]
COLLECTION_NAME = SPARSE_CFG["collection_name"]
DEFAULT_TOP_K = SPARSE_CFG.get("top_k", 5)

# -------------------------------------------------
# Initialize shared state (once per process)
# -------------------------------------------------
q_client = QdrantClient(url=QDRANT_URL)

print(f"[INFO] Sparse retriever initialized")
print(f"[INFO] Collection: {COLLECTION_NAME}")

# -------------------------------------------------
# Public API
# -------------------------------------------------
def retrieve_sparse(query: str, top_k: int = DEFAULT_TOP_K, client: QdrantClient | None = None):
    """
    Retrieve top-K documents from sparse Qdrant BM25 collection.

    Args:
        query (str): input query
        top_k (int): number of results

    Returns:
        List of ScoredPoint objects with .id, .score, .payload attributes
        (compatible with evaluate_sparse.py expectations)
    """
    q_client_to_use = client or q_client

    response = q_client_to_use.query_points(
        collection_name=COLLECTION_NAME,
        query=Document(
            text=query,
            model="Qdrant/bm25",
        ),
        using="bm25",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    return response.points

# -------------------------------------------------
# Example usage (non-interactive)
# -------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is Gemma?",
        "Who is speaking in the keynote?",
    ]

    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retrieve_sparse(q)

        for i, point in enumerate(results, 1):
            print(f"\nResult {i}")
            print("-" * 40)
            print(f"Score: {point.score:.4f}")
            print(f"Doc ID: {point.payload.get('doc_id', point.id)}")
            text = point.payload.get("text", "")
            print(f"Text: {text[:300]}..." if len(text) > 300 else f"Text: {text}")


# import pickle
# import numpy as np
# import json
# from pathlib import Path

# class SparseRetriever:
#     """
#     BM25-based sparse retriever using pre-built BM25 encoder.
#     Fully independent of Qdrant.
#     """
#     def __init__(
#         self,
#         bm25_path: str = "data/models/bm25_encoder.pkl",
#         docs_path: str = "data/canonical/all_documents.json"
#     ):
#         # Load BM25 encoder
#         if not Path(bm25_path).exists():
#             raise FileNotFoundError(f"BM25 pickle not found at {bm25_path}")
#         with open(bm25_path, "rb") as f:
#             self.bm25_encoder = pickle.load(f)

#         # Load documents
#         if not Path(docs_path).exists():
#             raise FileNotFoundError(f"Documents file not found at {docs_path}")
#         with open(docs_path, "r", encoding="utf-8") as f:
#             self.documents = json.load(f)

#     def retrieve(self, query: str, top_k: int = 5):
#         """
#         Retrieve top-K documents using BM25.
        
#         Args:
#             query (str): text query
#             top_k (int): number of top results

#         Returns:
#             List of tuples: (score, document dict)
#         """
#         query_tokens = query.lower().split()
#         scores = self.bm25_encoder.bm25.get_scores(query_tokens)
#         top_indices = np.argsort(scores)[::-1][:top_k]
#         return [(scores[i], self.documents[i]) for i in top_indices]


# # ---------------------------
# # Example usage
# # ---------------------------
# if __name__ == "__main__":
#     retriever = SparseRetriever()
#     queries = [
#         "Who is speaking in the video?",
#         "What is Gemma?"
#     ]
#     for q in queries:
#         print(f"\n=== Query: {q} ===")
#         results = retriever.retrieve(q, top_k=5)
#         if not results:
#             print("No results found.")
#         else:
#             for i, (score, doc) in enumerate(results, 1):
#                 text_snippet = doc.get("text", "")[:300]
#                 print(f"\nResult {i}")
#                 print("-" * 40)
#                 print(f"Score: {score:.4f}")
#                 print(f"Doc ID: {doc.get('doc_id', 'N/A')}")
#                 print(f"Text: {text_snippet}...")
