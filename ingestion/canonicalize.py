# ingestion/canonicalize.py
import json
from pathlib import Path
from datetime import datetime, UTC
import yaml

# --------------------
# Paths
# --------------------

BASE_DIR = Path(__file__).resolve().parents[1]

CHUNK_DIR = BASE_DIR / "data" / "chunked"
CANONICAL_DIR = BASE_DIR / "data" / "canonical"
CONFIG_PATH = BASE_DIR / "configs" / "ingestion.yaml"
MANIFEST_PATH = CANONICAL_DIR / "_manifest.json"

CANONICAL_DIR.mkdir(parents=True, exist_ok=True)
CANONICAL_SCHEMA_VERSION = "canonical_v1"

# --------------------
# Config
# --------------------
def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------
# Manifest
# --------------------
def log_manifest(entry: dict) -> None:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH) as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            manifest = []
    else:
        manifest = []

    manifest.append(entry)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

# --------------------
# Canonical conversion
# --------------------
def convert_chunks_to_canonical(
    chunks: list[dict],
    video_id: str,
    title: str,
) -> list[dict]:
    canonical_docs = []

    for chunk in chunks:
        chunk_idx = int(chunk["chunk_id"])

        canonical_docs.append(
            {
                "id": f"{video_id}__chunk_{chunk_idx:03d}",
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "video_id": video_id,
                "title": title,
                "timestamp_start": chunk["start"],
                "timestamp_end": chunk["end"],
                "text": chunk["text"],
                "source": "youtube",
                "speaker": "unknown",
            }
        )

    return canonical_docs

# --------------------
# I/O helpers
# --------------------
def load_chunks(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_canonical_docs(video_id: str, docs: list[dict]) -> None:
    out_path = CANONICAL_DIR / f"{video_id}.json"
    with open(out_path, "w") as f:
        json.dump(docs, f, indent=2)


# --------------------
# Main
# --------------------
def main(overwrite: bool = False) -> None:
    config = load_config(CONFIG_PATH)
    video_titles = config.get("video_titles", {})

    for chunk_file in CHUNK_DIR.glob("*.json"):
        if chunk_file.name.startswith("_"):
            continue

        video_id = chunk_file.stem
        out_path = CANONICAL_DIR / f"{video_id}.json"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {video_id}")
            continue

        try:
            chunks = load_chunks(chunk_file)
            title = video_titles.get(video_id, f"Google I/O 2025 – {video_id}")
            canonical_docs = convert_chunks_to_canonical(chunks=chunks, video_id=video_id, title=title)
            save_canonical_docs(video_id, canonical_docs)
            log_manifest({...})
            print(f"[OK] Canonicalized {video_id} ({len(canonical_docs)} docs)")

        except Exception as e:
            log_manifest({...})
            print(f"[ERROR] {video_id}: {e}")

    # ✅ Always rebuild from disk — independent of what was skipped/processed
    all_docs = []
    for canonical_file in CANONICAL_DIR.glob("*.json"):
        if canonical_file.name.startswith("_") or canonical_file.name == "all_documents.json":
            continue
        with open(canonical_file) as f:
            all_docs.extend(json.load(f))

    with open(CANONICAL_DIR / "all_documents.json", "w") as f:
        json.dump(all_docs, f, indent=2)

    print(f"\n[OK] Saved all_documents.json ({len(all_docs)} total docs)")

# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    main(overwrite=config.get("overwrite", False))