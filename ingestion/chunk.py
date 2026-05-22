# ingestion/chunk.py
import json
import logging
from pathlib import Path
from datetime import datetime, UTC
import yaml
from typing import List, Dict
from ingestion.utils import log_manifest as _log_manifest_shared

logger = logging.getLogger(__name__)

# --------------------
# Paths
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
CHUNK_DIR = BASE_DIR / "data" / "chunked"
CONFIG_PATH = BASE_DIR / "ingestion" / "ingestion.yaml"
MANIFEST_PATH = CHUNK_DIR / "_manifest.json"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)

RAW_SCHEMA_VERSION = "raw_v1"
CHUNK_SCHEMA_VERSION = "chunk_v1"

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
    _log_manifest_shared(MANIFEST_PATH, entry)



# --------------------
# Chunking logic
# --------------------
def chunk_segments(
    segments: List[Dict],
    max_chars: int,
    overlap_chars: int,
) -> List[Dict]:
    """
    Chunk transcript segments by approximate character count
    with overlap for RAG.
    """
    chunks = []
    current_segments = []
    current_length = 0

    for i, seg in enumerate(segments):
        seg_len = len(seg["text"])

        if current_length + seg_len > max_chars and current_segments:
            chunks.append(current_segments)

            # build overlap
            overlap = []
            overlap_len = 0
            for s in reversed(current_segments):
                if overlap_len >= overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s["text"])

            current_segments = overlap.copy()
            current_length = overlap_len

        current_segments.append(seg)
        current_length += seg_len

    if current_segments:
        chunks.append(current_segments)

    return chunks

# --------------------
# Normalize chunk schema
# --------------------
def build_chunk_payload(
    video_id: str,
    chunk_idx: int,
    segments: List[Dict],
) -> Dict:
    return {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_id": chunk_idx,
        "video_id": video_id,
        "start": segments[0]["start"],
        "end": segments[-1]["start"] + segments[-1]["duration"],
        "text": " ".join(s["text"] for s in segments),
        "source_segment_ids": list(range(len(segments))),
    }

# --------------------
# Main
# --------------------
def main(overwrite: bool = False) -> None:
    config = load_config(CONFIG_PATH)

    max_chars = config.get("chunking", {}).get("max_chars", 2000)
    overlap_chars = config.get("chunking", {}).get("overlap_chars", 200)

    for raw_file in RAW_DIR.glob("*.json"):
        if raw_file.name == "_manifest.json":
            continue

        out_path = CHUNK_DIR / raw_file.name

        if out_path.exists() and not overwrite:
            logger.info(f"[SKIP] {raw_file.stem}")
            continue

        try:
            with open(raw_file) as f:
                raw = json.load(f)

            assert raw["schema_version"] == RAW_SCHEMA_VERSION

            video_id = raw["video_id"]
            segments = raw["segments"]

            grouped = chunk_segments(
                segments,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

            chunks = [
                build_chunk_payload(video_id, i, group)
                for i, group in enumerate(grouped)
            ]

            with open(out_path, "w") as f:
                json.dump(chunks, f, indent=2)

            log_manifest(
                {
                    "video_id": video_id,
                    "num_chunks": len(chunks),
                    "status": "ok",
                    "schema_version": CHUNK_SCHEMA_VERSION,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            logger.info(f"[OK] {video_id}: {len(chunks)} chunks")

        except json.JSONDecodeError as e:
            log_manifest(
                {
                    "video_id": raw_file.stem,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            logger.exception(f"[ERROR] {raw_file.stem}: JSON parse error: {e}")
        except OSError as e:
            log_manifest(
                {
                    "video_id": raw_file.stem,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            logger.exception(f"[ERROR] {raw_file.stem}: File I/O error: {e}")
        except Exception as e:
            log_manifest(
                {
                    "video_id": raw_file.stem,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            logger.exception(f"[ERROR] {raw_file.stem}: {e}")


# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    config = load_config(CONFIG_PATH)
    main(overwrite=config.get("overwrite", False))
