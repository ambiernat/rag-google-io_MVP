# ingestion/fetch.py
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
import yaml
import time
from ingestion.utils import log_manifest as _log_manifest_shared

logger = logging.getLogger(__name__)

# --------------------
# Paths
# --------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_PATH = BASE_DIR / "ingestion" / "ingestion.yaml"
MANIFEST_PATH = RAW_DIR / "_manifest.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_SCHEMA_VERSION = "raw_v1"


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
# Fetch logic
# --------------------

api = YouTubeTranscriptApi()

def fetch_transcript(video_id: str, languages=("en",)):
    logger.info("  → Requesting transcript from YouTube...")
    transcripts = api.list(video_id)
    logger.info("  → Found available transcripts")

    transcript = transcripts.find_transcript(languages)
    logger.info(f"  → Downloading transcript in {transcript.language_code}...")

    segments = [
        {
            "text": item.text,
            "start": item.start,
            "duration": item.duration,
        }
        for item in transcript.fetch()
    ]

    logger.info(f"  → Downloaded {len(segments)} segments")
    return segments, transcript.language_code


# --------------------
# Normalize raw schema
# --------------------
def build_raw_payload(
    video_id: str,
    language: str,
    segments: list[dict],
) -> dict:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "video_id": video_id,
        "language": language,
        "segments": segments,
    }

# --------------------
# Main
# --------------------
def main(video_ids: list[str], languages: list[str], overwrite: bool = False) -> None:
    languages = tuple(languages)
    total = len(video_ids)

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting fetch for {total} videos")
    logger.info(f"Overwrite mode: {overwrite}")
    logger.info(f"{'='*60}\n")

    for idx, video_id in enumerate(video_ids, 1):
        logger.info(f"\n[{idx}/{total}] Processing: {video_id}")
        logger.info(f"  URL: https://www.youtube.com/watch?v={video_id}")

        out_path = RAW_DIR / f"{video_id}.json"

        if out_path.exists() and not overwrite:
            logger.info("  [SKIP] Already exists (overwrite=False)")
            continue

        if out_path.exists() and overwrite:
            logger.info("  [OVERWRITE] File exists, re-downloading...")

        try:
            start_time = time.time()
            segments, language = fetch_transcript(video_id, languages)
            fetch_duration = time.time() - start_time

            logger.info("  → Saving to disk...")

            payload = build_raw_payload(
                video_id=video_id,
                language=language,
                segments=segments,
            )

            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

            log_manifest(
                {
                    "video_id": video_id,
                    "language": language,
                    "status": "ok",
                    "schema_version": RAW_SCHEMA_VERSION,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            logger.info(f"  ✓ SUCCESS in {fetch_duration:.1f}s")
            logger.info("  → Waiting 2 seconds before next video...")
            time.sleep(2)

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            log_manifest(
                {
                    "video_id": video_id,
                    "language": languages[0],
                    "status": "no_transcript",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.warning(f"  NO TRANSCRIPT: {e}")

        except Exception as e:
            log_manifest(
                {
                    "video_id": video_id,
                    "language": languages[0],
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            logger.exception(f"  ERROR: {e}")
            logger.info("  → Waiting 5 seconds before retry...")
            time.sleep(5)

    logger.info(f"\n{'='*60}")
    logger.info(f"Fetch complete! Processed {total} videos")
    logger.info(f"{'='*60}\n")

# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    config = load_config(CONFIG_PATH)

    main(
        video_ids=config["videos"],
        languages=config.get("languages", ["en"]),
        overwrite=config.get("overwrite", False),
    )
