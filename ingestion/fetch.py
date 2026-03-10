# ingestion/fetch.py
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
import json
from pathlib import Path
from datetime import datetime, timezone
import yaml
import time

# --------------------
# Paths
# --------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_PATH = BASE_DIR / "configs" / "ingestion.yaml"
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
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
    else:
        manifest = []

    manifest.append(entry)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

# --------------------
# Fetch logic
# --------------------

api = YouTubeTranscriptApi()

def fetch_transcript(video_id: str, languages=("en",)):
    print(f"  → Requesting transcript from YouTube...")  # ← Added
    transcripts = api.list(video_id)
    print(f"  → Found available transcripts")  # ← Added
    
    transcript = transcripts.find_transcript(languages)
    print(f"  → Downloading transcript in {transcript.language_code}...")  # ← Added

    segments = [
        {
            "text": item.text,
            "start": item.start,
            "duration": item.duration,
        }
        for item in transcript.fetch()
    ]

    print(f"  → Downloaded {len(segments)} segments")  # ← Added
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
    total = len(video_ids)  # ← Added
    
    print(f"\n{'='*60}")  # ← Added
    print(f"Starting fetch for {total} videos")  # ← Added
    print(f"Overwrite mode: {overwrite}")  # ← Added
    print(f"{'='*60}\n")  # ← Added

    for idx, video_id in enumerate(video_ids, 1):  # ← Changed to enumerate
        print(f"\n[{idx}/{total}] Processing: {video_id}")  # ← Added
        print(f"  URL: https://www.youtube.com/watch?v={video_id}")  # ← Added
        
        out_path = RAW_DIR / f"{video_id}.json"

        if out_path.exists() and not overwrite:
            print(f"  [SKIP] Already exists (overwrite=False)")
            continue
        
        if out_path.exists() and overwrite:
            print(f"  [OVERWRITE] File exists, re-downloading...")  # ← Added

        try:
            start_time = time.time()  # ← Added
            segments, language = fetch_transcript(video_id, languages)
            fetch_duration = time.time() - start_time  # ← Added
            
            print(f"  → Saving to disk...")  # ← Added
            
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

            print(f"  ✓ SUCCESS in {fetch_duration:.1f}s")  # ← Changed
            print(f"  → Waiting 2 seconds before next video...")  # ← Added
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
            print(f"  ✗ NO TRANSCRIPT: {e}")  # ← Changed

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
            print(f"  ✗ ERROR: {e}")  # ← Changed
            print(f"  → Waiting 5 seconds before retry...")  # ← Added
            time.sleep(5)
    
    print(f"\n{'='*60}")  # ← Added
    print(f"Fetch complete! Processed {total} videos")  # ← Added
    print(f"{'='*60}\n")  # ← Added

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
