"""
Build Merged Granary + Child Speech Manifest for Tokenizer Training
====================================================================
Merges all Granary shard manifests and child speech manifests into a
single JSONL file, preserving the original Granary field format.

Output:
    /lp-dev/amelia/inclusive-asr-moe/tokenizers/tokenizer_text_all.json

Usage:
    python build_tokenizer_manifest.py
"""

import json
import glob
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_MANIFEST = Path(
    "/lp-dev/amelia/inclusive-asr-moe/tokenizers/tokenizer_text_all.json"
)

CHILD_MANIFESTS = [
    "/lp-dev/amelia/data/jasmin/child/train.json",
    "/lp-dev/amelia/data/jasmin/child/val.json",
    "/lp-dev/amelia/data/kidstalc/cleaned/train.json",
    "/lp-dev/amelia/data/myst/filtered_data/train.json",
    "/lp-dev/amelia/data/myst/filtered_data/val.json",
    "/lp-dev/amelia/data/pavsig/orto/train.json",
    "/lp-dev/amelia/data/pavsig/orto/val.json",
]

# Glob patterns using actual filename format: manifest_*.json
GRANARY_MANIFEST_GLOBS = [
    # English
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/AMI/IHM-ASR/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/MOSEL/en/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YTC/en*/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/en/0_by_whisper/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/en/1_by_whisper/bucket*/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/en/0_from_captions/sharded_manifests_updated/manifest_*.json",
    # German
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/MOSEL/de/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YTC/de/webds_de/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/de/0_by_whisper/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/de/1_by_whisper/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/de/0_from_captions/sharded_manifests_updated/manifest_*.json",
    # Dutch
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/MOSEL/nl/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YTC/nl/webds_nl/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/nl/0_by_whisper/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/nl/1_by_whisper/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/nl/0_from_captions/sharded_manifests_updated/manifest_*.json",
    # Polish
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/MOSEL/pl/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YTC/pl/webds_pl/sharded_manifests/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/pl/0_by_whisper/sharded_manifests_updated/manifest_*.json",
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/pl/0_from_captions/sharded_manifests_updated/manifest_*.json",
]


def get_text(item):
    for field in ("text", "answer", "cleaned_text"):
        val = item.get(field, "").strip()
        if val:
            return val
    return None


def process_file(path, out_file, keep_all_fields=True):
    written = skipped = 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                text = get_text(item)
                if not text:
                    skipped += 1
                    continue
                if keep_all_fields:
                    item["text"] = text
                    out_file.write(
                        json.dumps(item, ensure_ascii=False) + "\n"
                    )
                else:
                    out_file.write(json.dumps({
                        "audio_filepath": item.get("audio_filepath", ""),
                        "duration": item.get("duration", 0.0),
                        "text": text,
                    }, ensure_ascii=False) + "\n")
                written += 1
    except FileNotFoundError:
        print(f"  [WARN] Not found: {path}")
    return written, skipped


def main():
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Building merged manifest")
    print(f"Output: {OUTPUT_MANIFEST}")
    print("=" * 65)

    total_written = 0
    total_skipped = 0

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as out:

        # --- Child speech ---
        print("\n[1/2] Child speech manifests")
        for path in CHILD_MANIFESTS:
            if not Path(path).exists():
                print(f"  [SKIP] {path}")
                continue
            w, s = process_file(path, out, keep_all_fields=False)
            print(f"  {Path(path).name:50s}  {w:>7,} entries")
            total_written += w
            total_skipped += s

        # --- Granary ---
        print("\n[2/2] Granary shard manifests")
        all_paths = []
        for pattern in GRANARY_MANIFEST_GLOBS:
            matched = sorted(glob.glob(pattern))
            if not matched:
                print(f"  [WARN] No match: {pattern}")
            all_paths.extend(matched)

        print(f"  Resolved {len(all_paths)} shard files — processing...")

        g_written = g_skipped = 0
        for i, path in enumerate(all_paths, 1):
            w, s = process_file(path, out, keep_all_fields=True)
            g_written += w
            g_skipped += s
            if i % 500 == 0:
                print(f"  {i}/{len(all_paths)} files, "
                      f"{g_written:,} entries so far...")

        print(f"  Done: {len(all_paths)} files, {g_written:,} entries")
        total_written += g_written
        total_skipped += g_skipped

    print()
    print("=" * 65)
    print("DONE")
    print("=" * 65)
    print(f"  Total entries   : {total_written:,}")
    print(f"  Skipped (empty) : {total_skipped:,}")
    print(f"  Output          : {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()