"""
KidsTALC Data Cleaning Script
==============================
Cleans KidsTALC German child speech manifests by applying conservative
filtering. Unlike PAVSig, KidsTALC already has orthographic German
transcriptions — no IPA conversion needed.

Filtering criteria:
  - Remove empty transcriptions
  - Remove utterances shorter than min_duration seconds
  - Remove utterances longer than max_duration seconds
  - Remove utterances with fewer than min_words words
  - Normalize whitespace in text

Usage:
    python clean_kidstalc.py

Input:
    /lp-dev/amelia/data/kidstalc/{train,val,test}.json

Output:
    /lp-dev/amelia/data/kidstalc/cleaned/{train,val,test}.json
"""

import json
import re
import sys
from pathlib import Path

# =============================================================================
# FILTERING PARAMETERS
# Conservative thresholds to preserve as much child speech as possible.
# KidsTALC is only ~11h total, so we cannot afford aggressive filtering.
# =============================================================================

MIN_DURATION = 0.5    # seconds — remove sub-half-second clips
MIN_WORDS    = 1      # minimum number of words after normalization

# =============================================================================
# TEXT NORMALISATION
# =============================================================================

# German-specific filler tokens and disfluencies that carry no ASR signal.
# These are single-token utterances only — we don't strip them from longer
# utterances since they may be part of meaningful context.
SINGLE_TOKEN_FILLERS = {
    # Affirmatives / negatives
    "ja", "nee", "nein", "doch", "ok", "okay",
    # Discourse particles
    "da", "und", "aber", "oder", "auch", "so", "noch", "jetzt",
    "dann", "mal", "nur", "schon", "hier", "da", "an", "auf",
    # Interjections / hesitations
    "äh", "ähm", "hm", "hmh", "mmh", "mm", "oh", "ah", "ach",
    "ooh", "uuh", "nja", "na", "hä", "ey", "hey",
    # Single-word fragments that appear in the data
    "er", "die", "der", "ein", "ist", "da", "und", "zu",
    "ja", "nee", "erst", "leicht", "grün", "zwei",
}


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and strip leading/trailing spaces.
    Does NOT lowercase — German capitalisation is meaningful for nouns
    and the tokenizer should see natural case.
    """
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_valid(item: dict) -> tuple[bool, str]:
    """
    Check whether a manifest entry passes all filters.
    Returns (is_valid, reason_if_rejected).
    """
    text     = normalize_text(item.get("text", ""))
    duration = item.get("duration", 0.0)

    # Empty text
    if not text:
        return False, "empty_text"

    # Duration bounds
    if duration < MIN_DURATION:
        return False, f"too_short ({duration:.2f}s)"

    # Word count
    words = text.split()
    if len(words) < MIN_WORDS:
        return False, f"too_few_words ({len(words)})"

    return True, ""


def clean_manifest(input_path: Path, output_path: Path) -> dict:
    """
    Clean one manifest file.
    Returns a stats dictionary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "written": 0,
        "rejected": 0,
        "reasons": {},
    }

    with open(input_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for lineno, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON error at line {lineno}: {e}")
                stats["rejected"] += 1
                stats["reasons"]["json_error"] = \
                    stats["reasons"].get("json_error", 0) + 1
                continue

            stats["total"] += 1

            valid, reason = is_valid(item)

            if not valid:
                stats["rejected"] += 1
                stats["reasons"][reason.split(" ")[0]] = \
                    stats["reasons"].get(reason.split(" ")[0], 0) + 1
                continue

            # Write with normalized text
            item["text"] = normalize_text(item["text"])
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write("\n")
            stats["written"] += 1

    return stats


def compute_hours(manifest_path: Path) -> float:
    """Sum durations in a manifest to get total hours."""
    total = 0.0
    if not manifest_path.exists():
        return total
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                total += item.get("duration", 0.0)
            except Exception:
                pass
    return total / 3600.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_in  = Path("/lp-dev/amelia/data/kidstalc")
    base_out = Path("/lp-dev/amelia/data/kidstalc/cleaned")

    splits = ["train", "val"] #, "test"]

    print("=" * 60)
    print("KidsTALC Cleaning Script")
    print(f"  min_duration : {MIN_DURATION}s")
    print(f"  min_words    : {MIN_WORDS}")
    print("=" * 60)

    existing = [s for s in splits if (base_in / f"{s}.json").exists()]
    if not existing:
        print(f"ERROR: No manifest files found in {base_in}")
        print(f"       Expected: train.json, val.json, test.json")
        sys.exit(1)

    total_in_hours  = 0.0
    total_out_hours = 0.0

    for split in splits:
        input_path  = base_in  / f"{split}.json"
        output_path = base_out / f"{split}.json"

        if not input_path.exists():
            print(f"\n[SKIP] {input_path} not found.")
            continue

        hours_before = compute_hours(input_path)
        print(f"\nCleaning {split} ({hours_before:.2f}h)...")

        stats = clean_manifest(input_path, output_path)

        hours_after = compute_hours(output_path)
        retention   = (stats["written"] / stats["total"] * 100) \
                       if stats["total"] > 0 else 0

        print(f"  Input  : {stats['total']:>6,} utterances  ({hours_before:.2f}h)")
        print(f"  Output : {stats['written']:>6,} utterances  ({hours_after:.2f}h)")
        print(f"  Removed: {stats['rejected']:>6,} utterances  "
              f"({retention:.1f}% retained)")
        if stats["reasons"]:
            print(f"  Rejection breakdown:")
            for reason, count in sorted(stats["reasons"].items(),
                                        key=lambda x: -x[1]):
                print(f"    {reason:30s}: {count:,}")
        print(f"  -> {output_path}")

        total_in_hours  += hours_before
        total_out_hours += hours_after

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total input  : {total_in_hours:.2f}h")
    print(f"  Total output : {total_out_hours:.2f}h")
    print(f"  Hours removed: {total_in_hours - total_out_hours:.2f}h "
          f"({(1 - total_out_hours/total_in_hours)*100:.1f}%)"
          if total_in_hours > 0 else "")
    print(f"\n  Output directory: {base_out}")


if __name__ == "__main__":
    main()