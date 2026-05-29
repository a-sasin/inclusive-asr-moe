"""
JASMIN Child Speech Manifest Creator
=====================================
Reads existing JASMIN manifests (train/val/test), filters to Group 1
(native Dutch child speakers, ages 6-13), cleans transcription text,
and writes new speaker-independent manifests to:

    /lp-dev/amelia/data/jasmin/child/{train,val,test}.json

Child speaker identification:
    Group == 1 in recordings.txt = native Dutch children (confirmed from
    speakers.txt: ages 6-13, HomeLanguage1=dut, no CEF level)

Text cleaning applied:
    - Strip trailing/leading punctuation (. , ; : ! ?)
    - Remove utterances containing only JASMIN non-speech markers
      (ggg, xxx, yyy, uh, um and combinations thereof)
    - Remove truncated utterances (ending in ... or containing n...)
    - Normalize whitespace
    - Remove empty utterances after cleaning
    - Keep min_duration >= 0.5s

Usage:
    python make_jasmin_child_manifests.py

Input:
    /data/JASMIN/manifests/{train,val,test}.json       (existing manifests)
    /data/JASMIN/Data/data/meta/text/nl/recordings.txt (speaker metadata)

Output:
    /lp-dev/amelia/data/jasmin/child/{train,val,test}.json
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

RECORDINGS_TXT = Path("/data/JASMIN/Data/data/meta/text/nl/recordings.txt")

INPUT_DIR  = Path("/data/JASMIN/manifests")
OUTPUT_DIR = Path("/lp-dev/amelia/data/jasmin/child")

SPLITS = ["train", "dev", "test"]
SPLIT_RENAME = {"dev": "val"}  # dev.json -> val.json in output

# Only include Group 1 = native Dutch children
CHILD_GROUP = "1"

# Only include Netherlands Dutch speakers (DialectRegion starts with N)
# Flemish speakers have region codes starting with V — excluded
NL_DIALECT_PREFIX = "N"

# Minimum duration in seconds
MIN_DURATION = 0.5

# =============================================================================
# STEP 1: LOAD CHILD SPEAKER ROOTS FROM RECORDINGS.TXT
# =============================================================================

def load_child_roots(recordings_path: Path) -> set:
    """
    Parse recordings.txt and return the set of Root IDs (e.g. 'fn000048')
    that belong to Group 1 (native Dutch children) AND have a Netherlands
    Dutch dialect region (DialectRegion starting with 'N').

    recordings.txt columns (tab-separated):
    Root  SpeakerID  Component  Group  Age  Gender  CEF  DialectRegion
    Duration(seconds)  Duration(days)
    """
    child_roots = set()
    skipped = 0
    rejected_vl = 0

    with open(recordings_path, encoding="utf-8") as f:
        header = f.readline()  # skip header

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                skipped += 1
                continue

            root    = parts[0].strip()   # e.g. fn000048
            group   = parts[3].strip()   # e.g. 1, 2, 3, 4, 5
            dialect = parts[7].strip()   # e.g. N2c, N4a, V2c

            if group != CHILD_GROUP:
                continue

            # NL filter: dialect region must start with N
            if not dialect.startswith(NL_DIALECT_PREFIX):
                rejected_vl += 1
                continue

            child_roots.add(root)

    print(f"  Child roots loaded   : {len(child_roots)} (NL only)")
    if rejected_vl:
        print(f"  Rejected VL speakers : {rejected_vl}")
    if skipped:
        print(f"  Skipped malformed    : {skipped}")

    return child_roots


# =============================================================================
# STEP 2: TEXT CLEANING
# =============================================================================

# JASMIN non-speech markers that should cause utterance removal if they
# are the only content after cleaning
NON_SPEECH_TOKENS = {"ggg", "xxx", "yyy", "uh", "um", "mmm", "hmm"}

# Regex: trailing punctuation to strip
_TRAILING_PUNCT = re.compile(r"[.,:;!?]+$")

# Regex: truncation marker (word cut off with ...)
_TRUNCATION     = re.compile(r"\.\.\.")

# Regex: mid-word truncation like 'n...' or 'leu...'
_MID_TRUNCATION = re.compile(r"\w+\.\.\.")


def clean_text(text: str) -> str | None:
    """
    Clean a JASMIN transcription.
    Returns cleaned string, or None if utterance should be discarded.
    """
    # Remove truncation markers — if present, discard the utterance
    # because it's incomplete and will confuse CTC training
    if _TRUNCATION.search(text):
        return None

    # Strip trailing punctuation
    text = _TRAILING_PUNCT.sub("", text).strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Empty after cleaning
    if not text:
        return None

    # Check if utterance is purely non-speech markers
    tokens = text.lower().split()
    if all(t in NON_SPEECH_TOKENS for t in tokens):
        return None

    # Remove individual non-speech tokens from within utterance
    # e.g. 'uh ggg sommige mensen' -> 'sommige mensen'
    cleaned_tokens = [t for t in text.split()
                      if t.lower() not in NON_SPEECH_TOKENS]
    if not cleaned_tokens:
        return None

    text = " ".join(cleaned_tokens).strip()

    # Final empty check
    if not text:
        return None

    return text


# =============================================================================
# STEP 3: PROCESS MANIFESTS
# =============================================================================

def process_manifest(
    input_path: Path,
    output_path: Path,
    child_roots: set,
) -> dict:
    """
    Filter and clean one manifest split.
    Returns stats dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "written": 0,
        "rejected_not_child": 0,
        "rejected_truncated": 0,
        "rejected_non_speech": 0,
        "rejected_empty": 0,
        "rejected_too_short": 0,
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
                print(f"  [WARN] JSON error line {lineno}: {e}")
                continue

            stats["total"] += 1

            # Extract root from filename e.g. fn000048_00001.wav -> fn000048
            audio_path = Path(item["audio_filepath"])
            root = audio_path.stem.split("_")[0]

            # Filter: must be a child speaker
            if root not in child_roots:
                stats["rejected_not_child"] += 1
                continue

            # Filter: minimum duration
            duration = item.get("duration", 0.0)
            if duration < MIN_DURATION:
                stats["rejected_too_short"] += 1
                continue

            # Clean text
            original_text = item.get("text", "")

            # Detect truncation before cleaning (for stats)
            if _TRUNCATION.search(original_text):
                stats["rejected_truncated"] += 1
                continue

            cleaned = clean_text(original_text)

            if cleaned is None:
                # Distinguish non-speech vs empty
                tokens = original_text.lower().split()
                if all(t in NON_SPEECH_TOKENS for t in tokens) and tokens:
                    stats["rejected_non_speech"] += 1
                else:
                    stats["rejected_empty"] += 1
                continue

            # Write cleaned entry
            item["text"] = cleaned
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write("\n")
            stats["written"] += 1

    return stats


def compute_hours(manifest_path: Path) -> float:
    if not manifest_path.exists():
        return 0.0
    total = 0.0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                total += json.loads(line).get("duration", 0.0)
            except Exception:
                pass
    return total / 3600.0


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("JASMIN Child Speech Manifest Creator")
    print("=" * 65)

    # Validate inputs
    if not RECORDINGS_TXT.exists():
        print(f"ERROR: recordings.txt not found at {RECORDINGS_TXT}")
        sys.exit(1)

    existing_splits = [s for s in SPLITS
                       if (INPUT_DIR / f"{s}.json").exists()]
    if not existing_splits:
        print(f"ERROR: No manifest files found in {INPUT_DIR}")
        sys.exit(1)

    # Load child speaker roots
    print(f"\nLoading child speakers from:\n  {RECORDINGS_TXT}")
    child_roots = load_child_roots(RECORDINGS_TXT)

    # Process each split
    total_in_hours  = 0.0
    total_out_hours = 0.0
    total_written   = 0
    total_rejected  = 0

    for split in SPLITS:
        input_path  = INPUT_DIR  / f"{split}.json"
        output_name = SPLIT_RENAME.get(split, split)
        output_path = OUTPUT_DIR / f"{output_name}.json"

        if not input_path.exists():
            print(f"\n[SKIP] {input_path} not found.")
            continue

        hours_before = compute_hours(input_path)
        print(f"\nProcessing {split} ({hours_before:.1f}h total, "
              f"all speakers)...")

        stats = process_manifest(input_path, output_path, child_roots)

        hours_after = compute_hours(output_path)
        retention   = (stats["written"] / stats["total"] * 100
                       if stats["total"] > 0 else 0)

        print(f"  Input              : {stats['total']:>7,} utterances")
        print(f"  Written (child)    : {stats['written']:>7,} utterances  "
              f"({hours_after:.2f}h)  [{retention:.1f}% of input]")
        print(f"  -- Not child       : {stats['rejected_not_child']:>7,}")
        print(f"  -- Truncated (...)  : {stats['rejected_truncated']:>7,}")
        print(f"  -- Non-speech only : {stats['rejected_non_speech']:>7,}")
        print(f"  -- Empty           : {stats['rejected_empty']:>7,}")
        print(f"  -- Too short (<0.5s): {stats['rejected_too_short']:>7,}")
        print(f"  -> {output_path}")

        total_in_hours  += hours_after   # child hours before cleaning
        total_out_hours += hours_after
        total_written   += stats["written"]
        total_rejected  += (stats["rejected_truncated"] +
                            stats["rejected_non_speech"] +
                            stats["rejected_empty"] +
                            stats["rejected_too_short"])

    # Final summary
    print()
    print("=" * 65)
    print("SUMMARY — Child speech only")
    print("=" * 65)
    print(f"  Total utterances : {total_written:,}")
    print(f"  Total hours      : {total_out_hours:.2f}h")
    print(f"  Output directory : {OUTPUT_DIR}")

    # Speaker count across all output splits
    all_roots = set()
    for split in SPLITS:
        p = OUTPUT_DIR / f"{split}.json"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    root = Path(item["audio_filepath"]).stem.split("_")[0]
                    all_roots.add(root)
                except Exception:
                    pass
    print(f"  Unique speakers  : {len(all_roots)}")

    # Age distribution from recordings.txt
    print()
    print("Age distribution of included child speakers:")
    age_counts = defaultdict(int)
    with open(RECORDINGS_TXT, encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            root  = parts[0].strip()
            group = parts[3].strip()
            age   = parts[4].strip()
            if group == CHILD_GROUP and root in all_roots and age.isdigit():
                age_counts[int(age)] += 1
    for age in sorted(age_counts):
        print(f"    Age {age:2d}: {age_counts[age]} speakers")


if __name__ == "__main__":
    main()