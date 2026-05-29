#!/usr/bin/env python3
"""Generate word-disjoint and speaker-disjoint PAVSig manifests.

Combines all available data into a pool, holds out ~20% of unique words and
~20% of unique speakers for test, then splits the remaining (train/validation)
pool into 80% train and 20% validation. Validation is included in the final
train manifest.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Configurable inputs and outputs.
INPUT_TRAIN_MANIFEST = Path("/lp-dev/amelia/data/pavsig/orto/train.json")
INPUT_VALIDATION_MANIFEST = Path("/lp-dev/amelia/data/pavsig/orto/val.json")
INPUT_TEST_MANIFEST = Path("/lp-dev/amelia/data/pavsig/orto/test.json")
OUTPUT_DIR = Path("/lp-dev/amelia/data/pavsig/training")

# Field names to use when reading manifests.
WORD_FIELD = "text"
SPEAKER_FIELD = "speaker_id"

# Split fractions: test words/speakers held out, then validation as fraction of train pool.
TEST_WORD_FRACTION = 0.2
TEST_SPEAKER_FRACTION = 0.2
VALIDATION_FRACTION = 0.2

# Random seed for reproducibility.
RANDOM_SEED = 42

# Output filenames.
NEW_TRAIN_MANIFEST = OUTPUT_DIR / "new_train_manifest.jsonl"
NEW_VALIDATION_MANIFEST = OUTPUT_DIR / "new_val_manifest.jsonl"
NEW_TEST_MANIFEST = OUTPUT_DIR / "new_test_manifest.jsonl"
SUMMARY_REPORT = OUTPUT_DIR / "split_summary_report.txt"

# If the manifest does not contain an explicit speaker field, infer it from the
# audio path. This matches the current PAVSig manifest layout.
SPEAKER_ID_PATH_REGEX = re.compile(r"/(\d{5})/(\d{5}-\d+-audio)/")


@dataclass(frozen=True)
class ManifestEntry:
    data: dict
    source: Path
    line_number: int
    word: str
    speaker_id: str


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}: {exc.msg}"
                ) from exc
    return entries


def extract_word(sample: dict, source: Path, line_number: int) -> str:
    if WORD_FIELD not in sample:
        raise KeyError(
            f"Missing required word field '{WORD_FIELD}' in {source} line {line_number}."
        )
    word = str(sample[WORD_FIELD]).strip()
    if not word:
        raise ValueError(
            f"Empty word field '{WORD_FIELD}' in {source} line {line_number}."
        )
    return word


def extract_speaker_id(sample: dict, source: Path, line_number: int) -> str:
    speaker_value = sample.get(SPEAKER_FIELD)
    if speaker_value not in (None, ""):
        return str(speaker_value).strip()

    audio_filepath = str(sample.get("audio_filepath", "")).strip()
    if not audio_filepath:
        raise KeyError(
            f"Missing required field 'audio_filepath' in {source} line {line_number}."
        )

    match = SPEAKER_ID_PATH_REGEX.search(audio_filepath)
    if match:
        return match.group(1)

    raise KeyError(
        "Missing required speaker field '{field}' in {source} line {line_number}, "
        "and speaker could not be inferred from audio_filepath '{audio}'."
        .format(
            field=SPEAKER_FIELD,
            source=source,
            line_number=line_number,
            audio=audio_filepath,
        )
    )


def load_entries(path: Path) -> list[ManifestEntry]:
    raw_entries = read_manifest(path)
    parsed_entries: list[ManifestEntry] = []
    for line_number, sample in enumerate(raw_entries, start=1):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Expected a JSON object in {path} line {line_number}, got {type(sample).__name__}."
            )
        word = extract_word(sample, path, line_number)
        speaker_id = extract_speaker_id(sample, path, line_number)
        parsed_entries.append(
            ManifestEntry(
                data=sample,
                source=path,
                line_number=line_number,
                word=word,
                speaker_id=speaker_id,
            )
        )
    return parsed_entries


def write_jsonl(path: Path, samples: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False))
            handle.write("\n")


def summarize_entries(entries: list[ManifestEntry]) -> tuple[int, set[str], set[str]]:
    words = {entry.word for entry in entries}
    speaker_ids = {entry.speaker_id for entry in entries}
    return len(entries), words, speaker_ids


def main() -> None:
    configure_logging()
    random.seed(RANDOM_SEED)
    logging.info("Random seed set to: %d", RANDOM_SEED)
    logging.info("Reading source manifests")

    original_train = load_entries(INPUT_TRAIN_MANIFEST)
    original_validation = load_entries(INPUT_VALIDATION_MANIFEST)
    original_test = load_entries(INPUT_TEST_MANIFEST)

    # Combine all samples into a single pool.
    full_pool = original_train + original_validation + original_test
    
    logging.info("Total samples in full pool: %d", len(full_pool))
    logging.info("  - Train: %d", len(original_train))
    logging.info("  - Validation: %d", len(original_validation))
    logging.info("  - Test: %d", len(original_test))

    # Extract unique words and speakers from the full pool.
    all_words = {entry.word for entry in full_pool}
    all_speakers = {entry.speaker_id for entry in full_pool}
    
    logging.info("Unique words in full pool: %d", len(all_words))
    logging.info("Unique speakers in full pool: %d", len(all_speakers))

    # Determine how many words and speakers to hold out for test.
    num_test_words = max(1, int(len(all_words) * TEST_WORD_FRACTION))
    num_test_speakers = max(1, int(len(all_speakers) * TEST_SPEAKER_FRACTION))
    
    logging.info(
        "Holding out for test: %d words (%.1f%%), %d speakers (%.1f%%)",
        num_test_words,
        (num_test_words / len(all_words)) * 100,
        num_test_speakers,
        (num_test_speakers / len(all_speakers)) * 100,
    )

    # Randomly select words and speakers to hold out.
    held_out_test_words = set(random.sample(sorted(all_words), num_test_words))
    held_out_test_speakers = set(random.sample(sorted(all_speakers), num_test_speakers))
    
    logging.info("Selected %d words and %d speakers for test holdout", len(held_out_test_words), len(held_out_test_speakers))

    # Partition samples into test and train/val pool.
    test_samples: list[ManifestEntry] = []
    trainval_pool: list[ManifestEntry] = []
    dropped_samples: list[ManifestEntry] = []

    for entry in full_pool:
        word_held_out = entry.word in held_out_test_words
        speaker_held_out = entry.speaker_id in held_out_test_speakers

        # Assign to test only if BOTH word and speaker are held out.
        if word_held_out and speaker_held_out:
            test_samples.append(entry)
        # Assign to train/val only if NEITHER word nor speaker is held out.
        elif not word_held_out and not speaker_held_out:
            trainval_pool.append(entry)
        # Otherwise, drop the sample (exactly one condition is held out).
        else:
            dropped_samples.append(entry)

    logging.info("Test samples: %d", len(test_samples))
    logging.info("Train/validation pool samples: %d", len(trainval_pool))
    logging.info("Dropped samples: %d", len(dropped_samples))

    # Split the train/val pool into validation (~20%) and train (~80%).
    num_validation = max(1, int(len(trainval_pool) * VALIDATION_FRACTION))
    num_train = len(trainval_pool) - num_validation

    # Shuffle the pool before splitting.
    shuffled_trainval = list(trainval_pool)
    random.shuffle(shuffled_trainval)

    validation_samples = shuffled_trainval[:num_validation]
    base_train_samples = shuffled_trainval[num_validation:]

    logging.info("Validation samples: %d", len(validation_samples))
    logging.info("Base train samples: %d", len(base_train_samples))

    # Final train manifest = base train + validation.
    final_train_samples = base_train_samples + validation_samples
    logging.info("Final train manifest (train + validation): %d", len(final_train_samples))

    # Verify disjointness.
    final_train_words = {entry.word for entry in final_train_samples}
    final_train_speakers = {entry.speaker_id for entry in final_train_samples}
    test_words = {entry.word for entry in test_samples}
    test_speakers = {entry.speaker_id for entry in test_samples}

    word_overlap = final_train_words & test_words
    speaker_overlap = final_train_speakers & test_speakers

    if word_overlap:
        raise RuntimeError(
            f"Word overlap remains between train and test: {sorted(word_overlap)[:10]}"
        )
    if speaker_overlap:
        raise RuntimeError(
            f"Speaker overlap remains between train and test: {sorted(speaker_overlap)[:10]}"
        )

    logging.info("Verified: No word overlap between train and test")
    logging.info("Verified: No speaker overlap between train and test")

    # Write manifests.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(NEW_TRAIN_MANIFEST, (entry.data for entry in final_train_samples))
    write_jsonl(NEW_VALIDATION_MANIFEST, (entry.data for entry in validation_samples))
    write_jsonl(NEW_TEST_MANIFEST, (entry.data for entry in test_samples))

    # Generate summary report.
    total_combined = len(full_pool)
    total_dropped = len(dropped_samples)
    total_dropped_pct = (total_dropped / total_combined * 100.0) if total_combined else 0.0

    report_lines = [
        "PAVSig Disjoint Split Summary",
        "=" * 60,
        "",
        "Original Manifest Counts",
        f"  Original train samples: {len(original_train)}",
        f"  Original validation samples: {len(original_validation)}",
        f"  Original test samples: {len(original_test)}",
        f"  Total combined: {total_combined}",
        "",
        "New Split Counts",
        f"  New train manifest (train + validation): {len(final_train_samples)}",
        f"  New validation manifest: {len(validation_samples)}",
        f"  New test manifest: {len(test_samples)}",
        "",
        "Dropped Samples",
        f"  Samples dropped: {total_dropped}",
        f"  Percentage of total: {total_dropped_pct:.2f}%",
        f"  (Dropped because word XOR speaker was held out, but not both)",
        "",
        "Vocabulary and Speaker Disjointness",
        f"  Unique words/forms in train: {len(final_train_words)}",
        f"  Unique words/forms in test: {len(test_words)}",
        f"  Word overlap: {'NO (verified)' if not word_overlap else 'YES (ERROR)'}",
        f"  Unique speakers in train: {len(final_train_speakers)}",
        f"  Unique speakers in test: {len(test_speakers)}",
        f"  Speaker overlap: {'NO (verified)' if not speaker_overlap else 'YES (ERROR)'}",
        "",
        "Split Configuration",
        f"  Test word holdout fraction: {TEST_WORD_FRACTION:.1%}",
        f"  Test speaker holdout fraction: {TEST_SPEAKER_FRACTION:.1%}",
        f"  Validation fraction (from train/val pool): {VALIDATION_FRACTION:.1%}",
        f"  Random seed: {RANDOM_SEED}",
        f"  Held-out test words: {num_test_words}",
        f"  Held-out test speakers: {num_test_speakers}",
        "",
        "Output Files",
        f"  Train manifest: {NEW_TRAIN_MANIFEST}",
        f"  Validation manifest: {NEW_VALIDATION_MANIFEST}",
        f"  Test manifest: {NEW_TEST_MANIFEST}",
    ]

    SUMMARY_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    logging.info("Wrote train manifest: %s (%d samples)", NEW_TRAIN_MANIFEST, len(final_train_samples))
    logging.info("Wrote validation manifest: %s (%d samples)", NEW_VALIDATION_MANIFEST, len(validation_samples))
    logging.info("Wrote test manifest: %s (%d samples)", NEW_TEST_MANIFEST, len(test_samples))
    logging.info("Wrote summary report: %s", SUMMARY_REPORT)
    logging.info("Split complete with no word or speaker overlap")


if __name__ == "__main__":
    main()