#!/usr/bin/env python3
"""Fix audio format issues in PAVSig manifests for Lhotse/NeMo training.

Ensures all audio files are:
- Readable and valid
- Mono (1-D, single channel)
- Compatible with Lhotse expectations

Converts multi-channel audio to mono by averaging channels while preserving
the original sampling rate. Writes updated manifests with corrected paths.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import soundfile as sf


# Configuration
MANIFEST_DIR = Path("/lp-dev/amelia/data/pavsig/training")
INPUT_TRAIN_MANIFEST = MANIFEST_DIR / "new_train_manifest.jsonl"
INPUT_VAL_MANIFEST = MANIFEST_DIR / "new_val_manifest.jsonl"
INPUT_TEST_MANIFEST = MANIFEST_DIR / "new_test_manifest.jsonl"

OUTPUT_AUDIO_DIR = Path("/lp-dev/amelia/data/pavsig/training/mono_audio")
OUTPUT_TRAIN_MANIFEST = MANIFEST_DIR / "new_train_manifest.mono.jsonl"
OUTPUT_VAL_MANIFEST = MANIFEST_DIR / "new_val_manifest.mono.jsonl"
OUTPUT_TEST_MANIFEST = MANIFEST_DIR / "new_test_manifest.mono.jsonl"
REPORT_PATH = MANIFEST_DIR / "audio_fix_report.txt"

AUDIO_FIELD = "audio_filepath"


@dataclass
class AudioStats:
    """Statistics about audio processing."""
    total_files: int = 0
    files_already_mono: int = 0
    files_converted_to_mono: int = 0
    files_failed: int = 0
    files_read_error: int = 0
    
    # Channel distribution
    channel_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Sampling rate distribution
    sample_rates: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    
    # Track failures
    failed_files: list[tuple[str, str]] = field(default_factory=list)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_manifest(path: Path) -> list[dict]:
    """Read a JSONL manifest file."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    
    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e
    return entries


def write_manifest(path: Path, entries: list[dict]) -> None:
    """Write entries to a JSONL manifest file."""
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")


def get_audio_path(sample: dict) -> Optional[str]:
    """Extract audio path from sample."""
    path = sample.get(AUDIO_FIELD, "").strip()
    return path if path else None


def process_audio_file(
    audio_path: str,
    output_dir: Path,
    stats: AudioStats,
) -> Optional[str]:
    """
    Process a single audio file. 
    
    Returns the path to the (possibly converted) mono audio file,
    or None if the file failed to process.
    """
    stats.total_files += 1
    
    # Check if file exists
    if not Path(audio_path).exists():
        error_msg = f"File not found: {audio_path}"
        logging.warning("Sample %d: %s", stats.total_files, error_msg)
        stats.files_failed += 1
        stats.failed_files.append((audio_path, error_msg))
        return None
    
    # Try to read audio
    try:
        data, sr = sf.read(audio_path, dtype="float32")
    except Exception as e:
        error_msg = f"Failed to read audio: {str(e)}"
        logging.warning("Sample %d from %s: %s", stats.total_files, audio_path, error_msg)
        stats.files_read_error += 1
        stats.files_failed += 1
        stats.failed_files.append((audio_path, error_msg))
        return None
    
    # Detect shape and channels
    if data.ndim == 1:
        # Already mono
        num_channels = 1
        num_samples = len(data)
        logging.debug("Sample %d: Already mono (sr=%d, samples=%d)", stats.total_files, sr, num_samples)
        stats.files_already_mono += 1
        stats.channel_counts[1] += 1
        stats.sample_rates[sr] += 1
        return audio_path
    
    elif data.ndim == 2:
        # Multi-channel: shape is (samples, channels)
        num_samples, num_channels = data.shape
        logging.debug(
            "Sample %d: Multi-channel (sr=%d, samples=%d, channels=%d)",
            stats.total_files, sr, num_samples, num_channels,
        )
        stats.channel_counts[num_channels] += 1
        stats.sample_rates[sr] += 1
        
        # Convert to mono by averaging channels
        mono_data = data.mean(axis=1).astype("float32")
        
        # Generate output path
        source_path = Path(audio_path)
        output_filename = f"{stats.total_files:06d}_{source_path.stem}.wav"
        output_path = output_dir / output_filename
        
        # Write mono audio
        try:
            sf.write(str(output_path), mono_data, sr)
            logging.debug("Wrote mono audio to: %s", output_path)
            stats.files_converted_to_mono += 1
            return str(output_path)
        except Exception as e:
            error_msg = f"Failed to write mono audio: {str(e)}"
            logging.error("Sample %d: %s", stats.total_files, error_msg)
            stats.files_failed += 1
            stats.failed_files.append((audio_path, error_msg))
            return None
    
    else:
        error_msg = f"Unexpected audio shape: {data.shape} (expected 1D or 2D)"
        logging.warning("Sample %d from %s: %s", stats.total_files, audio_path, error_msg)
        stats.files_failed += 1
        stats.failed_files.append((audio_path, error_msg))
        return None


def process_manifest(
    manifest_path: Path,
    output_dir: Path,
    stats: AudioStats,
) -> list[dict]:
    """
    Process all audio files in a manifest.
    
    Returns the updated manifest entries with corrected audio paths.
    """
    entries = read_manifest(manifest_path)
    updated_entries: list[dict] = []
    
    for sample in entries:
        audio_path = get_audio_path(sample)
        if not audio_path:
            logging.warning("Sample missing audio_filepath: %s", sample)
            continue
        
        # Process the audio file
        new_audio_path = process_audio_file(audio_path, output_dir, stats)
        
        if new_audio_path is None:
            # Skip this sample if audio processing failed
            continue
        
        # Update sample with new audio path (if changed)
        updated_sample = dict(sample)
        updated_sample[AUDIO_FIELD] = new_audio_path
        updated_entries.append(updated_sample)
    
    return updated_entries


def main() -> None:
    configure_logging()
    logging.info("Starting audio format fix for PAVSig manifests")
    
    # Create output directory
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Output audio directory: %s", OUTPUT_AUDIO_DIR)
    
    # Process each manifest
    stats = AudioStats()
    
    logging.info("Processing train manifest...")
    train_entries = process_manifest(INPUT_TRAIN_MANIFEST, OUTPUT_AUDIO_DIR, stats)
    logging.info("Train manifest: %d entries (after filtering failures)", len(train_entries))
    
    logging.info("Processing validation manifest...")
    val_entries = process_manifest(INPUT_VAL_MANIFEST, OUTPUT_AUDIO_DIR, stats)
    logging.info("Validation manifest: %d entries (after filtering failures)", len(val_entries))
    
    logging.info("Processing test manifest...")
    test_entries = process_manifest(INPUT_TEST_MANIFEST, OUTPUT_AUDIO_DIR, stats)
    logging.info("Test manifest: %d entries (after filtering failures)", len(test_entries))
    
    # Write updated manifests
    logging.info("Writing updated manifests...")
    write_manifest(OUTPUT_TRAIN_MANIFEST, train_entries)
    write_manifest(OUTPUT_VAL_MANIFEST, val_entries)
    write_manifest(OUTPUT_TEST_MANIFEST, test_entries)
    
    logging.info("Wrote train manifest: %s", OUTPUT_TRAIN_MANIFEST)
    logging.info("Wrote validation manifest: %s", OUTPUT_VAL_MANIFEST)
    logging.info("Wrote test manifest: %s", OUTPUT_TEST_MANIFEST)
    
    # Generate report
    report_lines = [
        "PAVSig Audio Format Fix Report",
        "=" * 70,
        "",
        "Processing Summary",
        f"  Total audio files processed: {stats.total_files}",
        f"  Files already mono: {stats.files_already_mono}",
        f"  Files converted to mono: {stats.files_converted_to_mono}",
        f"  Files failed: {stats.files_failed}",
        f"    - Read errors: {stats.files_read_error}",
        f"    - Write errors: {stats.files_failed - stats.files_read_error}",
        "",
        "Channel Count Distribution (Before Fix)",
        "  Channel Count | Number of Files",
        "  " + "-" * 50,
    ]
    
    for channels in sorted(stats.channel_counts.keys()):
        count = stats.channel_counts[channels]
        report_lines.append(f"  {channels:>14} | {count:>15}")
    
    report_lines.extend([
        "",
        "Sampling Rate Distribution",
        "  Sampling Rate (Hz) | Number of Files",
        "  " + "-" * 50,
    ])
    
    for sr in sorted(stats.sample_rates.keys()):
        count = stats.sample_rates[sr]
        report_lines.append(f"  {sr:>18} | {count:>15}")
    
    report_lines.extend([
        "",
        "Manifest Information",
        f"  Input train manifest: {INPUT_TRAIN_MANIFEST}",
        f"  Output train manifest: {OUTPUT_TRAIN_MANIFEST}",
        f"  Output train entries: {len(train_entries)}",
        "",
        f"  Input validation manifest: {INPUT_VAL_MANIFEST}",
        f"  Output validation manifest: {OUTPUT_VAL_MANIFEST}",
        f"  Output validation entries: {len(val_entries)}",
        "",
        f"  Input test manifest: {INPUT_TEST_MANIFEST}",
        f"  Output test manifest: {OUTPUT_TEST_MANIFEST}",
        f"  Output test entries: {len(test_entries)}",
        "",
        f"  Output audio directory: {OUTPUT_AUDIO_DIR}",
    ])
    
    if stats.failed_files:
        report_lines.extend([
            "",
            "Failed Files",
            "  Path | Error",
            "  " + "-" * 70,
        ])
        for path, error in stats.failed_files[:50]:  # Show first 50 failures
            report_lines.append(f"  {path} | {error}")
        if len(stats.failed_files) > 50:
            report_lines.append(f"  ... and {len(stats.failed_files) - 50} more failures")
    
    report_text = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    
    logging.info("Wrote report: %s", REPORT_PATH)
    logging.info("Audio format fix complete!")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print(report_text)
    print("=" * 70)


if __name__ == "__main__":
    main()
