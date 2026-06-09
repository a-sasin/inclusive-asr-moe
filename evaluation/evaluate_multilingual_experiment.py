"""
Multilingual & Child-Speech Experiment — Per-Language WER Evaluation
=====================================================================

Evaluates a model on multiple language test sets with per-language WER breakdown.

Supported languages/datasets:
    • Polish (PL)       → /data/cv/nemo/pl/test.json
    • Dutch (NL)        → /data/cv/nemo/nl/test.json
    • German (DE)       → /data/cv/nemo/de/test.json
    • English (EN)      → /data/librispeech_nemo/test_clean.json
    • English (Multilingual trained) → varies
    • MyST (child EN)    → /lp-dev/amelia/data/myst/test.json
    • JASMIN (child NL)  → /lp-dev/amelia/data/jasmin/child/test.json
    • KidsTalc (child EN)→ /lp-dev/amelia/data/kidstalc/cleaned/val.json (used as test)
    • PAVSig (child PL)  → /lp-dev/amelia/data/pavsig/test.json

Usage
-----
    # Evaluate on all available languages:
    python evaluate_multilingual_experiment.py \\
        --checkpoint /path/to/model.nemo

        /lp-dev/amelia/inclusive-asr-moe/experiments/NEW/multilingual/moe/moe_fastconformer_multilingual_2026-04-22_13-48-13/2026-04-22_13-48-26/checkpoints/moe_fastconformer_multilingual_2026-04-22_13-48-13.nemo
    
    # Evaluate on specific languages only:
    python evaluate_multilingual_experiment.py \\
        --checkpoint /path/to/model.nemo \\
        --languages pl de en_librispeech child_en
    
    # Use custom test set paths:
    python evaluate_multilingual_experiment.py \\
        --checkpoint /path/to/model.nemo \\
        --pl_test /custom/pl_test.json \\
        --en_test /custom/en_test.json
    
    # Save results to JSON:
    python evaluate_multilingual_experiment.py \\
        --checkpoint /path/to/model.nemo \\
        --output results.json
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List

import editdistance
import torch
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils import logging


# ---------------------------------------------------------------------------
# Default test sets by language/dataset
# ---------------------------------------------------------------------------
DEFAULT_TEST_SETS = {
    "pl": "/data/cv/nemo/pl/test.json",
    "nl": "/data/cv/nemo/nl/test.json",
    "de": "/data/cv/nemo/de/test.json",
    "en_librispeech": "/data/librispeech_nemo/test_clean.json",
    "child_en_myst": "/lp-dev/amelia/data/myst/test.json",
    "child_nl_jasmin": "/lp-dev/amelia/data/jasmin/child/test.json",
    "child_en_kidstalc": "/lp-dev/amelia/data/kidstalc/cleaned/val.json",
    "child_pl_pavsig": "/lp-dev/amelia/data/pavsig/training/new_test_manifest.mono.jsonl",
}

LANGUAGE_LABELS = {
    "pl": "Polish (PL)",
    "nl": "Dutch (NL)",
    "de": "German (DE)",
    "en_librispeech": "English—LibriSpeech (EN)",
    "child_en_myst": "English—Child MyST",
    "child_nl_jasmin": "Dutch—Child JASMIN",
    "child_en_kidstalc": "English—Child KidsTalc",
    "child_pl_pavsig": "Polish—Child PAVSig",
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """WER result on one test set."""
    language: str
    dataset: str
    num_utterances: int = 0
    num_words: int = 0
    edit_distance: int = 0
    wer: float = 0.0


@dataclass
class ModelResult:
    """Full evaluation result for one model."""
    checkpoint: str
    languages: List[TestResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> list[dict]:
    """Load NeMo JSONL manifest."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def int_to_english_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t = (n // 10) * 10
        r = n % 10
        return _TENS[t] if r == 0 else f"{_TENS[t]} {_ONES[r]}"
    if n < 1000:
        h = n // 100
        r = n % 100
        return f"{_ONES[h]} hundred" if r == 0 else f"{_ONES[h]} hundred {int_to_english_words(r)}"
    if n < 10000:
        th = n // 1000
        r = n % 1000
        return f"{_ONES[th]} thousand" if r == 0 else f"{_ONES[th]} thousand {int_to_english_words(r)}"
    return str(n)


def normalize_text_for_wer(text: str, mode: str) -> str:
    if mode == "raw":
        return text.strip()

    t = text.lower().strip()
    # Remove common annotation/noise markers in child speech transcripts.
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\+[^+]*\+", " ", t)

    if mode == "basic_num":
        def repl_num(m: re.Match) -> str:
            try:
                return int_to_english_words(int(m.group(0)))
            except Exception:
                return m.group(0)

        t = re.sub(r"\b\d+\b", repl_num, t)

    # Keep letters, digits, apostrophes and spaces; normalize everything else to spaces.
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def evaluate_model_on_manifest(
    model: EncDecCTCModelBPE,
    manifest_path: str,
    batch_size: int = 16,
    num_workers: int = 4,
    text_norm: str = "basic_num",
) -> tuple[int, int, int]:
    """
    Run inference and compute WER components.
    
    Returns: (num_utterances, total_words, total_edit_distance)
    """
    records = load_manifest(manifest_path)
    audio_files = [r["audio_filepath"] for r in records]
    references = [r.get("text", "") for r in records]

    # Filter out empty references
    valid = [(a, r) for a, r in zip(audio_files, references) if r.strip()]
    if len(valid) < len(audio_files):
        logging.warning(
            f"Filtered {len(audio_files) - len(valid)} empty-text entries "
            f"from {manifest_path}"
        )
    audio_files, references = zip(*valid) if valid else ([], [])

    if not audio_files:
        logging.warning(f"No valid utterances in {manifest_path}")
        return 0, 0, 0

    hypotheses = model.transcribe(
        list(audio_files),
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=False,
    )

    total_dist = 0
    total_words = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_text = hyp.text if hasattr(hyp, "text") else str(hyp)
        ref_norm = normalize_text_for_wer(ref, text_norm)
        hyp_norm = normalize_text_for_wer(hyp_text, text_norm)
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()
        total_dist += editdistance.eval(hyp_words, ref_words)
        total_words += len(ref_words)

    return len(audio_files), total_words, total_dist


def evaluate_model_on_languages(
    checkpoints: str,
    language_test_sets: Dict[str, str],
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda",
    text_norm: str = "basic_num",
) -> ModelResult:
    """
    Load a model and evaluate on multiple language test sets.
    
    Args:
        checkpoints: Path to .nemo checkpoint
        language_test_sets: dict of {language_key: test_manifest_path}
        batch_size: inference batch size
        num_workers: DataLoader workers
        device: 'cuda' or 'cpu'
        text_norm: normalization mode for WER
    
    Returns:
        ModelResult with per-language WER
    """
    logging.info(f"Loading model from {checkpoints}")
    t0 = time.time()

    try:
        model = EncDecCTCModelBPE.restore_from(checkpoints, map_location=device)
    except Exception as e:
        if "key_phrase_items_list" in str(e):
            logging.warning(f"  Config compatibility issue detected, attempting reload with override...")
            try:
                model = EncDecCTCModelBPE.restore_from(checkpoints, map_location=device, strict=False)
            except Exception:
                logging.error(f"  Failed to load model: {e}")
                raise
        else:
            raise

    model.eval()
    model = model.to(device)
    logging.info(f"  Loaded in {time.time() - t0:.1f}s on {device}")

    result = ModelResult(checkpoint=checkpoints)

    for lang_key, test_path in sorted(language_test_sets.items()):
        if not Path(test_path).exists():
            logging.warning(f"  Test set not found: {test_path} ({lang_key})")
            continue

        logging.info(f"  Evaluating {LANGUAGE_LABELS.get(lang_key, lang_key)}: {test_path}")

        num_utt, num_words, edit_dist = evaluate_model_on_manifest(
            model, test_path, batch_size, num_workers, text_norm
        )

        if num_words == 0:
            logging.warning(f"    No valid utterances in {test_path}")
            wer = float("inf")
        else:
            wer = edit_dist / num_words

        test_result = TestResult(
            language=lang_key,
            dataset=test_path,
            num_utterances=num_utt,
            num_words=num_words,
            edit_distance=edit_dist,
            wer=wer,
        )
        result.languages.append(test_result)
        logging.info(f"    WER = {wer * 100:.2f}% ({edit_dist}/{num_words} errors)")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_results_table(result: ModelResult):
    """Print a formatted per-language WER table."""
    SEP = "=" * 95

    print(f"\n{SEP}")
    print("  MULTILINGUAL & CHILD-SPEECH — PER-LANGUAGE WER EVALUATION")
    print(SEP)

    # Header
    print(
        f"  {'Language / Dataset':<40} {'WER':>10} {'Utterances':>12} {'Words':>10}"
    )
    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10}")

    # Results
    total_words = 0
    total_edit_dist = 0
    
    for test_res in result.languages:
        label = LANGUAGE_LABELS.get(test_res.language, test_res.language)
        import math
        if math.isinf(test_res.wer):
            wer_str = "—"
        else:
            wer_str = f"{test_res.wer * 100:.2f}%"

        print(
            f"  {label:<40} {wer_str:>10} {test_res.num_utterances:>12,} {test_res.num_words:>10,}"
        )
        
        if not math.isinf(test_res.wer):
            total_words += test_res.num_words
            total_edit_dist += test_res.edit_distance

    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10}")
    
    # Overall WER
    if total_words > 0:
        overall_wer = total_edit_dist / total_words
        print(
            f"  {'OVERALL':<40} {overall_wer * 100:>10.2f}% {'-':>12} {total_words:>10,}"
        )

    print(SEP + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multilingual ASR: per-language WER on adult & child test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model checkpoint
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model .nemo checkpoint")

    # Language selection
    p.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=list(DEFAULT_TEST_SETS.keys()),
        help="Languages to evaluate (default: all available)"
    )

    # Custom test paths (override defaults)
    p.add_argument("--pl_test", type=str, default=None, help="Polish test manifest")
    p.add_argument("--nl_test", type=str, default=None, help="Dutch test manifest")
    p.add_argument("--de_test", type=str, default=None, help="German test manifest")
    p.add_argument("--en_librispeech_test", type=str, default=None, help="English LibriSpeech test")
    p.add_argument("--child_en_myst_test", type=str, default=None, help="Child EN MyST test")
    p.add_argument("--child_nl_jasmin_test", type=str, default=None, help="Child NL JASMIN test")
    p.add_argument("--child_en_kidstalc_test", type=str, default=None, help="Child EN KidsTalc test (val)")
    p.add_argument("--child_pl_pavsig_test", type=str, default=None, help="Child PL PAVSig test")

    # Inference settings
    p.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument(
        "--text_norm",
        type=str,
        default="basic_num",
        choices=["raw", "basic", "basic_num"],
        help="Reference/hypothesis normalization: raw | basic | basic_num",
    )

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Save results to JSON file")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Build language test set map with overrides
    lang_test_sets = dict(DEFAULT_TEST_SETS)
    
    overrides = {
        "pl": args.pl_test,
        "nl": args.nl_test,
        "de": args.de_test,
        "en_librispeech": args.en_librispeech_test,
        "child_en_myst": args.child_en_myst_test,
        "child_nl_jasmin": args.child_nl_jasmin_test,
        "child_en_kidstalc": args.child_en_kidstalc_test,
        "child_pl_pavsig": args.child_pl_pavsig_test,
    }
    for lang, override in overrides.items():
        if override:
            lang_test_sets[lang] = override

    # Select languages to evaluate
    selected_langs = {lang: lang_test_sets[lang] for lang in args.languages if lang in lang_test_sets}
    
    if not selected_langs:
        print("ERROR: No valid languages selected.")
        print(f"  Available: {', '.join(DEFAULT_TEST_SETS.keys())}")
        sys.exit(1)

    # Verify checkpoint
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Verify test sets
    print("\n  Test Sets:")
    for lang, path in selected_langs.items():
        if Path(path).exists():
            n = sum(1 for l in open(path) if l.strip())
            label = LANGUAGE_LABELS.get(lang, lang)
            print(f"    ✓ {label:<40} {n:>6,} utterances")
        else:
            print(f"    ✗ {lang}: {path} (NOT FOUND)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    print(f"  Text normalization: {args.text_norm}\n")

    # Evaluate
    result = evaluate_model_on_languages(
        checkpoints=args.checkpoint,
        language_test_sets=selected_langs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        text_norm=args.text_norm,
    )

    # Print results
    print_results_table(result)

    # Save JSON
    if args.output and result.languages:
        out = {
            "experiment": "Multilingual & Child-Speech Per-Language WER",
            "checkpoint": args.checkpoint,
            "text_normalization": args.text_norm,
            "languages": [asdict(r) for r in result.languages],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logging.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
