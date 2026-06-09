"""
English Baseline & Fine-Tuning Experiment — Evaluation + Absolute Bias
=======================================================================

Evaluates up to 4 model configurations on adult (LibriSpeech) and child
(MyST) test sets, then computes per-model WER and **Absolute Bias**.

    Absolute Bias = WER_child − WER_adult  (percentage points)

A positive value means the model performs worse on children; the goal of
child fine-tuning is to shrink this gap.

Models evaluated:
    1. Dense pretrained       (adult-only LibriSpeech)
    2. Dense fine-tuned       (+ LibriSpeech+MySTmix)
    3. MoE pretrained         (adult-only LibriSpeech)
    4. MoE fine-tuned         (+ LibriSpeech+MySTmix)
    5. MoE fine-tuned (LB off) (+ LibriSpeech+MySTmix, load balance loss off)

Usage
-----
    # Evaluate all four (pass .nemo files):
    python evaluate_english_experiment.py \\
        --dense_pretrained  /path/to/dense_pretrained.nemo \\
        --dense_finetuned   /path/to/dense_finetuned.nemo  \\
        --moe_pretrained    /path/to/moe_pretrained.nemo   \\
        --moe_finetuned     /path/to/moe_finetuned.nemo

    # Evaluate only what you have so far:
    python evaluate_english_experiment.py \\
        --dense_pretrained /path/to/dense.nemo

    # Override test sets:
    python evaluate_english_experiment.py \\
        --dense_pretrained /path/to/dense.nemo \\
        --adult_test /path/to/adult_test.json  \\
        --child_test /path/to/child_test.json

    # Save results to JSON:
    python evaluate_english_experiment.py ... --output results.json
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import editdistance
import torch
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils import logging


# ---------------------------------------------------------------------------
# Default test sets
# ---------------------------------------------------------------------------
DEFAULT_ADULT_TEST = "/data/librispeech/test_clean.json"
DEFAULT_CHILD_TEST = "/lp-dev/amelia/data/myst/test.json"

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """WER result on one test set."""
    dataset: str
    num_utterances: int = 0
    num_words: int = 0
    edit_distance: int = 0
    wer: float = 0.0

@dataclass
class ModelResult:
    """Full evaluation result for one model."""
    name: str
    checkpoint: str
    adult: TestResult = field(default_factory=lambda: TestResult("adult"))
    child: TestResult = field(default_factory=lambda: TestResult("child"))
    absolute_bias: float = 0.0


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
    # Keep larger numbers as digits if they appear rarely.
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
) -> TestResult:
    """
    Run inference and compute WER.

    Returns a TestResult with WER, edit distance, word count, etc.
    """
    records = load_manifest(manifest_path)
    audio_files = [r["audio_filepath"] for r in records]
    references = [r.get("text", "") for r in records]

    # Filter out empty references (some MyST entries have noise markers)
    valid = [(a, r) for a, r in zip(audio_files, references) if r.strip()]
    if len(valid) < len(audio_files):
        logging.warning(
            f"Filtered {len(audio_files) - len(valid)} empty-text entries "
            f"from {manifest_path}"
        )
    audio_files, references = zip(*valid) if valid else ([], [])

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

    wer = total_dist / total_words if total_words > 0 else float("inf")

    return TestResult(
        dataset=manifest_path,
        num_utterances=len(audio_files),
        num_words=total_words,
        edit_distance=total_dist,
        wer=wer,
    )


def evaluate_single_model(
    name: str,
    nemo_path: str,
    adult_manifest: str,
    child_manifest: str,
    batch_size: int,
    num_workers: int,
    device: str,
    text_norm: str,
) -> ModelResult:
    """Load one .nemo checkpoint and evaluate on adult, child, and YTC test sets."""
    logging.info(f"Loading model '{name}' from {nemo_path}")
    t0 = time.time()

    try:
        model = EncDecCTCModelBPE.restore_from(nemo_path, map_location=device)
    except Exception as e:
        if "key_phrase_items_list" in str(e):
            # Workaround for NeMo config version mismatch
            logging.warning(f"  Config compatibility issue detected, attempting reload with override...")
            try:
                model = EncDecCTCModelBPE.restore_from(nemo_path, map_location=device, strict=False)
            except Exception:
                logging.error(f"  Failed to load model: {e}")
                raise
        else:
            raise
    
    model.eval()
    model = model.to(device)

    logging.info(f"  Loaded in {time.time() - t0:.1f}s on {device}")

    result = ModelResult(name=name, checkpoint=nemo_path)

    # ---- Adult test ----
    if Path(adult_manifest).exists():
        logging.info(f"  Evaluating on adult test: {adult_manifest}")
        result.adult = evaluate_model_on_manifest(
            model, adult_manifest, batch_size, num_workers, text_norm
        )
        result.adult.dataset = "adult"
        logging.info(f"  Adult WER = {result.adult.wer * 100:.2f}%")
    else:
        logging.warning(f"  Adult manifest not found: {adult_manifest}")

    # ---- Child test ----
    if Path(child_manifest).exists():
        logging.info(f"  Evaluating on child test: {child_manifest}")
        result.child = evaluate_model_on_manifest(
            model, child_manifest, batch_size, num_workers, text_norm
        )
        result.child.dataset = "child"
        logging.info(f"  Child WER = {result.child.wer * 100:.2f}%")
    else:
        logging.warning(f"  Child manifest not found: {child_manifest}")

    # ---- Absolute Bias ----
    if result.adult.num_words > 0 and result.child.wer < float("inf"):
        result.absolute_bias = result.child.wer - result.adult.wer
    else:
        result.absolute_bias = float("nan")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_results_table(results: list[ModelResult]):
    """Print a formatted comparison table."""
    SEP = "=" * 90

    print(f"\n{SEP}")
    print("  ENGLISH BASELINE & FINE-TUNING — EVALUATION RESULTS")
    print(SEP)

    # Header
    print(
        f"  {'Model':<35} {'Adult WER':>10} {'Child WER':>10} "
        f"{'Abs. Bias':>10} {'Bias Δ':>10}"
    )
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Find pretrained biases for delta computation
    pretrained_bias = {}
    for r in results:
        if "pretrained" in r.name.lower():
            arch = "dense" if "dense" in r.name.lower() else "moe"
            pretrained_bias[arch] = r.absolute_bias

    import math
    for r in results:
        adult_str = f"{r.adult.wer * 100:.2f}%" if r.adult.num_words > 0 else "—"
        child_str = f"{r.child.wer * 100:.2f}%" if r.child.num_words > 0 else "—"

        if math.isnan(r.absolute_bias):
            bias_str = "—"
        else:
            bias_str = f"{r.absolute_bias * 100:+.2f} pp"

        # Delta = how much bias changed after fine-tuning
        delta_str = ""
        if "finetuned" in r.name.lower():
            arch = "dense" if "dense" in r.name.lower() else "moe"
            if arch in pretrained_bias and not math.isnan(pretrained_bias[arch]):
                delta = r.absolute_bias - pretrained_bias[arch]
                delta_str = f"{delta * 100:+.2f} pp"

        print(
            f"  {r.name:<35} {adult_str:>10} {child_str:>10} "
            f"{bias_str:>10} {delta_str:>10}"
        )

    print(SEP)

    # Interpretation
    print("\n  Interpretation:")
    print("    • Absolute Bias = Child WER − Adult WER (pp)")
    print("    • Positive bias → model is worse on children")
    print("    • Bias Δ (pp) → change in absolute bias after fine-tuning")
    print("    • Negative Δ → fine-tuning reduced the adult–child gap")
    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate English ASR models: adult vs child WER + absolute bias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model checkpoints (all optional — evaluate whichever you have)
    p.add_argument("--dense_pretrained", type=str, default=None,
                   help="Path to dense Conformer pretrained .nemo")
    p.add_argument("--dense_finetuned", type=str, default=None,
                   help="Path to dense Conformer fine-tuned .nemo")
    p.add_argument("--moe_pretrained", type=str, default=None,
                   help="Path to MoE Conformer pretrained .nemo")
    p.add_argument("--moe_finetuned", type=str, default=None,
                   help="Path to MoE Conformer fine-tuned .nemo")
    p.add_argument("--moe_finetuned_lb_off", type=str, default=None,
                   help="Path to MoE Conformer fine-tuned (.nemo) with load-balance loss off")

    # Test data
    p.add_argument("--adult_test", type=str, default=DEFAULT_ADULT_TEST,
                   help="Adult test manifest (LibriSpeech test-clean)")
    p.add_argument("--child_test", type=str, default=DEFAULT_CHILD_TEST,
                   help="Child test manifest (MyST test)")

    # Inference settings
    p.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument(
        "--text_norm",
        type=str,
        default="basic_num",
        choices=["raw", "basic", "basic_num"],
        help="Reference/hypothesis normalization for WER: raw | basic | basic_num",
    )

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Save results to this JSON file")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Collect which models to evaluate
    models_to_eval = []
    if args.dense_pretrained:
        models_to_eval.append(("Dense Pretrained (LibriSpeech)", args.dense_pretrained))
    if args.dense_finetuned:
        models_to_eval.append(("Dense Fine-tuned (+ LibriSpeech+MySTmix)", args.dense_finetuned))
    if args.moe_pretrained:
        models_to_eval.append(("MoE Pretrained (LibriSpeech)", args.moe_pretrained))
    if args.moe_finetuned:
        models_to_eval.append(("MoE Fine-tuned (+ LibriSpeech+MySTmix)", args.moe_finetuned))
    if args.moe_finetuned_lb_off:
        models_to_eval.append(
            ("MoE Fine-tuned LB off (+ LibriSpeech+MySTmix)", args.moe_finetuned_lb_off)
        )

    if not models_to_eval:
        print("ERROR: Provide at least one model checkpoint.")
        print("  --dense_pretrained /path/to/dense.nemo")
        print("  --moe_pretrained   /path/to/moe.nemo")
        sys.exit(1)

    # Verify test manifests
    for label, path in [("Adult (LibriSpeech)", args.adult_test), ("Child (MyST)", args.child_test)]:
        if Path(path).exists():
            n = sum(1 for l in open(path) if l.strip())
            print(f"  ✓ {label} test: {path}  ({n:,} utterances)")
        else:
            print(f"  ✗ {label} test not found: {path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}\n")
    print(f"  Text normalization: {args.text_norm}")

    # Evaluate each model
    all_results = []
    for name, ckpt_path in models_to_eval:
        if not Path(ckpt_path).exists():
            logging.warning(f"Checkpoint not found, skipping '{name}': {ckpt_path}")
            continue

        result = evaluate_single_model(
            name=name,
            nemo_path=ckpt_path,
            adult_manifest=args.adult_test,
            child_manifest=args.child_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            text_norm=args.text_norm,
        )
        all_results.append(result)

    # Print comparison table
    if all_results:
        print_results_table(all_results)

    # Save to JSON
    if args.output and all_results:
        out = {
            "experiment": "English Baseline & Fine-Tuning — Absolute Bias",
            "adult_test": args.adult_test,
            "child_test": args.child_test,
            "models": [asdict(r) for r in all_results],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
