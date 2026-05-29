


"""
Quick single-model WER evaluation on MyST (child) and LibriSpeech (adult) test sets.

Simpler than evaluate_english.py — no JSON output, just prints WER to stdout.
Useful for spot-checking a single checkpoint during development.

Usage:
    python evaluation/evaluate.py --ckpt /path/to/model.nemo
    python evaluation/evaluate.py --ckpt /path/to/model.nemo --batch_size 32
"""

import argparse
import json
from pathlib import Path

import editdistance
import torch
from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.utils import logging


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CKPT = "/lp-dev/amelia/inclusive-asr-moe/final_weights/en_child_moe_lb_off.nemo"

DATASETS = {
    "child_speech (MyST)": "/lp-dev/amelia/data/myst/test.json",
    "adult_speech (LibriSpeech test-clean)": "/data/librispeech/test_clean.json",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_manifest(path: str) -> list[dict]:
    """Load a NeMo-style JSONL manifest (one JSON object per line)."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_wer(
    model: EncDecCTCModelBPE,
    manifest_path: str,
    batch_size: int,
    num_workers: int,
) -> tuple[float, int, int]:
    """
    Run inference on every utterance in *manifest_path* and compute WER.

    Returns:
        wer            – word error rate as a fraction (e.g. 0.15 = 15 %)
        total_dist     – summed edit distances (numerator)
        total_words    – summed reference word counts (denominator)
    """
    records    = load_manifest(manifest_path)
    audio_files = [r["audio_filepath"] for r in records]
    references  = [r.get("text", "")   for r in records]

    hypotheses = model.transcribe(
        audio_files,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=False,
    )

    total_dist  = 0
    total_words = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_text     = hyp.text if hasattr(hyp, "text") else str(hyp)
        total_dist  += editdistance.eval(hyp_text.split(), ref.split())
        total_words += len(ref.split())

    wer = total_dist / total_words if total_words > 0 else float("inf")
    return wer, total_dist, total_words


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate NeMo CTC-BPE checkpoint on test sets")
    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
        help="Path to a .ckpt file (default: best Conformer-CTC-BPE checkpoint)",
    )
    p.add_argument("--batch_size",  type=int, default=16,  help="Inference batch size")
    p.add_argument("--num_workers", type=int, default=4,   help="DataLoader workers")
    p.add_argument(
        "--datasets",
        nargs="*",
        metavar="NAME:PATH",
        help=(
            "Additional manifests to evaluate. Format: name:path, e.g. "
            "--datasets custom_set:/path/to/manifest.json"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # merge built-in datasets with any extra ones supplied on the CLI
    datasets = dict(DATASETS)
    if args.datasets:
        for entry in args.datasets:
            name, path = entry.split(":", 1)
            datasets[name] = path

    # ---- load model --------------------------------------------------------
    logging.info(f"Loading checkpoint: {args.ckpt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = EncDecCTCModelBPE.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model  = model.to(device)
    logging.info(f"Model loaded on {device}")

    # ---- evaluate ----------------------------------------------------------
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  Checkpoint : {Path(args.ckpt).name}")
    print(f"  Device     : {device}")
    print(SEP)

    grand_dist  = 0
    grand_words = 0

    for name, path in datasets.items():
        if not Path(path).exists():
            logging.warning(f"Manifest not found – skipping: {path}")
            continue

        n_utts = sum(1 for line in open(path) if line.strip())
        logging.info(f"Evaluating  '{name}'  ({n_utts:,} utterances) …")

        wer, dist, words = compute_wer(model, path, args.batch_size, args.num_workers)
        grand_dist  += dist
        grand_words += words

        print(f"\n  [{name}]")
        print(f"    Utterances : {n_utts:>8,}")
        print(f"    Words      : {words:>8,}")
        print(f"    WER        : {wer * 100:>7.2f} %")

    # ---- overall pooled WER ------------------------------------------------
    overall_wer = grand_dist / grand_words if grand_words > 0 else float("inf")

    print(f"\n{SEP}")
    print(
        f"  OVERALL WER : {overall_wer * 100:.2f} %"
        f"   (words: {grand_words:,}  |  datasets: {len(datasets)})"
    )
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()

