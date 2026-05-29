#!/usr/bin/env python3
"""
Compile evaluation results into a formatted per-language WER table.

Reads all JSON results and produces a comprehensive table matching the LaTeX format.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


LANG_ORDER = ["en_librispeech", "nl", "de", "pl"]
CHILD_LANG_MAP = {
    "en_librispeech": "child_en_myst",
    "nl": "child_nl_jasmin",
    "de": "child_en_kidstalc",
    "pl": "child_pl_pavsig",
}

LANG_LABELS = {
    "en_librispeech": "EN",
    "nl": "NL",
    "de": "DE",
    "pl": "PL",
}

DATASET_NAMES = {
    "en_librispeech": "LibriSpeech",
    "nl": "CommonVoice",
    "de": "CommonVoice",
    "pl": "CommonVoice",
}

MODEL_GROUPS = {
    "dense_adult": ("Dense (adult)", "Adult"),
    "moe_adult": ("MoE (adult)", "Adult"),
    "dense_child": ("Dense\u2080 (child)", "Child"),
    "moe_child_lb_off": ("MoE\u2080 (child, LB off)", "Child"),
    "moe_child_lb_on": ("MoE\u2080 (child, LB on)", "Child"),
}


def load_result(json_path: Path) -> Optional[Dict]:
    """Load a result JSON file."""
    if not json_path.exists():
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {json_path}: {e}")
        return None


def extract_wer(result: Optional[Dict], lang_key: str) -> Optional[float]:
    """Extract WER for a language from result dict."""
    if not result or "languages" not in result:
        return None
    
    for lang_result in result["languages"]:
        if lang_result["language"] == lang_key:
            wer = lang_result.get("wer")
            if wer is not None and wer != float('inf'):
                return wer
    return None


def format_wer(wer: Optional[float]) -> str:
    """Format WER as percentage string."""
    if wer is None or wer == float('inf'):
        return "--"
    return f"{wer * 100:.2f}"


def compute_bias(adult_wer: Optional[float], child_wer: Optional[float]) -> str:
    """Compute absolute bias (child - adult) in percentage points."""
    if adult_wer is None or child_wer is None:
        return "--"
    if adult_wer == float('inf') or child_wer == float('inf'):
        return "--"
    bias_pp = (child_wer - adult_wer) * 100  # Convert to pp
    return f"{bias_pp:+.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Compile evaluation results into per-language WER table"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/lp-dev/amelia/inclusive-asr-moe/results/multilingual",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save table to text file (default: print to stdout)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print("Loading results...")
    
    # Load all model results
    all_results = {}
    for model_key, (model_display, _) in MODEL_GROUPS.items():
        json_path = results_dir / f"{model_key}_results.json"
        result = load_result(json_path)
        if result:
            all_results[model_key] = result
            print(f"  ✓ {model_key}")
        else:
            print(f"  ✗ {model_key} (not found or corrupted)")

    if not all_results:
        print("\nError: No valid results found")
        return 1

    # Build table
    lines = []
    lines.append("=" * 110)
    lines.append("MULTILINGUAL & CHILD-SPEECH EVALUATION — PER-LANGUAGE WER")
    lines.append("=" * 110)
    lines.append("")
    lines.append(f"{'Model':<45} {'Lvl.':<6} {'EN':<15} {'NL':<15} {'DE':<15} {'PL†':<15}")
    lines.append(f"{'':45} {'':6} {'':15} {'':15} {'':15}")
    lines.append(f"{'':45} {'':6} {'WER (%)':<8}{'Bias':<7} {'WER (%)':<8}{'Bias':<7} {'WER (%)':<8}{'Bias':<7} {'WER (%)':<8}{'Bias':<7}")
    lines.append("-" * 110)

    # Print adult-trained models
    adult_models = [k for k, (_, g) in MODEL_GROUPS.items() if g == "Adult"]
    for model_key in adult_models:
        if model_key not in all_results:
            continue
        
        result = all_results[model_key]
        model_display, _ = MODEL_GROUPS[model_key]
        
        # Row 1: Adult (first level)
        line_parts = [f"{model_display:<45} {'Adult':<6}"]
        for lang_key in LANG_ORDER:
            adult_wer = extract_wer(result, lang_key)
            child_wer = extract_wer(result, CHILD_LANG_MAP[lang_key])
            wer_str = format_wer(adult_wer)
            bias_str = compute_bias(adult_wer, child_wer)
            line_parts.append(f"{wer_str:<8}{bias_str:<7}")
        lines.append("".join(line_parts))
        
        # Row 2: Child
        line_parts = [f"{'':45} {'Child':<6}"]
        for lang_key in LANG_ORDER:
            child_wer = extract_wer(result, CHILD_LANG_MAP[lang_key])
            wer_str = format_wer(child_wer)
            line_parts.append(f"{wer_str:<8}{'':7}")
        lines.append("".join(line_parts))
        
        lines.append("")

    lines.append("-" * 110)
    lines.append("")

    # Print child-finetuned models
    child_models = [k for k, (_, g) in MODEL_GROUPS.items() if g == "Child"]
    for model_key in child_models:
        if model_key not in all_results:
            continue
        
        result = all_results[model_key]
        model_display, _ = MODEL_GROUPS[model_key]
        
        # Row 1: Adult
        line_parts = [f"{model_display:<45} {'Adult':<6}"]
        for lang_key in LANG_ORDER:
            adult_wer = extract_wer(result, lang_key)
            child_wer = extract_wer(result, CHILD_LANG_MAP[lang_key])
            wer_str = format_wer(adult_wer)
            bias_str = compute_bias(adult_wer, child_wer)
            line_parts.append(f"{wer_str:<8}{bias_str:<7}")
        lines.append("".join(line_parts))
        
        # Row 2: Child
        line_parts = [f"{'':45} {'Child':<6}"]
        for lang_key in LANG_ORDER:
            child_wer = extract_wer(result, CHILD_LANG_MAP[lang_key])
            wer_str = format_wer(child_wer)
            line_parts.append(f"{wer_str:<8}{'':7}")
        lines.append("".join(line_parts))
        
        lines.append("")

    lines.append("=" * 110)
    lines.append("")
    lines.append("Note: † PAVSig contains pathological (sigmatism) speech and is not directly comparable.")
    lines.append("      'Bias' columns show absolute difference (Child WER - Adult WER) in percentage points.")
    lines.append("")

    table_text = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(table_text)
        print(f"\nTable saved to: {args.output}")
    else:
        print("\n" + table_text)

    return 0


if __name__ == "__main__":
    exit(main())
