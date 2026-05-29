#!/usr/bin/env python3
"""Compute per-utterance duration statistics per corpus/language from YAML manifests.

Usage:
  python utterance_stats.py /data/file1.yaml [/data/file2.yaml ...]

For each (corpus, language) pair found in the YAML, it expands the sharded
manifest paths, reads every JSONL manifest, and reports:
  count, total_hours, mean_sec, median_sec, std_sec, min_sec, max_sec, p5_sec, p95_sec

Output is printed as a Markdown table and also saved to utterance_stats.txt
next to this script.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from statistics import median, mean, stdev
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(HERE, "utterance_stats.txt")

# Matches  manifest__OP_0..63_CL_.json  or  manifest__OP_0..63_CL_.json
SHARD_PATTERN = re.compile(r"__OP_(\d+)\.\.(\d+)_CL_")


def expand_manifest_path(manifest_filepath: str) -> List[str]:
    """Expand a sharded manifest path to a list of actual file paths.

    Handles patterns like manifest__OP_0..63_CL_.json by looking in the same
    directory for manifest_N.json files, or falls back to a glob.
    """
    path = manifest_filepath.replace("//", "/")
    m = SHARD_PATTERN.search(path)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        directory = os.path.dirname(path)
        # Replace the __OP_X..Y_CL_ token with a single shard index
        prefix = SHARD_PATTERN.sub("{}", path)
        candidates = []
        for i in range(start, end + 1):
            p = prefix.format(i).replace("//", "/")
            if os.path.exists(p):
                candidates.append(p)
        if candidates:
            return candidates
        # fallback: glob everything in directory
        found = sorted(glob.glob(os.path.join(directory, "manifest_*.json")))
        return found if found else []
    # No shard pattern – try as-is or glob
    path_clean = path.replace("//", "/")
    if os.path.exists(path_clean):
        return [path_clean]
    directory = os.path.dirname(path_clean)
    found = sorted(glob.glob(os.path.join(directory, "manifest_*.json")))
    return found if found else []


def _yield_from_list(items):
    for item in items:
        if isinstance(item, dict) and "manifest_filepath" in item:
            yield (
                str(item.get("corpus", item.get("dataset", "(unknown)"))),
                str(item.get("language", item.get("lang", "(unknown)"))),
                item["manifest_filepath"],
            )
        elif isinstance(item, dict):
            # recurse one more level (e.g. input_cfg items that are dicts with a list)
            for v in item.values():
                if isinstance(v, list):
                    yield from _yield_from_list(v)


def iter_entries(yaml_docs: List[Any]):
    """Yield (corpus, language, manifest_filepath) from loaded YAML documents."""
    for doc in yaml_docs:
        if isinstance(doc, list):
            yield from _yield_from_list(doc)
        elif isinstance(doc, dict):
            for v in doc.values():
                if isinstance(v, list):
                    yield from _yield_from_list(v)


def percentile(sorted_data: List[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (k - lo)


def collect_stats(yaml_docs: List[Any], max_shards: int = 3):
    """Returns dict (corpus, language) -> list of durations (seconds).

    max_shards: max number of shard files to sample per YAML entry (for speed).
    Set to 0 to read all shards.
    """
    # Track which shard files we've already read to avoid double-counting
    # across duplicate YAML entries for the same (corpus, lang)
    seen_shards: Dict[Tuple[str, str], set] = defaultdict(set)
    group_durations: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    entries = list(iter_entries(yaml_docs))
    total = len(entries)
    print(f"Found {total} manifest entries in YAML.", file=sys.stderr)
    if max_shards:
        print(f"Sampling up to {max_shards} shard(s) per entry (use --max-shards 0 for all).", file=sys.stderr)

    for idx, (corpus, lang, mfp) in enumerate(entries):
        files = expand_manifest_path(mfp)
        if not files:
            print(f"  [WARN] no files found for: {mfp}", file=sys.stderr)
            continue

        key = (corpus, lang)
        already = seen_shards[key]
        new_files = [f for f in files if f not in already]
        if not new_files:
            continue

        # Sample evenly across the available shards
        if max_shards and len(new_files) > max_shards:
            step = len(new_files) / max_shards
            new_files = [new_files[int(i * step)] for i in range(max_shards)]

        seen_shards[key].update(new_files)

        n_utt = 0
        for fpath in new_files:
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                            dur = d.get("duration")
                            if dur is not None:
                                group_durations[key].append(float(dur))
                                n_utt += 1
                        except Exception:
                            continue
            except Exception as e:
                print(f"  [WARN] could not read {fpath}: {e}", file=sys.stderr)

        print(f"  [{idx+1}/{total}] {corpus}/{lang}: +{n_utt} utts from {len(new_files)} shards", file=sys.stderr)

    return group_durations


def print_stats_table(group_durations: Dict[Tuple[str, str], List[float]], out=None):
    def p(*args, **kwargs):
        print(*args, **kwargs)
        if out:
            print(*args, **kwargs, file=out)

    header = "| Corpus | Language | # Utts | Total hrs | Mean s | Median s | Std s | Min s | Max s | p5 s | p95 s |"
    sep    = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    p("## Utterance duration statistics per corpus × language")
    p(header)
    p(sep)

    rows = []
    for (corpus, lang), durations in group_durations.items():
        sd = sorted(durations)
        n = len(sd)
        if n == 0:
            continue
        total_hrs = sum(sd) / 3600.0
        m = mean(sd)
        med = median(sd)
        std = stdev(sd) if n > 1 else 0.0
        mn = sd[0]
        mx = sd[-1]
        p5 = percentile(sd, 5)
        p95 = percentile(sd, 95)
        rows.append((corpus, lang, n, total_hrs, m, med, std, mn, mx, p5, p95))

    # sort by corpus then language
    rows.sort(key=lambda r: (r[0], r[1]))

    for corpus, lang, n, total_hrs, m, med, std, mn, mx, p5, p95 in rows:
        p(f"| {corpus} | {lang} | {n:,} | {total_hrs:.1f} | {m:.2f} | {med:.2f} | {std:.2f} | {mn:.2f} | {mx:.2f} | {p5:.2f} | {p95:.2f} |")


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Utterance duration stats from YAML manifests")
    ap.add_argument("files", nargs="+", help="YAML file(s)")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output text file path")
    ap.add_argument(
        "--max-shards", type=int, default=3,
        help="Max shard files to sample per YAML entry (0 = all, default: 3)",
    )
    args = ap.parse_args(argv)

    yaml_docs: List[Any] = []
    for fn in args.files:
        try:
            with open(fn, "r", encoding="utf-8") as fh:
                yaml_docs.extend(yaml.safe_load_all(fh))
        except FileNotFoundError:
            print(f"File not found: {fn}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to parse {fn}: {e}", file=sys.stderr)

    if not yaml_docs:
        print("No YAML content loaded.", file=sys.stderr)
        return 2

    group_durations = collect_stats(yaml_docs, max_shards=args.max_shards)

    with open(args.output, "w", encoding="utf-8") as out_fh:
        print_stats_table(group_durations, out=out_fh)

    print(f"\nSaved to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
