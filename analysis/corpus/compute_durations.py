#!/usr/bin/env python3
"""Compute duration totals per language and per dataset from YAML files.

Usage:
  python compute_durations.py /path/to/file.yaml [more.yaml]

The script searches the YAML structure recursively for dictionaries that contain
a `duration` key. It will try to extract `language` (or `lang`) and `dataset`
from the same dict; if not present, it will inherit values from parent keys
when possible.

Outputs Markdown tables: per-language totals, per-dataset totals, and a
language x dataset breakdown.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os

try:
    import yaml
except Exception:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    raise


def iter_dicts_with_context(obj: Any, context: Dict[str, Any] | None = None):
    """Recursively iterate dict-like objects, yielding (d, context).

    The context is a dict of keys found on the path (e.g. dataset, language)
    which may be inherited by nested entries.
    """
    if context is None:
        context = {}

    if isinstance(obj, dict):
        # Merge possible language/dataset keys into context for children
        local_context = dict(context)
        for k in ("language", "lang", "dataset", "set", "corpus"):
            if k in obj:
                local_context[k] = obj[k]

        yield obj, local_context

        for v in obj.values():
            yield from iter_dicts_with_context(v, local_context)

    elif isinstance(obj, list):
        for item in obj:
            yield from iter_dicts_with_context(item, context)


def collect_durations(objs: Iterable[Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[Tuple[str, str], float]]:
    """Collect totals per language, per dataset, and per (language,dataset).

    Returns (lang_totals, dataset_totals, lang_dataset_totals)
    """
    lang_totals: Dict[str, float] = defaultdict(float)
    dataset_totals: Dict[str, float] = defaultdict(float)
    lang_dataset_totals: Dict[Tuple[str, str], float] = defaultdict(float)

    DURATION_KEYS = ("duration", "duration_seconds", "seconds", "length", "hours")

    for obj in objs:
        for d, ctx in iter_dicts_with_context(obj):
            if not isinstance(d, dict):
                continue
            if not any(k in d for k in DURATION_KEYS):
                continue

            # detect duration from multiple possible keys
            dur = None
            if "duration" in d:
                try:
                    dur = float(d.get("duration"))
                except Exception:
                    dur = None
            if dur is None and "duration_seconds" in d:
                try:
                    dur = float(d.get("duration_seconds"))
                except Exception:
                    dur = None
            if dur is None and "seconds" in d:
                try:
                    dur = float(d.get("seconds"))
                except Exception:
                    dur = None
            if dur is None and "length" in d:
                try:
                    dur = float(d.get("length"))
                except Exception:
                    dur = None
            # hours field: convert to seconds
            if dur is None and "hours" in d:
                try:
                    dur = float(d.get("hours")) * 3600.0
                except Exception:
                    dur = None
            # allow fallback to context (parent) if a numeric hours/duration key exists
            if dur is None:
                for key in ("duration", "duration_seconds", "seconds", "length", "hours"):
                    if key in ctx:
                        try:
                            val = float(ctx[key])
                            if key == "hours":
                                val = val * 3600.0
                            dur = val
                            break
                        except Exception:
                            continue

            if dur is None:
                # skip if no numeric duration found
                continue

            # determine language
            lang = None
            for k in ("language", "lang", "locale"):
                if k in d:
                    lang = d[k]
                    break
            if lang is None:
                for k in ("language", "lang"):
                    if k in ctx:
                        lang = ctx[k]
                        break
            if lang is None:
                lang = "(unknown)"

            # determine dataset (prefer explicit `dataset` or `corpus`, else infer from manifest path)
            dataset = None
            for k in ("dataset", "set", "name", "corpus"):
                if k in d:
                    dataset = d[k]
                    break
            if dataset is None:
                for k in ("dataset", "set", "corpus"):
                    if k in ctx:
                        dataset = ctx[k]
                        break

            if dataset is None and "manifest_filepath" in d:
                try:
                    mp = str(d.get("manifest_filepath"))
                    # prefer parent directory name if it seems meaningful
                    parent = os.path.basename(os.path.dirname(mp))
                    dataset = parent or os.path.basename(mp)
                except Exception:
                    dataset = None

            if dataset is None:
                dataset = "(unknown)"

            lang = str(lang)
            dataset = str(dataset)

            lang_totals[lang] += dur
            dataset_totals[dataset] += dur
            lang_dataset_totals[(lang, dataset)] += dur

    return lang_totals, dataset_totals, lang_dataset_totals


def print_md_tables(lang_totals: Dict[str, float], dataset_totals: Dict[str, float], lang_dataset: Dict[Tuple[str, str], float], out=None):
    def p(*args, **kwargs):
        print(*args, **kwargs)
        if out is not None:
            kwargs["file"] = out
            print(*args, **kwargs)

    # Per-language
    p("## Duration per language")
    p("| Language | Total duration (seconds) |")
    p("|---:|---:|")
    for lang, val in sorted(lang_totals.items(), key=lambda x: -x[1]):
        p(f"| {lang} | {val:.2f} |")

    p()

    # Per-dataset
    p("## Duration per dataset")
    p("| Dataset | Total duration (seconds) |")
    p("|---|---:|")
    for ds, val in sorted(dataset_totals.items(), key=lambda x: -x[1]):
        p(f"| {ds} | {val:.2f} |")

    p()

    # Language x Dataset breakdown
    p("## Duration per language and dataset")
    # collect unique languages and datasets
    langs = sorted({k[0] for k in lang_dataset})
    dsets = sorted({k[1] for k in lang_dataset})

    # header
    header = "| Language \\ Dataset | " + " | ".join(dsets) + " | Total |"
    p(header)
    p("|---" + "|---" * (len(dsets) + 1) + "|")

    for lang in langs:
        row_vals = []
        row_total = 0.0
        for ds in dsets:
            v = lang_dataset.get((lang, ds), 0.0)
            row_vals.append(f"{v:.2f}")
            row_total += v
        row = "| " + lang + " | " + " | ".join(row_vals) + f" | {row_total:.2f} |"
        p(row)


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Aggregate durations from YAML files")
    p.add_argument("files", nargs="+", help="YAML file(s) to process")
    p.add_argument(
        "--output",
        default="/lp-dev/amelia/inclusive-asr-moe/analysis/corpus/duration.txt",
        help="Path to save the output Markdown tables",
    )
    args = p.parse_args(argv)

    all_objs: List[Any] = []
    for fn in args.files:
        try:
            with open(fn, "r", encoding="utf-8") as fh:
                data = list(yaml.safe_load_all(fh))
                # safe_load_all yields zero or more documents
                if not data:
                    continue
                # if a single document which is a list or dict, keep as is
                all_objs.extend(data)
        except FileNotFoundError:
            print(f"File not found: {fn}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Failed to parse {fn}: {e}", file=sys.stderr)
            continue

    if not all_objs:
        print("No YAML content found.")
        return 2

    lang_totals, dataset_totals, lang_dataset = collect_durations(all_objs)
    with open(args.output, "w", encoding="utf-8") as out_fh:
        print_md_tables(lang_totals, dataset_totals, lang_dataset, out=out_fh)
    print(f"\nResults saved to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
