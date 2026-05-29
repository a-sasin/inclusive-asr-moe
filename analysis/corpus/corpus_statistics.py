#!/usr/bin/env python3
"""Compute and print corpus-level statistics from NeMo JSONL manifests.

Reads one or more NeMo manifest files (JSONL, one JSON object per line with
fields: audio_filepath, duration, text) and reports per-corpus summaries:
sample count, total hours, unique speaker count (if speaker_id is present),
and text length distribution.

Usage:
    python corpus_statistics.py /path/to/manifest1.json [manifest2.json ...]

Output:
    Prints a markdown-formatted statistics table to stdout.
"""
