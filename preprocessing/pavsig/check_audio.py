#!/usr/bin/env python3
"""Check audio file integrity in PAVSig manifests.

Iterates over entries in a NeMo JSONL manifest, verifies that each referenced
audio file is readable and not corrupted, and reports any problematic entries.
Useful as a pre-training sanity check after fix_audio.py has normalised the
format of multi-channel recordings to mono.

Usage:
    python check_audio.py --manifest /path/to/manifest.json [--num_workers 4]
"""
