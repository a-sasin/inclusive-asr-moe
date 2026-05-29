#!/usr/bin/env python3
"""Restore speaker IDs in PAVSig manifests after multi-channel audio conversion.

After fix_audio.py converts multi-channel files to mono, the output filenames
may differ from the originals, causing speaker IDs extracted from audio paths
to be incorrect. This script re-maps the speaker IDs back to the canonical
PAVSig speaker identifiers using the original manifest as ground truth.

Usage:
    python restore_speakers.py \
        --original_manifest /path/to/original_manifest.json \
        --converted_manifest /path/to/converted_manifest.json \
        --output /path/to/fixed_manifest.json
"""
