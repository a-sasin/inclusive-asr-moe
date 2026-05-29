"""
prepare_cv25_nemo.py
====================
Complete pipeline: download CV 25.0 (NL / DE / PL) → quality filter →
speaker-independent split → resample MP3→WAV → NeMo JSONL manifests.

Key design decisions:
  - ALL languages use validated.tsv as the source pool (superset of the
    Mozilla train/dev/test split), then re-split speaker-independently.
    This unlocks the full validated audio for every language:
      NL ~102h  |  DE ~1100h  |  PL ~152h
  - Each split is capped independently:
      --train_hours  (default 45h)
      --val_hours    (default  1h)
      --test_hours   (default 12h)
    Languages that have less than the cap contribute everything they have.
  - Adaptive speaker cap: raised automatically when speaker count is low
    (fixes the NL 8h collapse caused by 58 speakers × 150-clip hard cap).

Paths (hardcoded for this project):
    Raw CV downloads : /data/cv/
    Script location  : /lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice/
    .env             : /lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice/.env
    NeMo manifests   : /data/cv/nemo/

Usage
-----
    # Dry run — reports hours without writing files
    python prepare_cv_nemo.py --dry_run

    # Full pipeline with defaults (45h train / 1h val / 12h test)
    python prepare_cv_nemo.py --num_workers 16

    # Custom targets
    python prepare_cv_nemo.py --num_workers 16 --train_hours 45 --val_hours 1 --test_hours 12
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm.contrib.concurrent import process_map

# ─────────────────────────────────────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path("/lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice")
DATA_DIR     = Path("/data/cv")
RAW_DIR      = DATA_DIR / "raw"
OUTPUT_DIR   = DATA_DIR / "nemo"
ENV_FILE     = SCRIPT_DIR / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Dataset IDs — CV 25.0
# ─────────────────────────────────────────────────────────────────────────────

DATASET_IDS = {
    "nl": "cmn2g7nu901fmo107a1ydn0n5",
    "de": "cmn4rsdh6009unz07jdn2ol9p",
    "pl": "cmn27nz69015hmm0720txf781",
}

LANGUAGE_NAMES = {"nl": "Dutch", "de": "German", "pl": "Polish"}

# All languages use validated.tsv as the source pool.
# Each split is capped independently via --train_hours / --val_hours / --test_hours.
DEFAULT_TRAIN_HOURS = 45.0
DEFAULT_VAL_HOURS   =  1.0
DEFAULT_TEST_HOURS  = 12.0

# ─────────────────────────────────────────────────────────────────────────────
# Quality thresholds
# ─────────────────────────────────────────────────────────────────────────────

MIN_UPVOTES    = 2
MAX_DOWNVOTES  = 0
MIN_DURATION_S = 1.0
MAX_DURATION_S = 15.0
MIN_TEXT_WORDS = 2


def quality_score(row) -> float:
    up   = float(row.get("up_votes")   or 0)
    down = float(row.get("down_votes") or 0)
    return up / (up + down + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Extract / download
# ─────────────────────────────────────────────────────────────────────────────

def extract_tar_gz(lang_code: str, force_reextract: bool = False) -> Path:
    tar_path = DATA_DIR / f"common_voice_{lang_code}.tar.gz"

    lang_dir_candidate = RAW_DIR / lang_code
    if lang_dir_candidate.exists() and (lang_dir_candidate / "validated.tsv").exists():
        if not force_reextract:
            log.info(f"[{lang_code}] Already extracted at {lang_dir_candidate}")
            return lang_dir_candidate

    if tar_path.exists():
        log.info(f"[{lang_code}] Found tar.gz at {tar_path}, extracting ...")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(RAW_DIR)
            log.info(f"[{lang_code}] Extraction complete")
            return _find_lang_dir(RAW_DIR, lang_code)
        except Exception as e:
            log.error(f"[{lang_code}] Failed to extract: {e}")

    log.info(f"[{lang_code}] tar.gz not found, downloading...")
    return download_language(lang_code)


def _load_env() -> None:
    if os.environ.get("MDC_API_KEY"):
        return
    if not ENV_FILE.exists():
        log.warning(f".env not found at {ENV_FILE}.")
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def download_language(lang_code: str) -> Path:
    import requests
    from tqdm import tqdm

    _load_env()
    api_key = os.environ.get("MDC_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"MDC_API_KEY not set. Cannot download {lang_code}.")

    dataset_id = DATASET_IDS[lang_code]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"[{lang_code}] Requesting download URL ...")

    resp = requests.post(
        f"https://datacollective.mozillafoundation.org/api/datasets/{dataset_id}/download",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    payload      = resp.json()
    download_url = payload["downloadUrl"]
    filename     = payload["filename"]
    size_bytes   = int(payload["sizeBytes"])
    archive_path = RAW_DIR / filename

    log.info(f"[{lang_code}] Downloading {filename} ({size_bytes/1e9:.1f} GB)")

    headers, downloaded = {}, 0
    if archive_path.exists():
        downloaded = archive_path.stat().st_size
        if downloaded >= size_bytes:
            log.info(f"[{lang_code}] Already fully downloaded.")
        else:
            headers["Range"] = f"bytes={downloaded}-"

    if downloaded < size_bytes:
        with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
            r.raise_for_status()
            mode = "ab" if downloaded > 0 else "wb"
            with open(archive_path, mode) as f:
                for chunk in tqdm(r.iter_content(chunk_size=1 << 20),
                                  desc=f"  {lang_code}", unit="MB",
                                  initial=downloaded // (1 << 20),
                                  total=size_bytes // (1 << 20)):
                    if chunk:
                        f.write(chunk)

    log.info(f"[{lang_code}] Extracting ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    return _find_lang_dir(RAW_DIR, lang_code)


def _find_lang_dir(root: Path, lang_code: str) -> Path:
    # validated.tsv is the primary indicator; fall back to train.tsv
    for tsv_name in ("validated.tsv", "train.tsv"):
        for candidate in root.rglob(tsv_name):
            parent = candidate.parent
            if parent.name == lang_code and (parent / "clips").exists():
                return parent
    raise FileNotFoundError(
        f"Could not find extracted CV directory for '{lang_code}' under {root}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Duration measurement
# ─────────────────────────────────────────────────────────────────────────────

def _get_duration_fast(mp3_path: str) -> float:
    try:
        from mutagen.mp3 import MP3
        return MP3(mp3_path).info.length
    except Exception:
        pass
    try:
        result = subprocess.run(["soxi", "-D", mp3_path],
                                capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return -1.0


def _duration_worker(args):
    (path,) = args
    return path, _get_duration_fast(path)


def measure_durations(paths: list, num_workers: int) -> dict:
    jobs = [(p,) for p in paths]
    results = process_map(_duration_worker, jobs,
                          max_workers=num_workers, chunksize=200,
                          desc="    Measuring durations")
    return {r[0]: r[1] for r in results if r[1] > 0}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Quality filtering
# ─────────────────────────────────────────────────────────────────────────────

def load_tsv(tsv_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        tsv_path, sep="\t", low_memory=False,
        dtype={"client_id": str, "path": str, "sentence": str,
               "up_votes": "Int64", "down_votes": "Int64",
               "age": str, "gender": str, "accents": str},
    )


def apply_quality_filter(
    df: pd.DataFrame,
    clips_dir: Path,
    duration_cache: dict,
    lang_code: str,
    split_name: str,
) -> pd.DataFrame:
    n_raw = len(df)

    df = df[
        (df["up_votes"].fillna(0)     >= MIN_UPVOTES) &
        (df["down_votes"].fillna(999) <= MAX_DOWNVOTES)
    ]
    df = df[df["sentence"].notna() & (df["sentence"].str.strip() != "")]
    df = df[df["sentence"].str.split().str.len() >= MIN_TEXT_WORDS]

    df = df.copy()
    df["_full_path"] = df["path"].apply(
        lambda p: str(clips_dir / (p if p.endswith(".mp3") else p + ".mp3"))
    )
    df = df[df["_full_path"].apply(lambda p: Path(p).exists())]

    df["duration"] = df["_full_path"].map(duration_cache)
    df = df[df["duration"].notna()]
    df["duration"] = df["duration"].astype(float)
    df = df[(df["duration"] >= MIN_DURATION_S) & (df["duration"] <= MAX_DURATION_S)]

    df["quality_score"] = df.apply(quality_score, axis=1)

    n_kept = len(df)
    hrs    = df["duration"].sum() / 3600
    log.info(
        f"      [{lang_code}/{split_name}] {n_raw:,} raw → {n_kept:,} kept "
        f"({hrs:.1f}h, up≥{MIN_UPVOTES} down≤{MAX_DOWNVOTES} "
        f"dur {MIN_DURATION_S}–{MAX_DURATION_S}s)"
    )
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Hour-balanced downsampling (adaptive speaker cap)
# ─────────────────────────────────────────────────────────────────────────────

def balance_to_target_hours(
    df: pd.DataFrame,
    target_hours: float,
    lang_code: str,
    split_name: str,
    max_clips_per_speaker: int = 150,
) -> pd.DataFrame:
    """
    Downsample df to target_hours with an adaptive per-speaker cap.

    The cap is raised automatically when the speaker count is too low for the
    hard cap to reach target_hours — which happens with Dutch (58 speakers).
    Without this fix NL train collapses to ~8h even though 54h is available.
    """
    available = df["duration"].sum() / 3600

    if available <= target_hours:
        log.info(
            f"      [{lang_code}/{split_name}] {available:.1f}h ≤ target "
            f"{target_hours:.1f}h — keeping all"
        )
        return df

    # Raise the cap if it would prevent reaching target_hours
    n_speakers       = df["client_id"].nunique()
    avg_dur_per_clip = df["duration"].mean()
    clips_needed     = int((target_hours * 3600) / avg_dur_per_clip) + 1
    min_cap_needed   = -(-clips_needed // n_speakers)   # ceiling division
    effective_cap    = max(max_clips_per_speaker, min_cap_needed)

    if effective_cap > max_clips_per_speaker:
        log.info(
            f"      [{lang_code}/{split_name}] Raising speaker cap "
            f"{max_clips_per_speaker} → {effective_cap} "
            f"({n_speakers} speakers, need ≥{min_cap_needed} clips/speaker "
            f"to reach {target_hours:.1f}h)"
        )

    log.info(
        f"      [{lang_code}/{split_name}] Downsampling "
        f"{available:.1f}h → {target_hours:.1f}h "
        f"(quality-ranked, speaker cap={effective_cap})"
    )

    df_sorted      = df.sort_values("quality_score", ascending=False)
    selected       = []
    speaker_counts = defaultdict(int)
    cumulative_h   = 0.0

    for _, row in df_sorted.iterrows():
        if cumulative_h >= target_hours:
            break
        spk = row["client_id"]
        if speaker_counts[spk] >= effective_cap:
            continue
        selected.append(row)
        speaker_counts[spk] += 1
        cumulative_h += row["duration"] / 3600

    result = pd.DataFrame(selected).reset_index(drop=True)
    log.info(
        f"      [{lang_code}/{split_name}] Selected {len(result):,} clips | "
        f"{cumulative_h:.1f}h | {result['client_id'].nunique():,} speakers"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b: Speaker-independent re-split (used for NL validated pool)
# ─────────────────────────────────────────────────────────────────────────────

def speaker_independent_split(
    df: pd.DataFrame,
    lang_code: str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train/val/test by speaker (no speaker appears in two splits).
    Returns (train_df, val_df, test_df).
    """
    import random
    random.seed(seed)

    speakers = list(df["client_id"].unique())
    random.shuffle(speakers)

    n        = len(speakers)
    n_test   = max(1, int(n * test_ratio))
    n_val    = max(1, int(n * val_ratio))

    test_spk = set(speakers[:n_test])
    val_spk  = set(speakers[n_test:n_test + n_val])
    train_spk = set(speakers[n_test + n_val:])

    train_df = df[df["client_id"].isin(train_spk)].reset_index(drop=True)
    val_df   = df[df["client_id"].isin(val_spk)].reset_index(drop=True)
    test_df  = df[df["client_id"].isin(test_spk)].reset_index(drop=True)

    log.info(
        f"  [{lang_code}] Speaker-independent split: "
        f"train {len(train_spk)} spk / val {len(val_spk)} spk / test {len(test_spk)} spk"
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        hrs = part["duration"].sum() / 3600
        log.info(f"      {name}: {len(part):,} clips | {hrs:.1f}h")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Text normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _base_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[–—−‐‑‒]",                   " ",  text)
    text = re.sub(r"[''`ʽ´ʻ]",                   "'",  text)
    text = re.sub(r'[""„‟″«»]',                  "",   text)
    text = re.sub(r"[\u00AD\u200B-\u200D\uFEFF]", "", text)
    return re.sub(r" +", " ", text).strip()


def normalize_nl(text: str) -> str:
    text = _base_clean(text)
    text = re.sub(r"[^a-zàâäéèêëîïôöùûüÿçæœ' ]", " ", text)
    text = re.sub(r"(?<![a-zàâäéèêëîïôöùûüÿçæœ])'|'(?![a-zàâäéèêëîïôöùûüÿçæœ])", " ", text)
    return re.sub(r" +", " ", text).strip()


def normalize_de(text: str) -> str:
    text = _base_clean(text)
    return re.sub(r" +", " ", re.sub(r"[^a-zäöüß ]", " ", text)).strip()


def normalize_pl(text: str) -> str:
    text = _base_clean(text)
    return re.sub(r" +", " ", re.sub(r"[^a-ząćęłńóśźż ]", " ", text)).strip()


NORMALIZERS = {"nl": normalize_nl, "de": normalize_de, "pl": normalize_pl}


def normalize_texts(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
    normalizer = NORMALIZERS[lang_code]
    df = df.copy()
    df["text"] = df["sentence"].apply(
        lambda t: normalizer(str(t) if pd.notna(t) else "")
    )
    before = len(df)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning(f"      Dropped {dropped} entries with empty text after normalisation")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Resample MP3 → 16 kHz mono WAV
# ─────────────────────────────────────────────────────────────────────────────

def _resample_worker(args):
    src_str, dst_str = args
    src, dst = Path(src_str), Path(dst_str)

    if dst.exists():
        try:
            import sox
            return src_str, dst_str, sox.file_info.duration(str(dst))
        except Exception:
            pass

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        import sox
        tfm = sox.Transformer()
        tfm.rate(samplerate=16000)
        tfm.channels(n_channels=1)
        tfm.build(input_filepath=str(src), output_filepath=str(dst))
        return src_str, dst_str, sox.file_info.duration(str(dst))
    except Exception:
        pass

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(dst)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        r = subprocess.run(["soxi", "-D", str(dst)],
                           capture_output=True, text=True, check=True)
        return src_str, dst_str, float(r.stdout.strip())
    except Exception:
        return src_str, dst_str, None


def resample_audio(df: pd.DataFrame, wav_dir: Path, num_workers: int) -> pd.DataFrame:
    wav_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (row["_full_path"],
         str(wav_dir / (Path(row["_full_path"]).stem + ".wav")))
        for _, row in df.iterrows()
    ]

    results = process_map(_resample_worker, jobs,
                          max_workers=num_workers, chunksize=50, desc="    Resample")

    lookup = {r[0]: (r[1], r[2]) for r in results}
    records, failed = [], 0

    for _, row in df.iterrows():
        wav_path, dur = lookup.get(row["_full_path"], (None, None))
        if not wav_path or not dur or dur <= 0:
            failed += 1
            continue
        rec = row.to_dict()
        rec["audio_filepath"] = wav_path
        rec["duration"]       = round(float(dur), 4)
        records.append(rec)

    if failed:
        log.warning(f"    {failed} clips failed resampling and were dropped")

    return pd.DataFrame(records).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Write NeMo JSONL manifest
# ─────────────────────────────────────────────────────────────────────────────

def write_manifest(df: pd.DataFrame, output_path: Path, lang_code: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            if not row.get("audio_filepath") or not row.get("text") or not row.get("duration"):
                continue
            record = {
                "audio_filepath": str(row["audio_filepath"]),
                "text":           str(row["text"]),
                "duration":       float(row["duration"]),
                "speaker":        str(row.get("client_id", "")),
                "language":       lang_code,
                "age_group":      "adult",
            }
            for opt in ["age", "gender", "accents", "up_votes", "down_votes"]:
                val = row.get(opt)
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    record[opt] = val
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    hrs = df["duration"].sum() / 3600 if "duration" in df.columns else 0
    spk = df["client_id"].nunique()    if "client_id" in df.columns else 0
    log.info(f"    → {output_path}: {written:,} clips | {hrs:.1f}h | {spk:,} speakers")


# ─────────────────────────────────────────────────────────────────────────────
# Speaker-independence check
# ─────────────────────────────────────────────────────────────────────────────

def verify_speaker_independence(lang_output_dir: Path, lang_code: str) -> None:
    splits = {}
    for split in ["train", "validation", "test"]:
        p = lang_output_dir / f"{split}.json"
        if not p.exists():
            continue
        spks = set()
        with open(p) as f:
            for line in f:
                spks.add(json.loads(line).get("speaker", ""))
        splits[split] = spks

    names = list(splits.keys())
    clean = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = splits[a] & splits[b]
            if overlap:
                log.warning(f"  [{lang_code}] SPEAKER LEAK: {a} ∩ {b} = {len(overlap)} speakers!")
                clean = False
            else:
                log.info(f"  [{lang_code}] OK: {a} ∩ {b} = 0 shared speakers")
    if clean:
        log.info(f"  [{lang_code}] Speaker independence verified ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Per-language pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_language(
    lang_code: str,
    lang_dir: Path,
    target_hours_train: float,
    target_hours_val: float,
    target_hours_test: float,
    num_workers: int,
    dry_run: bool,
) -> dict:
    clips_dir   = lang_dir / "clips"
    lang_output = OUTPUT_DIR / lang_code
    wav_dir     = lang_output / "wav"

    log.info(f"\n{'='*60}")
    log.info(f"  {LANGUAGE_NAMES[lang_code]} ({lang_code.upper()})")
    log.info(f"  Source : {lang_dir}")
    log.info(f"  Output : {lang_output}")
    log.info(f"  Targets: train {target_hours_train}h | val {target_hours_val}h | test {target_hours_test}h")
    log.info(f"{'='*60}")

    stats = {"lang": lang_code, "splits": {}}

    log.info(f"  Pre-measuring MP3 durations ...")
    all_mp3s = list(clips_dir.glob("*.mp3"))
    log.info(f"  Found {len(all_mp3s):,} MP3 files in {clips_dir}")
    duration_cache = measure_durations([str(p) for p in all_mp3s], num_workers)

    # ── Load validated.tsv (full quality-passed pool) ─────────────────────────
    validated_tsv = lang_dir / "validated.tsv"
    if not validated_tsv.exists():
        log.warning(f"  validated.tsv not found, falling back to train.tsv")
        validated_tsv = lang_dir / "train.tsv"

    log.info(f"\n  [loading {validated_tsv.name}]")
    df_all = load_tsv(validated_tsv)
    df_all = apply_quality_filter(df_all, clips_dir, duration_cache, lang_code, "validated")
    df_all = normalize_texts(df_all, lang_code)

    total_h = df_all["duration"].sum() / 3600
    log.info(f"  Total available after filter: {total_h:.1f}h")

    # ── Speaker-independent split ─────────────────────────────────────────────
    # Allocate speakers proportionally to the requested hours.
    total_target = target_hours_train + target_hours_val + target_hours_test
    val_ratio  = target_hours_val  / total_target
    test_ratio = target_hours_test / total_target

    train_df, val_df, test_df = speaker_independent_split(
        df_all, lang_code,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # ── Cap each split at its target ──────────────────────────────────────────
    splits = [
        ("train",      train_df, target_hours_train),
        ("validation", val_df,   target_hours_val),
        ("test",       test_df,  target_hours_test),
    ]

    for split_name, df, target in splits:
        df = balance_to_target_hours(df, target, lang_code, split_name)

        hrs = df["duration"].sum() / 3600
        spk = df["client_id"].nunique()
        stats["splits"][split_name] = {"clips": len(df), "hours": hrs, "speakers": spk}

        if dry_run:
            log.info(f"  [DRY RUN] {split_name}: {len(df):,} clips | {hrs:.1f}h | {spk:,} speakers")
            continue

        log.info(f"  Resampling {len(df):,} clips [{split_name}] to 16kHz WAV ...")
        df = resample_audio(df, wav_dir, num_workers)
        write_manifest(df, lang_output / f"{split_name}.json", lang_code)

    if not dry_run:
        verify_speaker_independence(lang_output, lang_code)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CV 25.0 NL/DE/PL → quality-filtered, speaker-split NeMo manifests.\n"
                    "All languages use validated.tsv for maximum data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--languages", nargs="+", default=["nl", "de", "pl"])
    parser.add_argument(
        "--train_hours", type=float, default=DEFAULT_TRAIN_HOURS,
        help=f"Max train hours per language (default: {DEFAULT_TRAIN_HOURS})",
    )
    parser.add_argument(
        "--val_hours", type=float, default=DEFAULT_VAL_HOURS,
        help=f"Max validation hours per language (default: {DEFAULT_VAL_HOURS})",
    )
    parser.add_argument(
        "--test_hours", type=float, default=DEFAULT_TEST_HOURS,
        help=f"Max test hours per language (default: {DEFAULT_TEST_HOURS})",
    )
    parser.add_argument(
        "--num_workers", type=int,
        default=max(1, multiprocessing.cpu_count() - 2),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_reextract", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    log.info("CV 25.0 NeMo Preparation Pipeline")
    log.info(f"  Languages    : {args.languages}")
    log.info(f"  Train target : {args.train_hours}h per language")
    log.info(f"  Val target   : {args.val_hours}h per language")
    log.info(f"  Test target  : {args.test_hours}h per language")
    log.info(f"  Workers      : {args.num_workers}")
    log.info(f"  Dry run      : {args.dry_run}")

    # Locate / extract archives
    lang_dirs = {}
    for lang_code in args.languages:
        try:
            lang_dirs[lang_code] = extract_tar_gz(
                lang_code, force_reextract=args.force_reextract
            )
            log.info(f"[{lang_code}] Ready at {lang_dirs[lang_code]}")
        except FileNotFoundError as e:
            log.error(f"[{lang_code}] {e}")

    if not lang_dirs:
        log.error("No language directories found. Exiting.")
        sys.exit(1)

    if args.dry_run:
        log.info("\n── Dry run: probing available hours ──────────────────────────")
        for lang_code, lang_dir in lang_dirs.items():
            clips_dir = lang_dir / "clips"
            tsv_path  = lang_dir / "validated.tsv"
            if not tsv_path.exists():
                tsv_path = lang_dir / "train.tsv"
            if not tsv_path.exists():
                continue
            df = load_tsv(tsv_path)
            paths    = [str(clips_dir / (p if p.endswith(".mp3") else p + ".mp3"))
                        for p in df["path"].dropna()]
            existing = [p for p in paths if Path(p).exists()]
            dur_cache = measure_durations(existing, args.num_workers)
            df = apply_quality_filter(df, clips_dir, dur_cache, lang_code, "probe")
            hrs = df["duration"].sum() / 3600
            spk = df["client_id"].nunique()
            total_target = args.train_hours + args.val_hours + args.test_hours
            log.info(
                f"  {LANGUAGE_NAMES[lang_code]:<8}: {hrs:.1f}h available | {spk} speakers | "
                f"need {total_target}h → {'OK' if hrs >= total_target else f'SHORT by {total_target-hrs:.1f}h'}"
            )
        return

    # Run per-language pipeline
    all_stats = []
    for lang_code, lang_dir in lang_dirs.items():
        stats = process_language(
            lang_code=lang_code,
            lang_dir=lang_dir,
            target_hours_train=args.train_hours,
            target_hours_val=args.val_hours,
            target_hours_test=args.test_hours,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
        all_stats.append(stats)

    # Merge train manifests
    train_manifests = [
        OUTPUT_DIR / lc / "train.json"
        for lc in lang_dirs
        if (OUTPUT_DIR / lc / "train.json").exists()
    ]
    merged_path = OUTPUT_DIR / "train_merged.json"
    if train_manifests:
        log.info(f"\nMerging {len(train_manifests)} train manifests → {merged_path}")
        with open(merged_path, "w", encoding="utf-8") as fout:
            for mp in train_manifests:
                with open(mp, encoding="utf-8") as fin:
                    fout.writelines(fin)

    # Final summary
    print("\n" + "=" * 70)
    print(f"{'FINAL SUMMARY — CV 25.0 NeMo Data':^70}")
    print("=" * 70)
    print(f"{'Lang':<8} {'Split':<12} {'Clips':>8} {'Hours':>8} {'Speakers':>10}")
    print("-" * 70)
    total_train_clips, total_train_hours = 0, 0.0
    for stats in all_stats:
        for split, info in stats["splits"].items():
            print(f"{stats['lang']:<8} {split:<12} {info['clips']:>8,} "
                  f"{info['hours']:>7.1f}h {info['speakers']:>10,}")
            if split == "train":
                total_train_clips += info["clips"]
                total_train_hours += info["hours"]
    print("-" * 70)
    print(f"{'Train total':<21} {total_train_clips:>8,} {total_train_hours:>7.1f}h")
    print("=" * 70)

    tokenizer_dir = OUTPUT_DIR / "tokenizer_spe_bpe_v1024"
    tarred_dir    = OUTPUT_DIR / "train_tarred"
    manifests_arg = ",".join(str(m) for m in train_manifests)
    print(f"""
Manifests : {OUTPUT_DIR}
Merged    : {merged_path}

── Next steps ──────────────────────────────────────────────────────

1. Build shared multilingual tokenizer:
   python ${{NEMO_ROOT}}/scripts/tokenizers/process_asr_text_tokenizer.py \\
     --manifest={manifests_arg} \\
     --vocab_size=1024 \\
     --data_root={tokenizer_dir} \\
     --tokenizer=spe --spe_type=bpe \\
     --spe_character_coverage=1.0 \\
     --log

2. Create tarred dataset:
   python ${{NEMO_ROOT}}/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \\
     --manifest_path={merged_path} \\
     --target_dir={tarred_dir} \\
     --num_shards=128 \\
     --max_duration=15.0 --min_duration=1.0 \\
     --shuffle --shuffle_seed=42 --workers=-1

3. NeMo training config:
   model:
     tokenizer:
       dir: {tokenizer_dir}
       type: bpe
     train_ds:
       is_tarred: true
       tarred_audio_filepaths: {tarred_dir}/audio__OP_0..127_CL_.tar
       manifest_filepath: {tarred_dir}/tarred_audio_manifest.json
     validation_ds:
       manifest_filepath:
         - {OUTPUT_DIR}/nl/validation.json
         - {OUTPUT_DIR}/de/validation.json
         - {OUTPUT_DIR}/pl/validation.json
     test_ds:
       manifest_filepath:
         - {OUTPUT_DIR}/nl/test.json
         - {OUTPUT_DIR}/de/test.json
         - {OUTPUT_DIR}/pl/test.json
""")


if __name__ == "__main__":
    main()


# """
# prepare_cv25_nemo.py
# ====================
# Complete pipeline: download CV 25.0 (NL / DE / PL) → quality filter →
# balance hours across languages → resample MP3→WAV → NeMo JSONL manifests.

# Paths (hardcoded for this project):
#     Raw CV downloads : /data/cv/
#     Script location  : /lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice/
#     .env             : /lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice/.env
#     NeMo manifests   : /data/cv/nemo/

# Prerequisites
# -------------
#     pip install datacollective mutagen tqdm pandas sox
#     sudo apt-get install ffmpeg sox

# Usage
# -----
#     cd /lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice

#     # Dry run first — reports actual hours without resampling
#     python prepare_cv_nemo.py --dry_run

#     # Full pipeline
#     python prepare_cv_nemo.py --num_workers 16

#     # Skip download if archives already present in /data/cv/raw/
#     python prepare_cv_nemo.py --skip_download --num_workers 16

#     # Force a specific hour target instead of auto-balancing
#     python prepare_cv_nemo.py --target_hours 50 --num_workers 16

# Output layout
# -------------
#     /data/cv/
#     ├── raw/                        Downloaded + extracted CV 25.0 archives
#     │   ├── nl/clips/*.mp3
#     │   ├── de/clips/*.mp3
#     │   └── pl/clips/*.mp3
#     └── nemo/
#         ├── nl/
#         │   ├── wav/                16kHz mono WAV files
#         │   ├── train.json          NeMo manifest
#         │   ├── validation.json
#         │   └── test.json
#         ├── de/  (same structure)
#         ├── pl/  (same structure)
#         └── train_merged.json       All 3 languages combined (for tokenizer)
# """

# import argparse
# import json
# import logging
# import multiprocessing
# import os
# import re
# import subprocess
# import sys
# import tarfile
# from collections import defaultdict
# from pathlib import Path
# import tqdm

# import pandas as pd
# from tqdm.contrib.concurrent import process_map

# # ─────────────────────────────────────────────────────────────────────────────
# # Project paths — edit here if anything moves
# # ─────────────────────────────────────────────────────────────────────────────

# SCRIPT_DIR  = Path("/lp-dev/amelia/inclusive-asr-moe/preprocessing/common_voice")
# DATA_DIR      = Path("/data/cv")            # Base directory (tar.gz + extracted)
# RAW_DIR       = DATA_DIR / "raw"            # CV archives extracted here
# OUTPUT_DIR    = DATA_DIR / "nemo"           # NeMo manifests + WAVs written here
# ENV_FILE      = SCRIPT_DIR / ".env"

# # ─────────────────────────────────────────────────────────────────────────────

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s",
#     datefmt="%H:%M:%S",
# )
# log = logging.getLogger(__name__)


# # ─────────────────────────────────────────────────────────────────────────────
# # Dataset IDs — CV 25.0 on Mozilla Data Collective
# # ─────────────────────────────────────────────────────────────────────────────

# DATASET_IDS = {
#     "nl": "cmn2g7nu901fmo107a1ydn0n5",
#     "de": "cmn4rsdh6009unz07jdn2ol9p",
#     "pl": "cmn27nz69015hmm0720txf781",
# }

# LANGUAGE_NAMES = {
#     "nl": "Dutch",
#     "de": "German",
#     "pl": "Polish",
# }

# SPLIT_MAP = {
#     "train":      "train.tsv",
#     "validation": "dev.tsv",
#     "test":       "test.tsv",
# }

# # ─────────────────────────────────────────────────────────────────────────────
# # Quality thresholds
# # ─────────────────────────────────────────────────────────────────────────────

# MIN_UPVOTES      = 2      # Hard minimum vote count
# MAX_DOWNVOTES    = 0      # Any downvote disqualifies
# MIN_DURATION_S   = 1.0   # Drop clips shorter than this
# MAX_DURATION_S   = 15.0  # Drop clips longer than this
# MIN_TEXT_WORDS   = 2     # Drop single-word / empty transcriptions


# def quality_score(row) -> float:
#     """
#     Continuous quality score in [0, 1].
#     Rewards many upvotes, penalises downvotes.
#     Used to rank German clips for downsampling.
#     """
#     up   = float(row.get("up_votes")   or 0)
#     down = float(row.get("down_votes") or 0)
#     return up / (up + down + 1)


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 0: Extract tar.gz + Download (if needed)
# # ─────────────────────────────────────────────────────────────────────────────

# def extract_tar_gz(lang_code: str, force_reextract: bool = False) -> Path:
#     """
#     Extract tar.gz if it exists in DATA_DIR, otherwise download.
#     Returns the extracted language directory.
#     """
#     tar_path = DATA_DIR / f"common_voice_{lang_code}.tar.gz"
#     lang_dir_candidate = RAW_DIR / lang_code
    
#     # Check if already extracted
#     if lang_dir_candidate.exists() and (lang_dir_candidate / "train.tsv").exists():
#         if not force_reextract:
#             log.info(f"[{lang_code}] Already extracted at {lang_dir_candidate}")
#             return lang_dir_candidate
    
#     # Try to extract tar.gz if it exists
#     if tar_path.exists():
#         log.info(f"[{lang_code}] Found tar.gz at {tar_path}, extracting ...")
#         RAW_DIR.mkdir(parents=True, exist_ok=True)
#         try:
#             with tarfile.open(tar_path, "r:gz") as tar:
#                 tar.extractall(RAW_DIR)
#             log.info(f"[{lang_code}] Extraction complete")
#             # Find the extracted directory
#             return _find_lang_dir(RAW_DIR, lang_code)
#         except Exception as e:
#             log.error(f"[{lang_code}] Failed to extract: {e}")
#             log.info(f"[{lang_code}] Falling back to download...")
    
#     # Fallback to download
#     log.info(f"[{lang_code}] tar.gz not found, downloading...")
#     return download_language(lang_code)


# def _load_env() -> None:
#     """Load MDC_API_KEY from .env if not already in environment."""
#     if os.environ.get("MDC_API_KEY"):
#         return
#     if not ENV_FILE.exists():
#         log.warning(
#             f".env file not found at {ENV_FILE}. "
#             f"Download functionality unavailable, but extraction will work if tar.gz exists."
#         )
#         return
#     with open(ENV_FILE) as f:
#         for line in f:
#             line = line.strip()
#             if line and not line.startswith("#") and "=" in line:
#                 key, _, value = line.partition("=")
#                 os.environ.setdefault(key.strip(), value.strip())


# def download_language(lang_code: str) -> Path:
#     """Download CV 25.0 archive directly via the MDC REST API (no SDK needed)."""
#     import requests

#     _load_env()
#     api_key = os.environ.get("MDC_API_KEY", "")
#     if not api_key:
#         raise RuntimeError(
#             f"MDC_API_KEY not set. Cannot download {lang_code}. "
#             f"Please set it in {ENV_FILE} or environment."
#         )
#     dataset_id = DATASET_IDS[lang_code]
#     RAW_DIR.mkdir(parents=True, exist_ok=True)

#     log.info(f"[{lang_code}] Requesting download URL for {LANGUAGE_NAMES[lang_code]} ...")

#     # Step 1: POST to get a presigned download URL
#     resp = requests.post(
#         f"https://datacollective.mozillafoundation.org/api/datasets/{dataset_id}/download",
#         headers={
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json",
#         },
#         timeout=30,
#     )
#     resp.raise_for_status()
#     payload      = resp.json()
#     download_url = payload["downloadUrl"]
#     filename     = payload["filename"]
#     size_bytes   = int(payload["sizeBytes"])
#     archive_path = RAW_DIR / filename

#     log.info(f"[{lang_code}] Downloading {filename} ({size_bytes/1e9:.1f} GB) → {archive_path}")

#     # Step 2: Stream download with progress bar (resumable)
#     headers = {}
#     downloaded = 0
#     if archive_path.exists():
#         downloaded = archive_path.stat().st_size
#         if downloaded >= size_bytes:
#             log.info(f"[{lang_code}] Archive already fully downloaded, skipping.")
#         else:
#             log.info(f"[{lang_code}] Resuming from byte {downloaded:,}")
#             headers["Range"] = f"bytes={downloaded}-"

#     if downloaded < size_bytes:
#         with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
#             r.raise_for_status()
#             mode = "ab" if downloaded > 0 else "wb"
#             with open(archive_path, mode) as f:
#                 for chunk in tqdm(
#                     r.iter_content(chunk_size=1 << 20),  # 1 MB chunks
#                     desc=f"  {lang_code}",
#                     unit="MB",
#                     initial=downloaded // (1 << 20),
#                     total=size_bytes // (1 << 20),
#                 ):
#                     if chunk:
#                         f.write(chunk)

#         log.info(f"[{lang_code}] Download complete: {archive_path}")

#     # Step 3: Extract the .tar.gz
#     lang_dir_candidate = RAW_DIR / lang_code
#     if lang_dir_candidate.exists() and (lang_dir_candidate / "train.tsv").exists():
#         log.info(f"[{lang_code}] Already extracted at {lang_dir_candidate}, skipping.")
#         return lang_dir_candidate

#     log.info(f"[{lang_code}] Extracting {filename} ...")
#     with tarfile.open(archive_path, "r:gz") as tar:
#         tar.extractall(RAW_DIR)
#     log.info(f"[{lang_code}] Extraction complete.")

#     return _find_lang_dir(RAW_DIR, lang_code)


# # def _find_lang_dir(root: Path, lang_code: str) -> Path:
# #     """Locate the extracted CV directory containing train.tsv + clips/."""
# #     for candidate in root.rglob("train.tsv"):
# #         parent = candidate.parent
# #         if (parent / "clips").exists():
# #             return parent
# #     raise FileNotFoundError(
# #         f"Could not find extracted CV directory for '{lang_code}' under {root}.\n"
# #         f"Expected a folder containing train.tsv and clips/."
# #     )

# def _find_lang_dir(root: Path, lang_code: str) -> Path:
#     """Locate the extracted CV directory for a specific language code."""
#     # CV 25.0 extracts to: cv-corpus-VERSION/LANG_CODE/
#     # Must match the directory named exactly lang_code
#     for candidate in root.rglob("train.tsv"):
#         parent = candidate.parent
#         if parent.name == lang_code and (parent / "clips").exists():
#             return parent
#     raise FileNotFoundError(
#         f"Could not find extracted CV directory for \'{lang_code}\' under {root}.\\n"
#         f"Expected a folder named \'{lang_code}\' containing train.tsv and clips/."
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 1: Fast duration measurement (reads MP3 headers, no decoding)
# # ─────────────────────────────────────────────────────────────────────────────

# def _get_duration_fast(mp3_path: str) -> float:
#     try:
#         from mutagen.mp3 import MP3
#         return MP3(mp3_path).info.length
#     except Exception:
#         pass
#     try:
#         result = subprocess.run(
#             ["soxi", "-D", mp3_path],
#             capture_output=True, text=True, check=True,
#         )
#         return float(result.stdout.strip())
#     except Exception:
#         return -1.0


# def _duration_worker(args):
#     (path,) = args
#     return path, _get_duration_fast(path)


# def measure_durations(paths: list, num_workers: int) -> dict:
#     """Return {path: duration_seconds} for all paths, in parallel."""
#     jobs = [(p,) for p in paths]
#     results = process_map(
#         _duration_worker, jobs,
#         max_workers=num_workers, chunksize=200,
#         desc="    Measuring durations",
#     )
#     return {r[0]: r[1] for r in results if r[1] > 0}


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 2: Quality filtering
# # ─────────────────────────────────────────────────────────────────────────────

# def load_tsv(tsv_path: Path) -> pd.DataFrame:
#     return pd.read_csv(
#         tsv_path, sep="\t", low_memory=False,
#         dtype={
#             "client_id":  str,
#             "path":       str,
#             "sentence":   str,
#             "up_votes":   "Int64",
#             "down_votes": "Int64",
#             "age":        str,
#             "gender":     str,
#             "accents":    str,
#         },
#     )


# def apply_quality_filter(
#     df: pd.DataFrame,
#     clips_dir: Path,
#     duration_cache: dict,
#     lang_code: str,
#     split_name: str,
# ) -> pd.DataFrame:
#     n_raw = len(df)

#     # Vote thresholds
#     df = df[
#         (df["up_votes"].fillna(0)   >= MIN_UPVOTES) &
#         (df["down_votes"].fillna(999) <= MAX_DOWNVOTES)
#     ]

#     # Non-empty sentence with minimum word count
#     df = df[df["sentence"].notna() & (df["sentence"].str.strip() != "")]
#     df = df[df["sentence"].str.split().str.len() >= MIN_TEXT_WORDS]

#     # Audio file must exist on disk
#     df = df.copy()
#     df["_full_path"] = df["path"].apply(
#         lambda p: str(clips_dir / (p if p.endswith(".mp3") else p + ".mp3"))
#     )
#     df = df[df["_full_path"].apply(lambda p: Path(p).exists())]

#     # Attach duration from cache and apply duration bounds
#     df["duration"] = df["_full_path"].map(duration_cache)
#     df = df[df["duration"].notna()]
#     df["duration"] = df["duration"].astype(float)
#     df = df[(df["duration"] >= MIN_DURATION_S) & (df["duration"] <= MAX_DURATION_S)]

#     # Attach continuous quality score
#     df["quality_score"] = df.apply(quality_score, axis=1)

#     n_kept = len(df)
#     hrs    = df["duration"].sum() / 3600
#     log.info(
#         f"      [{lang_code}/{split_name}] {n_raw:,} raw → {n_kept:,} kept "
#         f"({hrs:.1f}h, up≥{MIN_UPVOTES} down≤{MAX_DOWNVOTES} dur {MIN_DURATION_S}–{MAX_DURATION_S}s)"
#     )
#     return df.reset_index(drop=True)


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 3: Hour-balanced, speaker-stratified downsampling
# # ─────────────────────────────────────────────────────────────────────────────

# # def balance_to_target_hours(
# #     df: pd.DataFrame,
# #     target_hours: float,
# #     lang_code: str,
# #     split_name: str,
# #     max_clips_per_speaker: int = 150,
# # ) -> pd.DataFrame:
# #     """
# #     Downsample df to target_hours.
# #     - Ranks clips by quality_score (best first)
# #     - Enforces per-speaker cap to preserve diversity
# #     - Returns df unchanged if already within target
# #     """
# #     available = df["duration"].sum() / 3600

# #     if available <= target_hours:
# #         log.info(
# #             f"      [{lang_code}/{split_name}] {available:.1f}h ≤ target "
# #             f"{target_hours:.1f}h — keeping all"
# #         )
# #         return df

# #     log.info(
# #         f"      [{lang_code}/{split_name}] Downsampling "
# #         f"{available:.1f}h → {target_hours:.1f}h "
# #         f"(quality-ranked, speaker cap={max_clips_per_speaker})"
# #     )

# #     df_sorted = df.sort_values("quality_score", ascending=False)
# #     selected = []
# #     speaker_counts: dict = defaultdict(int)
# #     cumulative_hours = 0.0

# #     for _, row in df_sorted.iterrows():
# #         if cumulative_hours >= target_hours:
# #             break
# #         spk = row["client_id"]
# #         if speaker_counts[spk] >= max_clips_per_speaker:
# #             continue
# #         selected.append(row)
# #         speaker_counts[spk] += 1
# #         cumulative_hours += row["duration"] / 3600

# #     result = pd.DataFrame(selected).reset_index(drop=True)
# #     log.info(
# #         f"      [{lang_code}/{split_name}] Selected {len(result):,} clips | "
# #         f"{cumulative_hours:.1f}h | {result['client_id'].nunique():,} speakers"
# #     )
# #     return result


# def balance_to_target_hours(
#     df: pd.DataFrame,
#     target_hours: float,
#     lang_code: str,
#     split_name: str,
#     max_clips_per_speaker: int = 150,
# ) -> pd.DataFrame:
#     """
#     Downsample df to target_hours.
#     - Adaptive speaker cap: raised automatically for low-speaker languages so
#       the cap never prevents reaching target_hours when the data is there.
#     - Ranks clips by quality_score (best first).
#     - Enforces per-speaker cap to preserve diversity.
#     - Returns df unchanged if already within target.
#     """
#     available = df["duration"].sum() / 3600

#     if available <= target_hours:
#         log.info(
#             f"      [{lang_code}/{split_name}] {available:.1f}h ≤ target "
#             f"{target_hours:.1f}h — keeping all"
#         )
#         return df

#     # ── Adaptive cap ──────────────────────────────────────────────────────────
#     # If the hard cap would prevent us from reaching target_hours even using
#     # all speakers at full cap, raise it to the minimum needed.
#     n_speakers = df["client_id"].nunique()
#     avg_dur_per_clip = df["duration"].mean()  # seconds
#     clips_needed_total = int((target_hours * 3600) / avg_dur_per_clip) + 1
#     min_cap_needed = -(-clips_needed_total // n_speakers)  # ceiling division

#     effective_cap = max(max_clips_per_speaker, min_cap_needed)
#     if effective_cap > max_clips_per_speaker:
#         log.info(
#             f"      [{lang_code}/{split_name}] Raising speaker cap "
#             f"{max_clips_per_speaker} → {effective_cap} "
#             f"({n_speakers} speakers, need ≥{min_cap_needed} clips/speaker "
#             f"to reach {target_hours:.1f}h)"
#         )
#     # ─────────────────────────────────────────────────────────────────────────

#     log.info(
#         f"      [{lang_code}/{split_name}] Downsampling "
#         f"{available:.1f}h → {target_hours:.1f}h "
#         f"(quality-ranked, speaker cap={effective_cap})"
#     )

#     df_sorted = df.sort_values("quality_score", ascending=False)
#     selected = []
#     speaker_counts: dict = defaultdict(int)
#     cumulative_hours = 0.0

#     for _, row in df_sorted.iterrows():
#         if cumulative_hours >= target_hours:
#             break
#         spk = row["client_id"]
#         if speaker_counts[spk] >= effective_cap:
#             continue
#         selected.append(row)
#         speaker_counts[spk] += 1
#         cumulative_hours += row["duration"] / 3600

#     result = pd.DataFrame(selected).reset_index(drop=True)
#     log.info(
#         f"      [{lang_code}/{split_name}] Selected {len(result):,} clips | "
#         f"{cumulative_hours:.1f}h | {result['client_id'].nunique():,} speakers"
#     )
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 4: Language-specific text normalisation
# # ─────────────────────────────────────────────────────────────────────────────

# def _base_clean(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[–—−‐‑‒]",             " ",  text)
#     text = re.sub(r"[''`ʽ´ʻ]",             "'",  text)
#     text = re.sub(r'[""„‟″«»]',            "",   text)
#     text = re.sub(r"[\u00AD\u200B-\u200D\uFEFF]", "", text)
#     text = re.sub(r" +", " ", text).strip()
#     return text


# def normalize_nl(text: str) -> str:
#     """Dutch: letters + Dutch diacritics + apostrophe (for contractions like 't, 's)."""
#     text = _base_clean(text)
#     text = re.sub(r"[^a-zàâäéèêëîïôöùûüÿçæœ' ]", " ", text)
#     text = re.sub(r"(?<![a-zàâäéèêëîïôöùûüÿçæœ])'|'(?![a-zàâäéèêëîïôöùûüÿçæœ])", " ", text)
#     return re.sub(r" +", " ", text).strip()


# def normalize_de(text: str) -> str:
#     """German: letters + umlauts (ä ö ü) + ß."""
#     text = _base_clean(text)
#     text = re.sub(r"[^a-zäöüß ]", " ", text)
#     return re.sub(r" +", " ", text).strip()


# def normalize_pl(text: str) -> str:
#     """Polish: letters + Polish diacritics (ą ć ę ł ń ó ś ź ż)."""
#     text = _base_clean(text)
#     text = re.sub(r"[^a-ząćęłńóśźż ]", " ", text)
#     return re.sub(r" +", " ", text).strip()


# NORMALIZERS = {"nl": normalize_nl, "de": normalize_de, "pl": normalize_pl}


# def normalize_texts(df: pd.DataFrame, lang_code: str) -> pd.DataFrame:
#     normalizer = NORMALIZERS[lang_code]
#     df = df.copy()
#     df["text"] = df["sentence"].apply(
#         lambda t: normalizer(str(t) if pd.notna(t) else "")
#     )
#     before = len(df)
#     df = df[df["text"].str.strip() != ""].reset_index(drop=True)
#     dropped = before - len(df)
#     if dropped:
#         log.warning(f"      Dropped {dropped} entries with empty text after normalisation")
#     return df


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 5: Resample MP3 → 16 kHz mono WAV
# # ─────────────────────────────────────────────────────────────────────────────

# def _resample_worker(args):
#     src_str, dst_str = args
#     src, dst = Path(src_str), Path(dst_str)

#     if dst.exists():
#         # Already done — reuse
#         try:
#             import sox
#             return src_str, dst_str, sox.file_info.duration(str(dst))
#         except Exception:
#             pass

#     dst.parent.mkdir(parents=True, exist_ok=True)

#     # Try sox first (fastest)
#     try:
#         import sox
#         tfm = sox.Transformer()
#         tfm.rate(samplerate=16000)
#         tfm.channels(n_channels=1)
#         tfm.build(input_filepath=str(src), output_filepath=str(dst))
#         return src_str, dst_str, sox.file_info.duration(str(dst))
#     except Exception:
#         pass

#     # Fallback: ffmpeg
#     try:
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", str(src),
#              "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(dst)],
#             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
#         )
#         r = subprocess.run(
#             ["soxi", "-D", str(dst)], capture_output=True, text=True, check=True
#         )
#         return src_str, dst_str, float(r.stdout.strip())
#     except Exception:
#         return src_str, dst_str, None


# def resample_audio(df: pd.DataFrame, wav_dir: Path, num_workers: int) -> pd.DataFrame:
#     wav_dir.mkdir(parents=True, exist_ok=True)

#     jobs = [
#         (row["_full_path"],
#          str(wav_dir / (Path(row["_full_path"]).stem + ".wav")))
#         for _, row in df.iterrows()
#     ]

#     results = process_map(
#         _resample_worker, jobs,
#         max_workers=num_workers, chunksize=50, desc="    Resample",
#     )

#     lookup = {r[0]: (r[1], r[2]) for r in results}
#     records, failed = [], 0

#     for _, row in df.iterrows():
#         wav_path, dur = lookup.get(row["_full_path"], (None, None))
#         if not wav_path or not dur or dur <= 0:
#             failed += 1
#             continue
#         rec = row.to_dict()
#         rec["audio_filepath"] = wav_path
#         rec["duration"]       = round(float(dur), 4)
#         records.append(rec)

#     if failed:
#         log.warning(f"    {failed} clips failed resampling and were dropped")

#     return pd.DataFrame(records).reset_index(drop=True)


# # ─────────────────────────────────────────────────────────────────────────────
# # Stage 6: Write NeMo JSONL manifest
# # ─────────────────────────────────────────────────────────────────────────────

# def write_manifest(df: pd.DataFrame, output_path: Path, lang_code: str) -> None:
#     """
#     Write NeMo-compatible JSONL manifest.

#     Required fields : audio_filepath, text, duration
#     Optional fields : speaker, language, age_group, age, gender, accents,
#                       up_votes, down_votes  (kept for RQ2 routing analysis)
#     """
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     written = 0

#     with open(output_path, "w", encoding="utf-8") as f:
#         for _, row in df.iterrows():
#             if not row.get("audio_filepath") or not row.get("text") or not row.get("duration"):
#                 continue
#             record = {
#                 "audio_filepath": str(row["audio_filepath"]),
#                 "text":           str(row["text"]),
#                 "duration":       float(row["duration"]),
#                 "speaker":        str(row.get("client_id", "")),
#                 "language":       lang_code,
#                 "age_group":      "adult",
#             }
#             for opt in ["age", "gender", "accents", "up_votes", "down_votes"]:
#                 val = row.get(opt)
#                 if val is not None and not (isinstance(val, float) and pd.isna(val)):
#                     record[opt] = val
#             f.write(json.dumps(record, ensure_ascii=False) + "\n")
#             written += 1

#     hrs = df["duration"].sum() / 3600 if "duration" in df.columns else 0
#     spk = df["client_id"].nunique()    if "client_id" in df.columns else 0
#     log.info(f"    → {output_path}: {written:,} clips | {hrs:.1f}h | {spk:,} speakers")


# # ─────────────────────────────────────────────────────────────────────────────
# # Speaker-independence check
# # ─────────────────────────────────────────────────────────────────────────────

# def verify_speaker_independence(lang_output_dir: Path, lang_code: str) -> None:
#     splits = {}
#     for split in ["train", "validation", "test"]:
#         p = lang_output_dir / f"{split}.json"
#         if not p.exists():
#             continue
#         spks = set()
#         with open(p) as f:
#             for line in f:
#                 spks.add(json.loads(line).get("speaker", ""))
#         splits[split] = spks

#     names = list(splits.keys())
#     clean = True
#     for i in range(len(names)):
#         for j in range(i + 1, len(names)):
#             a, b = names[i], names[j]
#             overlap = splits[a] & splits[b]
#             if overlap:
#                 log.warning(f"  [{lang_code}] SPEAKER LEAK: {a} ∩ {b} = {len(overlap)} speakers!")
#                 clean = False
#             else:
#                 log.info(f"  [{lang_code}] OK: {a} ∩ {b} = 0 shared speakers")
#     if clean:
#         log.info(f"  [{lang_code}] Speaker independence verified ✓")


# # ─────────────────────────────────────────────────────────────────────────────
# # Per-language pipeline
# # ─────────────────────────────────────────────────────────────────────────────

# def process_language(
#     lang_code: str,
#     lang_dir: Path,
#     target_hours_train: float,
#     num_workers: int,
#     dry_run: bool,
# ) -> dict:
#     clips_dir   = lang_dir / "clips"
#     lang_output = OUTPUT_DIR / lang_code
#     wav_dir     = lang_output / "wav"

#     log.info(f"\n{'='*60}")
#     log.info(f"  {LANGUAGE_NAMES[lang_code]} ({lang_code.upper()})")
#     log.info(f"  Source : {lang_dir}")
#     log.info(f"  Output : {lang_output}")
#     log.info(f"{'='*60}")

#     stats = {"lang": lang_code, "splits": {}}

#     # Pre-measure all durations once (fast — reads MP3 headers only)
#     log.info(f"  Pre-measuring MP3 durations ...")
#     all_mp3s = list(clips_dir.glob("*.mp3"))
#     log.info(f"  Found {len(all_mp3s):,} MP3 files in {clips_dir}")
#     duration_cache = measure_durations([str(p) for p in all_mp3s], num_workers)

#     for split_name, tsv_file in SPLIT_MAP.items():
#         tsv_path = lang_dir / tsv_file
#         if not tsv_path.exists():
#             log.warning(f"  [{split_name}] TSV not found: {tsv_path} — skipping")
#             continue

#         log.info(f"\n  [{split_name}]")
#         df = load_tsv(tsv_path)
#         df = apply_quality_filter(df, clips_dir, duration_cache, lang_code, split_name)

#         # Balance train only — val/test kept complete for fair evaluation
#         if split_name == "train":
#             df = balance_to_target_hours(df, target_hours_train, lang_code, split_name)

#         df = normalize_texts(df, lang_code)

#         hrs = df["duration"].sum() / 3600
#         spk = df["client_id"].nunique()
#         stats["splits"][split_name] = {"clips": len(df), "hours": hrs, "speakers": spk}

#         if dry_run:
#             log.info(f"  [DRY RUN] {split_name}: {len(df):,} clips | {hrs:.1f}h | {spk:,} speakers")
#             continue

#         log.info(f"  Resampling {len(df):,} clips to 16kHz WAV ...")
#         df = resample_audio(df, wav_dir, num_workers)
#         write_manifest(df, lang_output / f"{split_name}.json", lang_code)

#     if not dry_run:
#         verify_speaker_independence(lang_output, lang_code)

#     return stats


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="CV 25.0 NL/DE/PL → quality-filtered, balanced NeMo manifests\n"
#                     f"  Data dir      : {DATA_DIR}\n"
#                     f"  Raw extracted : {RAW_DIR}\n"
#                     f"  NeMo output   : {OUTPUT_DIR}",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#     )
#     parser.add_argument(
#         "--languages", nargs="+", default=["nl", "de", "pl"],
#         help="Languages to process (default: nl de pl)",
#     )
#     parser.add_argument(
#         "--target_hours", type=float, default=None,
#         help="Training hours per language. Default: auto (= smallest language after filtering)",
#     )
#     parser.add_argument(
#         "--num_workers", type=int,
#         default=max(1, multiprocessing.cpu_count() - 2),
#         help="Parallel workers for duration measurement and resampling",
#     )
#     parser.add_argument(
#         "--dry_run", action="store_true",
#         help="Measure hours and report — no resampling, no files written",
#     )
#     parser.add_argument(
#         "--force_reextract", action="store_true",
#         help="Force re-extraction of tar.gz even if already extracted",
#     )
#     args = parser.parse_args()

#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     RAW_DIR.mkdir(parents=True, exist_ok=True)

#     log.info("CV 25.0 NeMo Preparation Pipeline")
#     log.info(f"  Languages       : {args.languages}")
#     log.info(f"  Data dir        : {DATA_DIR}")
#     log.info(f"  Raw dir         : {RAW_DIR}")
#     log.info(f"  Output dir      : {OUTPUT_DIR}")
#     log.info(f"  Workers         : {args.num_workers}")
#     log.info(f"  Dry run         : {args.dry_run}")
#     log.info(f"  Force re-extract: {args.force_reextract}")

#     # ── Locate, extract, or download archives ─────────────────────────────────
#     lang_dirs = {}
#     for lang_code in args.languages:
#         try:
#             # Try: (1) extract if tar.gz exists, (2) find if already extracted, (3) download
#             lang_dirs[lang_code] = extract_tar_gz(lang_code, force_reextract=False)
#             log.info(f"[{lang_code}] Ready at {lang_dirs[lang_code]}")
#         except FileNotFoundError as e:
#             log.error(f"[{lang_code}] {e}")

#     if not lang_dirs:
#         log.error("No language directories found. Exiting.")
#         sys.exit(1)

#     # ── Auto-detect target hours if not specified ──────────────────────────
#     if args.target_hours is None:
#         log.info("\n── Probing available hours for auto-balancing ────────────────")
#         available: dict = {}

#         for lang_code, lang_dir in lang_dirs.items():
#             clips_dir = lang_dir / "clips"
#             tsv_path  = lang_dir / "train.tsv"
#             if not tsv_path.exists():
#                 continue
#             df = load_tsv(tsv_path)
#             paths = [
#                 str(clips_dir / (p if p.endswith(".mp3") else p + ".mp3"))
#                 for p in df["path"].dropna()
#             ]
#             existing = [p for p in paths if Path(p).exists()]
#             dur_cache = measure_durations(existing, args.num_workers)
#             df = apply_quality_filter(df, clips_dir, dur_cache, lang_code, "train")
#             hrs = df["duration"].sum() / 3600
#             available[lang_code] = hrs
#             log.info(f"  {LANGUAGE_NAMES[lang_code]}: {hrs:.1f}h after quality filter")

#         target_hours = min(available.values())
#         bottleneck   = min(available, key=available.get)
#         log.info(
#             f"\n  Auto target : {target_hours:.1f}h "
#             f"(bottleneck: {LANGUAGE_NAMES[bottleneck]})\n"
#             f"  German will be downsampled by quality score to match.\n"
#         )

#         if args.dry_run:
#             log.info("── Dry run results ───────────────────────────────────────────")
#             for lang, hrs in available.items():
#                 action = "keep all" if hrs <= target_hours \
#                          else f"downsample → {target_hours:.1f}h"
#                 log.info(f"  {LANGUAGE_NAMES[lang]:<8}: {hrs:.1f}h → {action}")
#             log.info(
#                 f"\nRe-run without --dry_run to execute.\n"
#                 f"Or override with --target_hours N to use a specific value."
#             )
#             return
#     else:
#         target_hours = args.target_hours
#         log.info(f"  Using user-specified target: {target_hours:.1f}h per language")

#     # ── Full per-language pipeline ─────────────────────────────────────────
#     all_stats = []
#     for lang_code, lang_dir in lang_dirs.items():
#         stats = process_language(
#             lang_code=lang_code,
#             lang_dir=lang_dir,
#             target_hours_train=target_hours,
#             num_workers=args.num_workers,
#             dry_run=args.dry_run,
#         )
#         all_stats.append(stats)

#     if args.dry_run:
#         return

#     # ── Merge train manifests ──────────────────────────────────────────────
#     train_manifests = [
#         OUTPUT_DIR / lc / "train.json"
#         for lc in lang_dirs
#         if (OUTPUT_DIR / lc / "train.json").exists()
#     ]
#     merged_path = OUTPUT_DIR / "train_merged.json"

#     if train_manifests:
#         log.info(f"\nMerging {len(train_manifests)} train manifests → {merged_path}")
#         with open(merged_path, "w", encoding="utf-8") as fout:
#             for mp in train_manifests:
#                 with open(mp, encoding="utf-8") as fin:
#                     fout.writelines(fin)

#     # ── Print final summary ────────────────────────────────────────────────
#     print("\n" + "=" * 70)
#     print(f"{'FINAL SUMMARY — CV 25.0 NeMo Data':^70}")
#     print("=" * 70)
#     print(f"{'Lang':<8} {'Split':<12} {'Clips':>8} {'Hours':>8} {'Speakers':>10}")
#     print("-" * 70)
#     total_clips, total_hours = 0, 0.0
#     for stats in all_stats:
#         for split, info in stats["splits"].items():
#             print(f"{stats['lang']:<8} {split:<12} {info['clips']:>8,} "
#                   f"{info['hours']:>7.1f}h {info['speakers']:>10,}")
#             if split == "train":
#                 total_clips += info["clips"]
#                 total_hours += info["hours"]
#     print("-" * 70)
#     print(f"{'Train total':<21} {total_clips:>8,} {total_hours:>7.1f}h")
#     print("=" * 70)

#     print(f"\nManifests : {OUTPUT_DIR}")
#     print(f"Merged    : {merged_path}")

#     print("\n── Next steps ──────────────────────────────────────────────────────")
#     manifests_arg = ",".join(str(m) for m in train_manifests)
#     tokenizer_dir = OUTPUT_DIR / "tokenizer_spe_bpe_v1024"
#     tarred_dir    = OUTPUT_DIR / "train_tarred"
#     print(f"""
# 1. Build shared multilingual tokenizer:
#    python ${{NEMO_ROOT}}/scripts/tokenizers/process_asr_text_tokenizer.py \\
#      --manifest={manifests_arg} \\
#      --vocab_size=1024 \\
#      --data_root={tokenizer_dir} \\
#      --tokenizer=spe --spe_type=bpe \\
#      --spe_character_coverage=1.0 \\
#      --spe_max_sentencepiece_length=4 \\
#      --log

# 2. Create tarred dataset for cluster training:
#    python ${{NEMO_ROOT}}/scripts/speech_recognition/convert_to_tarred_audio_dataset.py \\
#      --manifest_path={merged_path} \\
#      --target_dir={tarred_dir} \\
#      --num_shards=256 \\
#      --max_duration=15.0 --min_duration=1.0 \\
#      --shuffle --shuffle_seed=42 \\
#      --sort_in_shards --workers=-1

# 3. NeMo training config snippet:
#    model:
#      tokenizer:
#        dir: {tokenizer_dir}
#        type: bpe
#      train_ds:
#        is_tarred: true
#        tarred_audio_filepaths: {tarred_dir}/audio__OP_0..255_CL_.tar
#        manifest_filepath: {tarred_dir}/tarred_audio_manifest.json
#      validation_ds:
#        manifest_filepath:
#          - {OUTPUT_DIR}/nl/validation.json
#          - {OUTPUT_DIR}/de/validation.json
#          - {OUTPUT_DIR}/pl/validation.json
#      test_ds:
#        manifest_filepath:
#          - {OUTPUT_DIR}/nl/test.json
#          - {OUTPUT_DIR}/de/test.json
#          - {OUTPUT_DIR}/pl/test.json
# """)


# if __name__ == "__main__":
#     main()