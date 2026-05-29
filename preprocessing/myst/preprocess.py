#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import inflect
import ray
import soundfile as sf
from whisper_normalizer.english import EnglishTextNormalizer


logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
LOG = logging.getLogger("myst_preprocess")

# Ensure aggressive flushing
for handler in LOG.handlers:
    handler.setLevel(logging.INFO)
    handler.flush()

NO_SPEECH = {"<discard>", "<no_signal>", "<silence>", "<noise>", "<non_speech>", ""}


class TextNormalizer:
    def __init__(self):
        self.whisper_norm = EnglishTextNormalizer()
        self.inflect_engine = inflect.engine()
        self.num_re = re.compile(r"\b\d+\b")

    def _numbers_to_words(self, text: str) -> str:
        def repl(match):
            try:
                return self.inflect_engine.number_to_words(int(match.group(0)), andword="")
            except (ValueError, inflect.NumOutOfRangeError):
                # Keep number as-is if it can't be converted
                return match.group(0)

        return self.num_re.sub(repl, text)

    def normalize(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = self.whisper_norm(text)
        text = self._numbers_to_words(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


@ray.remote(num_gpus=1, max_restarts=1)
class WhisperTranscriber:
    """Ray actor for parallel Whisper transcription. Each actor gets one GPU."""
    
    def __init__(self, model_size: str = "large", gpu_id: int = 0):
        import whisper
        self.model_size = model_size
        self.gpu_id = gpu_id
        self.whisper_model = None
        self.norm = TextNormalizer()
        
    def _load_whisper(self):
        if self.whisper_model is None:
            import whisper
            LOG.debug(f"[GPU {self.gpu_id}] Loading Whisper model: {self.model_size}")
            self.whisper_model = whisper.load_model(self.model_size)
    
    def transcribe_batch(self, items: List[Dict]) -> List[Dict]:
        """Process a batch of items and return results with transcription."""
        self._load_whisper()
        results = []
        for item in items:
            audio_fp = item.get("audio_filepath", "")
            audio_path = Path(audio_fp)
            
            result_item = {
                "audio_filepath": audio_fp,
                "ref_norm": item.get("ref_norm", ""),
                "hyp_norm": "",
                "wer": 100.0,
                "distorted": True,
                "missing_audio": False,
            }
            
            if not audio_path.exists():
                result_item["missing_audio"] = True
                results.append(result_item)
                continue
            
            try:
                hyp = self.whisper_model.transcribe(str(audio_path), language="en").get("text", "").strip()
                hyp_norm = self.norm.normalize(hyp)
                distortion = detect_distortion(audio_path)
                w = wer_percent(item.get("ref_norm", ""), hyp_norm)
                result_item.update({
                    "hyp_norm": hyp_norm,
                    "wer": float(w),
                    "distorted": bool(distortion)
                })
            except Exception as e:
                LOG.warning(f"[GPU {self.gpu_id}] Failed transcription for {audio_fp}: {e}")
            
            results.append(result_item)
        
        return results


def levenshtein_words(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a):
        cur = [i + 1]
        for j, wb in enumerate(b):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (wa != wb)))
        prev = cur
    return prev[-1]


def wer_percent(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 100.0
    return 100.0 * levenshtein_words(r, h) / len(r)


def detect_distortion(audio_path: Path, clip_threshold: float = 0.98, clip_ratio_threshold: float = 0.01) -> bool:
    try:
        audio, _ = sf.read(str(audio_path), always_2d=False)
    except Exception:
        return True

    if audio is None:
        return True

    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        return True

    clip_ratio = float((abs(audio) >= clip_threshold).mean())
    return bool(clip_ratio > clip_ratio_threshold)


def session_key(audio_filepath: str) -> str:
    return str(Path(audio_filepath).parent)


def utterance_idx(audio_filepath: str) -> int:
    stem = Path(audio_filepath).stem
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else 0


def safe_name(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


def concat_audio_ffmpeg(audio_files: List[str], output_audio: Path):
    output_audio.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        for fp in audio_files:
            tf.write(f"file '{fp}'\n")
        list_path = tf.name

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        str(output_audio),
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        Path(list_path).unlink(missing_ok=True)


class MystPreprocessor:
    def __init__(self, model_size: str = "large", dry_run: bool = False):
        self.norm = TextNormalizer()
        self.model_size = model_size
        self.dry_run = dry_run
        self.whisper_model = None

    def _load_whisper(self):
        if self.whisper_model is None:
            import whisper

            LOG.info("Loading Whisper model: %s", self.model_size)
            self.whisper_model = whisper.load_model(self.model_size)

    def _transcribe(self, audio_path: str) -> str:
        self._load_whisper()
        result = self.whisper_model.transcribe(audio_path, language="en")
        return (result.get("text") or "").strip()

    def load_manifest(self, path: Path) -> List[Dict]:
        rows = []
        with path.open("r") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def save_manifest(self, rows: List[Dict], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _load_cache(self, cache_path: Path) -> Dict[str, Dict]:
        cache = {}
        if cache_path.exists():
            with cache_path.open("r") as f:
                for line in f:
                    item = json.loads(line)
                    cache[item["audio_filepath"]] = item
        return cache

    def _append_cache(self, cache_path: Path, item: Dict):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()  # Force write to disk immediately

    def compute_wer_and_quality(
        self,
        rows: List[Dict],
        cache_path: Path,
        num_gpus: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, Dict]:
        """Compute WER and quality metrics in parallel using Ray actors distributed across GPUs."""
        cache = self._load_cache(cache_path)
        
        # Filter rows that aren't in cache
        rows_to_process = [r for r in rows if r.get("audio_filepath", "") not in cache]
        if not rows_to_process:
            LOG.info("All rows already in cache")
            return cache
        
        # Normalize texts and prepare items for batch processing
        items_to_process = []
        for row in rows_to_process:
            ref_norm = self.norm.normalize(row.get("text", ""))
            items_to_process.append({
                "audio_filepath": row.get("audio_filepath", ""),
                "ref_norm": ref_norm,
                "original_row": row,
            })
        
        total = len(items_to_process)
        batch_size = max(1, int(batch_size))
        
        LOG.info("Processing %d items with %d GPUs in batches of %d", total, num_gpus, batch_size)
        
        # Create Ray actors (one per GPU)
        actors = [
            WhisperTranscriber.remote(model_size=self.model_size, gpu_id=gpu_id)
            for gpu_id in range(num_gpus)
        ]
        
        # Submit all batches first
        pending = {}
        for batch_idx in range(0, len(items_to_process), batch_size):
            batch = items_to_process[batch_idx : batch_idx + batch_size]
            actor = actors[(batch_idx // batch_size) % len(actors)]
            ref = actor.transcribe_batch.remote(batch)
            pending[ref] = (batch_idx, batch)

        # Collect batches as they finish (out-of-order) to stream progress
        processed = 0
        while pending:
            done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
            done_ref = done_refs[0]
            batch_idx, batch = pending.pop(done_ref)

            try:
                results = ray.get(done_ref)
                for result in results:
                    self._append_cache(cache_path, result)
                    cache[result["audio_filepath"]] = result
                    processed += 1
                    if processed % 200 == 0 or processed == total:
                        msg = f"WER progress {processed}/{total}"
                        LOG.info(msg)
            except Exception as e:
                LOG.error("Error processing batch at index %d: %s", batch_idx, e)
                for item in batch:
                    result = self._process_item_serial(item)
                    self._append_cache(cache_path, result)
                    cache[result["audio_filepath"]] = result
                    processed += 1
                    if processed % 200 == 0 or processed == total:
                        msg = f"WER progress {processed}/{total}"
                        LOG.info(msg)
        
        return cache
    
    def _process_item_serial(self, item: Dict) -> Dict:
        """Fallback function to process a single item serially."""
        audio_fp = item.get("audio_filepath", "")
        audio_path = Path(audio_fp)
        
        result = {
            "audio_filepath": audio_fp,
            "ref_norm": item.get("ref_norm", ""),
            "hyp_norm": "",
            "wer": 100.0,
            "distorted": True,
            "missing_audio": False,
        }
        
        if not audio_path.exists():
            result["missing_audio"] = True
            return result
        
        try:
            hyp = self._transcribe(str(audio_path)) if not self.dry_run else ""
            hyp_norm = self.norm.normalize(hyp)
            distortion = detect_distortion(audio_path)
            w = wer_percent(item.get("ref_norm", ""), hyp_norm)
            result.update({"hyp_norm": hyp_norm, "wer": float(w), "distorted": bool(distortion)})
        except Exception as e:
            LOG.warning("Failed WER for %s: %s", audio_fp, e)
        
        return result

    def filter_rows(self, rows: List[Dict], quality: Dict[str, Dict]) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        kept = []
        removed = defaultdict(list)

        for row in rows:
            audio_fp = row.get("audio_filepath", "")
            dur = float(row.get("duration", 0.0))
            text_raw = row.get("text", "")
            text = self.norm.normalize(text_raw)
            q = quality.get(audio_fp, {})

            # 1) WER > 50 or distorted audio
            if q.get("missing_audio") or q.get("distorted") or float(q.get("wer", 101.0)) > 50.0:
                removed["high_wer_or_distorted"].append({
                    "audio_filepath": audio_fp,
                    "duration": dur,
                    "text": text_raw,
                    "wer": q.get("wer"),
                    "distorted": q.get("distorted"),
                    "missing_audio": q.get("missing_audio"),
                })
                continue

            # 2) fewer than 3 words or non-speech labels
            wc = len(text.split()) if text else 0
            if text_raw.strip().lower() in NO_SPEECH or text in NO_SPEECH or wc < 3:
                removed["short_or_nonspeech"].append(
                    {"audio_filepath": audio_fp, "duration": dur, "text": text_raw, "norm_text": text, "word_count": wc}
                )
                continue

            # 3) longer than 30s
            if dur > 30.0:
                removed["longer_than_30s"].append({"audio_filepath": audio_fp, "duration": dur, "text": text_raw})
                continue

            new_row = dict(row)
            new_row["text"] = text
            kept.append(new_row)

        return kept, removed

    def concatenate_short_files(
        self,
        rows: List[Dict],
        split: str,
        concat_audio_dir: Path,
        short_max_seconds: float = 12.0,
    ) -> List[Dict]:
        by_session = defaultdict(list)
        for r in rows:
            by_session[session_key(r["audio_filepath"])].append(r)

        final_rows = []
        concat_count = 0

        for sess, sess_rows in by_session.items():
            sess_rows.sort(key=lambda x: utterance_idx(x["audio_filepath"]))
            chunk = []
            chunk_dur = 0.0

            def flush_chunk():
                nonlocal chunk, chunk_dur, concat_count
                if not chunk:
                    return
                if len(chunk) == 1:
                    final_rows.append(chunk[0])
                else:
                    concat_count += 1
                    sess_tag = safe_name(sess)
                    out_audio = concat_audio_dir / split / f"{sess_tag}_concat_{concat_count:06d}.flac"
                    audio_files = [c["audio_filepath"] for c in chunk]
                    concat_audio_ffmpeg(audio_files, out_audio)
                    final_rows.append(
                        {
                            "audio_filepath": str(out_audio),
                            "duration": round(sum(float(c["duration"]) for c in chunk), 6),
                            "text": self.norm.normalize(" ".join(c["text"] for c in chunk)),
                            "source_audio_filepaths": audio_files,
                        }
                    )
                chunk = []
                chunk_dur = 0.0

            for r in sess_rows:
                d = float(r.get("duration", 0.0))

                # only concatenate short files; keep longer files standalone
                if d > short_max_seconds:
                    flush_chunk()
                    final_rows.append(r)
                    continue

                if chunk_dur + d <= 30.0:
                    chunk.append(r)
                    chunk_dur += d
                else:
                    flush_chunk()
                    chunk = [r]
                    chunk_dur = d

            flush_chunk()

        LOG.info("Concatenated %d chunks for split=%s", concat_count, split)
        return final_rows


def summarize(split: str, original_rows: List[Dict], final_rows: List[Dict], removed: Dict[str, List[Dict]]):
    o_dur = sum(float(x.get("duration", 0.0)) for x in original_rows)
    f_dur = sum(float(x.get("duration", 0.0)) for x in final_rows)
    LOG.info(
        "%s | original=%d (%.2fh) final=%d (%.2fh) removed=%d",
        split,
        len(original_rows),
        o_dur / 3600,
        len(final_rows),
        f_dur / 3600,
        len(original_rows) - len(final_rows),
    )
    for k, v in removed.items():
        LOG.info("%s | removed[%s]=%d", split, k, len(v))


def main():
    parser = argparse.ArgumentParser(description="MyST preprocessing pipeline")
    # parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="For cache and reports")
    parser.add_argument("--model", type=str, default="large")
    parser.add_argument("--short-max-seconds", type=float, default=12.0)
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs to use for parallel transcription")
    parser.add_argument("--batch-size", type=int, default=64, help="Transcription batch size per Ray task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Initialize Ray for multi-GPU processing
    if not ray.is_initialized():
        ray.init(num_gpus=args.num_gpus, log_to_driver=False)
        LOG.info("Initialized Ray with %d GPUs", args.num_gpus)
    
    try:
        pre = MystPreprocessor(model_size=args.model, dry_run=args.dry_run)

        for split, in_path in [("val", args.val), ("test", args.test)]:
        # for split, in_path in [("train", args.train), ("val", args.val), ("test", args.test)]:
            LOG.info("Processing split=%s from %s", split, in_path)
            rows = pre.load_manifest(in_path)

            cache_path = args.work_dir / "wer_cache" / f"{split}.jsonl"
            quality = pre.compute_wer_and_quality(
                rows,
                cache_path,
                num_gpus=args.num_gpus,
                batch_size=args.batch_size,
            )

            kept, removed = pre.filter_rows(rows, quality)

            final_rows = pre.concatenate_short_files(
                kept,
                split=split,
                concat_audio_dir=args.output_dir / "concat_audio",
                short_max_seconds=args.short_max_seconds,
            )

            out_manifest = args.output_dir / f"{split}.json"
            pre.save_manifest(final_rows, out_manifest)

            removed_out = args.work_dir / "removed" / f"{split}_removed.json"
            removed_out.parent.mkdir(parents=True, exist_ok=True)
            with removed_out.open("w") as f:
                json.dump(removed, f, indent=2)

            summarize(split, rows, final_rows, removed)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
