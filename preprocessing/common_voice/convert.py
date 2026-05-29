#!/usr/bin/env python3
"""Convert CommonVoice audio clips and TSV metadata to NeMo JSONL manifest format.

Reads a CommonVoice TSV split (e.g. train.tsv, dev.tsv, test.tsv), resamples
MP3 audio files to 16 kHz mono WAV, and writes a NeMo-compatible JSONL manifest
with fields: audio_filepath, duration, text.

Usage:
    python convert.py \
        --tsv_file /data/cv/nl/train.tsv \
        --audio_dir /data/cv/nl/clips \
        --output_dir /data/cv/nemo/nl \
        --split train

See prepare.py for the full end-to-end pipeline including splitting and filtering.
"""

import argparse
import json
import logging
import multiprocessing
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import sox
from tqdm.contrib.concurrent import process_map


RAW_ROOT = Path("/data/cv/raw/cv-corpus-25.0-2026-03-09")
OUTPUT_DIR = Path("/data/cv/nemo")
LANGUAGES = ["nl", "de", "pl"]

# Validated hours available per language (for logging only)
LANGUAGE_META = {
	"nl": {"name": "Dutch", "validated_hours": 126, "speakers": 1874},
	"de": {"name": "German", "validated_hours": 1389, "speakers": 20466},
	"pl": {"name": "Polish", "validated_hours": 177, "speakers": 3465},
}


def _base_clean(text: str) -> str:
	text = text.lower()
	text = re.sub(r"[–—−‐‑‒]", " ", text)
	text = re.sub(r"[''`ʽ´ʻ]", "'", text)
	text = re.sub(r'[""„‟″«»]', "", text)
	text = re.sub(r"[\u00AD\u200B-\u200D\uFEFF]", "", text)
	return re.sub(r" +", " ", text).strip()


def normalize_nl(text: str) -> str:
	text = _base_clean(text)
	text = re.sub(r"[^a-zàâäéèêëîïôöùûüÿçæœ' ]", " ", text)
	text = re.sub(
		r"(?<![a-zàâäéèêëîïôöùûüÿçæœ])'|'(?![a-zàâäéèêëîïôöùûüÿçæœ])",
		" ",
		text,
	)
	return re.sub(r" +", " ", text).strip()


def normalize_de(text: str) -> str:
	text = _base_clean(text)
	return re.sub(r" +", " ", re.sub(r"[^a-zäöüß ]", " ", text)).strip()


def normalize_pl(text: str) -> str:
	text = _base_clean(text)
	return re.sub(r" +", " ", re.sub(r"[^a-ząćęłńóśźż ]", " ", text)).strip()


NORMALIZERS = {
	"nl": normalize_nl,
	"de": normalize_de,
	"pl": normalize_pl,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Preprocess Common Voice 25.0 (nl/de/pl) into NeMo JSON manifests"
	)
	parser.add_argument(
		"--languages",
		nargs="+",
		default=LANGUAGES,
		choices=LANGUAGES,
		help="Languages to process",
	)
	parser.add_argument(
		"--num_workers",
		type=int,
		default=max(1, (os.cpu_count() or 2) - 2),
		help="Number of worker processes for resampling",
	)
	parser.add_argument("--val_ratio", type=float, default=0.05)
	parser.add_argument("--test_ratio", type=float, default=0.10)
	parser.add_argument("--val_cap_hours", type=float, default=10.0)
	parser.add_argument("--test_cap_hours", type=float, default=10.0)
	parser.add_argument("--dry_run", action="store_true")
	parser.add_argument("--report_filter_stages", action="store_true")
	parser.add_argument("--force_resample", action="store_true")
	return parser.parse_args()


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


def hours_from_seconds(seconds: float) -> float:
	return seconds / 3600.0


def _safe_float(v, default=0.0):
	try:
		if pd.isna(v):
			return default
		return float(v)
	except (ValueError, TypeError):
		return default


def _is_nonempty(value) -> bool:
	if value is None:
		return False
	if isinstance(value, float) and pd.isna(value):
		return False
	return str(value).strip() != ""


def log_filter_stage(
	lang: str,
	stage_name: str,
	before_len: int,
	after_df: pd.DataFrame,
	report_filter_stages: bool,
) -> None:
	if not report_filter_stages:
		return
	dropped = before_len - len(after_df)
	hours = hours_from_seconds(after_df["duration_sec"].sum()) if len(after_df) else 0.0
	logging.info(
		"[%s] Stage2 filter %-28s dropped=%7d | remaining=%8d | hours=%8.2f",
		lang,
		stage_name,
		dropped,
		len(after_df),
		hours,
	)


def load_clip_durations(lang_dir: Path, lang: str) -> dict:
	durations_path = lang_dir / "clip_durations.tsv"
	if not durations_path.exists():
		raise FileNotFoundError(f"Missing durations file: {durations_path}")

	logging.info("[%s] Stage1: Loading durations from %s", lang, durations_path)
	dur_df = pd.read_csv(durations_path, sep="\t", dtype={"clip": "string"})

	if "clip" not in dur_df.columns or "duration[ms]" not in dur_df.columns:
		raise ValueError(
			f"Expected columns 'clip' and 'duration[ms]' in {durations_path}, found: {list(dur_df.columns)}"
		)

	dur_df = dur_df.dropna(subset=["clip", "duration[ms]"]).copy()
	dur_df["duration_sec"] = pd.to_numeric(dur_df["duration[ms]"], errors="coerce") / 1000.0
	dur_df = dur_df.dropna(subset=["duration_sec"])  # drop invalid duration rows
	dur_df = dur_df[dur_df["duration_sec"] > 0]

	duration_map = {}
	for _, row in dur_df.iterrows():
		clip = str(row["clip"]).strip()
		stem = Path(clip).stem
		duration_map[stem] = float(row["duration_sec"])

	logging.info(
		"[%s] Stage1: Loaded %d clip durations (using duration[ms] from clip_durations.tsv)",
		lang,
		len(duration_map),
	)
	return duration_map


def load_and_filter_validated(
	lang: str,
	lang_dir: Path,
	duration_map: dict,
	report_filter_stages: bool,
) -> pd.DataFrame:
	validated_path = lang_dir / "validated.tsv"
	clips_dir = lang_dir / "clips"
	if not validated_path.exists():
		raise FileNotFoundError(f"Missing validated file: {validated_path}")

	logging.info("[%s] Stage2: Loading and filtering %s", lang, validated_path)
	df = pd.read_csv(validated_path, sep="\t", dtype="string", keep_default_na=False)
	if "path" not in df.columns or "sentence" not in df.columns or "client_id" not in df.columns:
		raise ValueError(
			f"{validated_path} must include columns: path, sentence, client_id. Found: {list(df.columns)}"
		)

	df = df.copy()
	df["path"] = df["path"].astype("string").str.strip()
	df["sentence"] = df["sentence"].astype("string").str.strip()
	df["client_id"] = df["client_id"].astype("string").str.strip()

	before = len(df)
	df = df[(df["path"] != "") & (df["sentence"] != "") & (df["client_id"] != "")].copy()
	if "duration_sec" not in df.columns:
		df["duration_sec"] = 0.0
	log_filter_stage(lang, "drop_empty_path_sentence_client", before, df, report_filter_stages)

	df["clip_stem"] = df["path"].map(lambda p: Path(str(p)).stem)
	df["duration_sec"] = df["clip_stem"].map(duration_map)
	before = len(df)
	df = df.dropna(subset=["duration_sec"]).copy()
	df = df[df["duration_sec"] > 0].copy()
	log_filter_stage(lang, "drop_missing_or_zero_duration", before, df, report_filter_stages)

	df["src_audio"] = df["path"].map(lambda p: clips_dir / str(p))
	before = len(df)
	df = df[df["src_audio"].map(Path.exists)].copy()
	log_filter_stage(lang, "drop_missing_mp3_files", before, df, report_filter_stages)

	# Derive a stable quality score from vote columns present in CV validated.tsv.
	df["up_votes_num"] = pd.to_numeric(df.get("up_votes", 0), errors="coerce").fillna(0)
	df["down_votes_num"] = pd.to_numeric(df.get("down_votes", 0), errors="coerce").fillna(0)
	df["quality_score_num"] = df["up_votes_num"] / (
		df["up_votes_num"] + df["down_votes_num"] + 1
	)

	total_hours = hours_from_seconds(df["duration_sec"].sum())
	logging.info(
		"[%s] Stage2: Filtered validated set to %d clips (%.2f hours, %d speakers)",
		lang,
		len(df),
		total_hours,
		df["client_id"].nunique(),
	)
	return df


def split_speakers(
	df: pd.DataFrame,
	lang: str,
	val_ratio: float,
	test_ratio: float,
) -> dict:
	logging.info("[%s] Stage3: Speaker-independent split", lang)
	speakers = df["client_id"].dropna().unique().tolist()
	random.seed(42)
	random.shuffle(speakers)

	n = len(speakers)
	n_test = int(n * test_ratio)
	n_val = int(n * val_ratio)

	test_speakers = set(speakers[:n_test])
	val_speakers = set(speakers[n_test : n_test + n_val])
	train_speakers = set(speakers[n_test + n_val :])

	train_df = df[df["client_id"].isin(train_speakers)].copy()
	val_df = df[df["client_id"].isin(val_speakers)].copy()
	test_df = df[df["client_id"].isin(test_speakers)].copy()

	overlap_tv = train_speakers.intersection(val_speakers)
	overlap_tt = train_speakers.intersection(test_speakers)
	overlap_vt = val_speakers.intersection(test_speakers)
	if overlap_tv or overlap_tt or overlap_vt:
		raise RuntimeError(
			f"[{lang}] Speaker overlap detected during split; train/val/test must be disjoint"
		)

	for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
		logging.info(
			"[%s] Stage3: %-5s speakers=%6d clips=%8d hours=%8.2f",
			lang,
			split_name,
			split_df["client_id"].nunique(),
			len(split_df),
			hours_from_seconds(split_df["duration_sec"].sum()),
		)

	return {"train": train_df, "val": val_df, "test": test_df}


def cap_split_by_hours(
	df: pd.DataFrame,
	split_name: str,
	cap_hours: float,
	lang: str,
	max_clips_per_speaker: int = 50,
) -> pd.DataFrame:
	_ = cap_hours
	_ = max_clips_per_speaker
	logging.info(
		"[%s] Stage4: Capping disabled; keeping all %s clips (%d clips, %.2f h, %d speakers)",
		lang,
		split_name,
		len(df),
		hours_from_seconds(df["duration_sec"].sum()),
		int(df["client_id"].nunique()),
	)
	return df.copy()


def normalize_split_text(df: pd.DataFrame, lang: str, split_name: str) -> pd.DataFrame:
	logging.info("[%s] Stage5: Text normalization for %s", lang, split_name)
	normalize = NORMALIZERS[lang]
	out_df = df.copy()
	out_df["text"] = out_df["sentence"].map(lambda s: normalize(str(s) if s is not None else ""))
	before = len(out_df)
	out_df = out_df[out_df["text"].astype("string").str.strip() != ""].copy()
	dropped = before - len(out_df)
	logging.info(
		"[%s] Stage5: %s dropped %d rows with empty normalized text; remaining %d",
		lang,
		split_name,
		dropped,
		len(out_df),
	)
	return out_df


def _resample_one(task: tuple) -> tuple:
	src_path, wav_path, force_resample = task
	src = Path(src_path)
	dst = Path(wav_path)

	if dst.exists() and not force_resample:
		try:
			dur = float(sox.file_info.duration(str(dst)))
			return str(dst), dur, "exists"
		except Exception as exc:
			return str(dst), 0.0, f"duration_failed_existing:{exc}"

	dst.parent.mkdir(parents=True, exist_ok=True)

	if dst.exists() and force_resample:
		try:
			dst.unlink()
		except Exception:
			pass

	try:
		tfm = sox.Transformer()
		tfm.convert(samplerate=16000, n_channels=1, bitdepth=16)
		tfm.build(str(src), str(dst))
		dur = float(sox.file_info.duration(str(dst)))
		return str(dst), dur, "sox"
	except Exception:
		ffmpeg_cmd = [
			"ffmpeg",
			"-y",
			"-i",
			str(src),
			"-ac",
			"1",
			"-ar",
			"16000",
			str(dst),
		]
		try:
			proc = subprocess.run(
				ffmpeg_cmd,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True,
				check=False,
			)
			if proc.returncode != 0:
				return str(dst), 0.0, f"ffmpeg_failed:{proc.stderr[:240]}"
			dur = float(sox.file_info.duration(str(dst)))
			return str(dst), dur, "ffmpeg"
		except Exception as exc:
			return str(dst), 0.0, f"resample_failed:{exc}"


def stage6_resample_and_duration(
	split_dfs: dict,
	lang: str,
	num_workers: int,
	force_resample: bool,
) -> dict:
	logging.info(
		"[%s] Stage6: Resampling to 16kHz mono WAV with %d workers (force=%s)",
		lang,
		num_workers,
		force_resample,
	)

	all_rows = []
	for split_name, df in split_dfs.items():
		subset = df.copy()
		subset["split"] = split_name
		all_rows.append(subset)
	all_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
	if all_df.empty:
		return split_dfs

	all_df["wav_path"] = all_df["clip_stem"].map(
		lambda stem: OUTPUT_DIR / lang / "wav" / f"{stem}.wav"
	)

	tasks = [
		(str(src), str(dst), force_resample)
		for src, dst in zip(all_df["src_audio"].tolist(), all_df["wav_path"].tolist())
	]

	results = process_map(
		_resample_one,
		tasks,
		max_workers=max(1, int(num_workers)),
		chunksize=16,
		desc=f"[{lang}] Stage6 resample",
	)

	duration_by_wav = {}
	method_counts = defaultdict(int)
	failed = 0
	for wav_path, duration, method in results:
		if duration > 0:
			duration_by_wav[wav_path] = duration
			method_counts[method] += 1
		else:
			failed += 1
			method_counts[method] += 1

	out = {}
	for split_name in ("train", "val", "test"):
		split_df = split_dfs[split_name].copy()
		split_df["wav_path"] = split_df["clip_stem"].map(
			lambda stem: OUTPUT_DIR / lang / "wav" / f"{stem}.wav"
		)
		split_df["duration"] = split_df["wav_path"].map(
			lambda p: duration_by_wav.get(str(p), 0.0)
		)
		before = len(split_df)
		split_df = split_df[split_df["duration"] > 0].copy()
		dropped = before - len(split_df)
		if dropped:
			logging.warning(
				"[%s] Stage6: %s dropped %d clips due to resample/duration failures",
				lang,
				split_name,
				dropped,
			)
		out[split_name] = split_df

	logging.info(
		"[%s] Stage6: Completed %d clips, failed=%d, methods=%s",
		lang,
		len(results),
		failed,
		dict(method_counts),
	)
	return out


def build_manifest_record(row: pd.Series, lang: str) -> dict:
	rec = {
		"audio_filepath": str(row["wav_path"]),
		"text": str(row["text"]),
		"duration": round(_safe_float(row["duration"]), 3),
		"speaker": str(row["client_id"]),
		"language": lang,
		"age_group": "adult",
	}

	optional_meta_cols = [
		"age",
		"gender",
		"accents",
		"variant",
		"locale",
		"segment",
		"up_votes",
		"down_votes",
	]
	for col in optional_meta_cols:
		if col in row and _is_nonempty(row[col]):
			rec[col] = row[col]

	return rec


def write_manifest(df: pd.DataFrame, manifest_path: Path, lang: str, split_name: str) -> dict:
	manifest_path.parent.mkdir(parents=True, exist_ok=True)
	clips = 0
	total_sec = 0.0
	speakers = set()
	with manifest_path.open("w", encoding="utf-8") as f:
		for _, row in df.iterrows():
			rec = build_manifest_record(row, lang)
			f.write(json.dumps(rec, ensure_ascii=False) + "\n")
			clips += 1
			total_sec += _safe_float(rec.get("duration", 0.0))
			if _is_nonempty(rec.get("speaker")):
				speakers.add(str(rec["speaker"]))

	stats = {
		"clips": clips,
		"hours": hours_from_seconds(total_sec),
		"speakers": len(speakers),
	}
	logging.info(
		"[%s] Stage7: Wrote %s with %d clips, %.2f hours, %d speakers",
		lang,
		manifest_path,
		stats["clips"],
		stats["hours"],
		stats["speakers"],
	)
	return stats


def read_manifest_speakers(manifest_path: Path) -> set:
	speakers = set()
	if not manifest_path.exists():
		return speakers
	with manifest_path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				rec = json.loads(line)
			except json.JSONDecodeError:
				continue
			spk = rec.get("speaker")
			if _is_nonempty(spk):
				speakers.add(str(spk))
	return speakers


def verify_speaker_independence(languages: list) -> None:
	logging.info("Stage9: Speaker independence verification")
	for lang in languages:
		lang_out = OUTPUT_DIR / lang
		train_s = read_manifest_speakers(lang_out / "train.json")
		val_s = read_manifest_speakers(lang_out / "val.json")
		test_s = read_manifest_speakers(lang_out / "test.json")
		o_tv = train_s.intersection(val_s)
		o_tt = train_s.intersection(test_s)
		o_vt = val_s.intersection(test_s)
		overlap_count = len(o_tv) + len(o_tt) + len(o_vt)
		if overlap_count == 0:
			logging.info("[%s] Stage9: OK - no speaker overlap across train/val/test", lang)
		else:
			logging.warning(
				"[%s] Stage9: WARNING - speaker overlap detected (count=%d)",
				lang,
				overlap_count,
			)


def print_final_summary(summary: dict, languages: list) -> None:
	logging.info("═" * 70)
	logging.info("FINAL SUMMARY - CV 25.0 NeMo Manifests")
	logging.info("═" * 70)
	logging.info("%-6s %-10s %10s %9s %10s", "Lang", "Split", "Clips", "Hours", "Speakers")
	logging.info("─" * 70)

	train_total_clips = 0
	train_total_hours = 0.0

	for lang in languages:
		for split in ("train", "val", "test"):
			s = summary.get(lang, {}).get(split, {"clips": 0, "hours": 0.0, "speakers": 0})
			logging.info(
				"%-6s %-10s %10d %9.1f %10d",
				lang,
				split,
				int(s["clips"]),
				float(s["hours"]),
				int(s["speakers"]),
			)
			if split == "train":
				train_total_clips += int(s["clips"])
				train_total_hours += float(s["hours"])

	logging.info("─" * 70)
	logging.info("%-17s %10d %9.1f", "Train total", train_total_clips, train_total_hours)
	logging.info("═" * 70)

	logging.info("Next step - NeMo training config:")
	logging.info("")
	logging.info("  train_ds:")
	logging.info("    manifest_filepath:")
	logging.info("      - /data/cv/nemo/nl/train.json")
	logging.info("      - /data/cv/nemo/de/train.json")
	logging.info("      - /data/cv/nemo/pl/train.json")
	logging.info("    concat_sampling_technique: temperature")
	logging.info("    concat_sampling_temperature: 2.0")
	logging.info("")
	logging.info("  validation_ds:")
	logging.info("    manifest_filepath:")
	logging.info("      - /data/cv/nemo/nl/val.json")
	logging.info("      - /data/cv/nemo/de/val.json")
	logging.info("      - /data/cv/nemo/pl/val.json")
	logging.info("")
	logging.info("  test_ds:")
	logging.info("    manifest_filepath:")
	logging.info("      - /data/cv/nemo/nl/test.json")
	logging.info("      - /data/cv/nemo/de/test.json")
	logging.info("      - /data/cv/nemo/pl/test.json")


def main() -> int:
	setup_logging()
	args = parse_args()

	languages = [lang for lang in args.languages if lang in LANGUAGES]
	if not languages:
		logging.error("No valid languages selected. Choose from: %s", LANGUAGES)
		return 1

	logging.info("Starting CV 25.0 -> NeMo preprocessing")
	logging.info(
		"Stage4 configuration: all caps disabled; --val_cap_hours and --test_cap_hours are ignored"
	)
	for lang in languages:
		meta = LANGUAGE_META.get(lang, {})
		logging.info(
			"[%s] Meta: name=%s validated_hours=%s speakers=%s",
			lang,
			meta.get("name", "unknown"),
			meta.get("validated_hours", "n/a"),
			meta.get("speakers", "n/a"),
		)

	summary = defaultdict(dict)
	train_manifest_paths = []

	for lang in languages:
		lang_dir = RAW_ROOT / lang
		if not lang_dir.exists():
			logging.error("[%s] Missing language directory: %s", lang, lang_dir)
			return 1

		out_lang_dir = OUTPUT_DIR / lang
		wav_dir = out_lang_dir / "wav"
		if not args.dry_run:
			wav_dir.mkdir(parents=True, exist_ok=True)

		duration_map = load_clip_durations(lang_dir, lang)
		validated_df = load_and_filter_validated(
			lang=lang,
			lang_dir=lang_dir,
			duration_map=duration_map,
			report_filter_stages=args.report_filter_stages,
		)

		split_dfs = split_speakers(
			df=validated_df,
			lang=lang,
			val_ratio=args.val_ratio,
			test_ratio=args.test_ratio,
		)

		split_dfs["val"] = cap_split_by_hours(split_dfs["val"], "val", args.val_cap_hours, lang)
		split_dfs["test"] = cap_split_by_hours(
			split_dfs["test"], "test", args.test_cap_hours, lang
		)

		split_dfs["train"] = normalize_split_text(split_dfs["train"], lang, "train")
		split_dfs["val"] = normalize_split_text(split_dfs["val"], lang, "val")
		split_dfs["test"] = normalize_split_text(split_dfs["test"], lang, "test")

		if args.dry_run:
			logging.info("[%s] Dry run active: skipping Stage6-Stage8 file writes", lang)
			for split_name in ("train", "val", "test"):
				s_df = split_dfs[split_name]
				summary[lang][split_name] = {
					"clips": len(s_df),
					"hours": hours_from_seconds(s_df["duration_sec"].sum()),
					"speakers": int(s_df["client_id"].nunique()),
				}
			continue

		split_dfs = stage6_resample_and_duration(
			split_dfs=split_dfs,
			lang=lang,
			num_workers=args.num_workers,
			force_resample=args.force_resample,
		)

		for split_name in ("train", "val", "test"):
			manifest_path = out_lang_dir / f"{split_name}.json"
			stats = write_manifest(split_dfs[split_name], manifest_path, lang, split_name)
			summary[lang][split_name] = stats
			if split_name == "train":
				train_manifest_paths.append(manifest_path)

	if not args.dry_run:
		merged_path = OUTPUT_DIR / "train_merged.json"
		logging.info("Stage8: Writing merged train manifest to %s", merged_path)
		merged_path.parent.mkdir(parents=True, exist_ok=True)
		with merged_path.open("w", encoding="utf-8") as out_f:
			for train_manifest in train_manifest_paths:
				if not train_manifest.exists():
					continue
				with train_manifest.open("r", encoding="utf-8") as in_f:
					for line in in_f:
						out_f.write(line)

		verify_speaker_independence(languages)
	else:
		logging.info("Dry run active: Stage8 and Stage9 skipped")

	print_final_summary(summary, languages)
	return 0


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except KeyboardInterrupt:
		logging.error("Interrupted by user")
		raise SystemExit(130)
