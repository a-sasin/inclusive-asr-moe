"""
Extract routing distributions for English MoE models on YODAS2 audio.

This version is intentionally matched to the working LibriSpeech/MyST
English extraction script.

Output schema:
    utterance_id, age_group, dataset_source, language, layer_idx,
    router_entropy, top1_expert, top2_expert, duration_sec,
    expert_0_prob, expert_1_prob, expert_2_prob, expert_3_prob, model

Usage:
    CUDA_VISIBLE_DEVICES=3 python run_yodas.py

Smoke test:
    CUDA_VISIBLE_DEVICES=3 python run_yodas.py \
        --models child_moe_lb_off \
        --max-utterances 10
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import nemo.collections.asr as nemo_asr


MODEL_FILES = {
    "adult_moe": "en_adult_moe.nemo",
    "child_moe_lb_on": "en_child_moe_lb_on.nemo",
    "child_moe_lb_off": "en_child_moe_lb_off.nemo",
}

NUM_LAYERS = 17
NUM_EXPERTS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract English MoE routing on YODAS2 audio."
    )

    parser.add_argument(
        "--model-dir",
        default="/lp-dev/amelia/inclusive-asr-moe/final_weights",
        help="Directory containing .nemo model files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/lp-dev/amelia/inclusive-asr-moe2/analysis/routing/routing_outputs_en",
        help="Output directory for YODAS routing CSVs.",
    )
    parser.add_argument(
        "--yodas-manifest",
        default=(
            "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/"
            "ASR_updated/YODAS2/en/0_by_whisper/sharded_manifests_updated/"
            "manifest_53.json"
        ),
        help="YODAS manifest JSON file.",
    )
    parser.add_argument(
        "--yodas-tar",
        default="/data/granary-pz/yodas2/en/0_by_whisper/audio_53.tar",
        help="YODAS tarred audio file.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["adult_moe", "child_moe_lb_on", "child_moe_lb_off"],
        help="Model keys to run.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=1300,
        help="Maximum number of YODAS utterances. Use -1 for all.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda or cuda:0.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language label written to CSV.",
    )
    parser.add_argument(
        "--keep-temp-audio",
        action="store_true",
        help="Keep temporary extracted audio files.",
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help="Print router tensor shapes for the first few hook calls.",
    )

    return parser.parse_args()


def get_audio_path_candidates(audio_filepath):
    if not audio_filepath:
        return []

    raw = str(audio_filepath).strip().replace("\\", "/")
    candidates = []

    def add(x):
        if x and x not in candidates:
            candidates.append(x)

    add(raw)
    add(raw.lstrip("/"))
    add(os.path.basename(raw))

    if ".tar/" in raw:
        after = raw.split(".tar/", 1)[1]
        add(after)
        add(after.lstrip("/"))
        add(os.path.basename(after))

    if "::" in raw:
        after = raw.split("::", 1)[1]
        add(after)
        add(after.lstrip("/"))
        add(os.path.basename(after))

    if ".tar:" in raw:
        after = raw.split(".tar:", 1)[1]
        add(after)
        add(after.lstrip("/"))
        add(os.path.basename(after))

    return candidates


class TarAudioResolver:
    def __init__(self, tar_path, temp_dir):
        self.tar_path = tar_path
        self.temp_dir = Path(temp_dir)
        self.tar = None
        self.member_by_name = {}
        self.member_by_basename = {}

        if not os.path.exists(self.tar_path):
            raise FileNotFoundError(f"Tar file not found: {self.tar_path}")

    def open(self):
        if self.tar is None:
            print(f"Opening tar: {self.tar_path}")
            self.tar = tarfile.open(self.tar_path, "r")
            self._build_index()

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def _build_index(self):
        print("Indexing tar members...")
        count = 0

        for member in self.tar.getmembers():
            if not member.isfile():
                continue

            name = member.name.replace("\\", "/")
            basename = os.path.basename(name)

            self.member_by_name[name] = member
            self.member_by_name[name.lstrip("/")] = member

            if basename and basename not in self.member_by_basename:
                self.member_by_basename[basename] = member

            count += 1

        print(f"Indexed {count} file members.")

    def resolve(self, audio_filepath):
        if audio_filepath and os.path.exists(audio_filepath):
            return audio_filepath, "disk"

        self.open()

        member = None
        for cand in get_audio_path_candidates(audio_filepath):
            cand_norm = cand.replace("\\", "/").lstrip("/")

            if cand_norm in self.member_by_name:
                member = self.member_by_name[cand_norm]
                break

            base = os.path.basename(cand_norm)
            if base in self.member_by_basename:
                member = self.member_by_basename[base]
                break

        if member is None:
            return None, None

        base = os.path.basename(member.name)
        ext = Path(base).suffix or ".wav"
        digest = hashlib.md5(member.name.encode("utf-8")).hexdigest()[:12]
        out_path = self.temp_dir / f"yodas_{digest}{ext}"

        if not out_path.exists():
            extracted = self.tar.extractfile(member)
            if extracted is None:
                return None, None
            with open(out_path, "wb") as f:
                shutil.copyfileobj(extracted, f)

        return str(out_path), "tar"


def extract_routing_per_utterance(
    model,
    manifest_path,
    tar_path,
    age_group="adult",
    dataset_source="yodas",
    language="en",
    max_utterances=1300,
    device="cuda",
    temp_dir=None,
    debug_shapes=False,
):
    """
    Extracts per-utterance expert routing distributions from OmniRouter.

    This is matched to the working English LibriSpeech/MyST extraction:
        logits -> softmax -> squeeze batch -> mean over frames
    """
    model.eval()
    model = model.to(device)

    with open(manifest_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if max_utterances is not None and max_utterances > 0:
        lines = lines[:max_utterances]

    records = []
    routing_buffer = {"probs_list": []}
    debug_counter = {"n": 0}

    def router_hook(module, inputs, output):
        try:
            logits = output[2]
        except Exception as exc:
            print(f"Router hook failed to unpack output: {exc}")
            return

        probs = torch.softmax(logits.float(), dim=-1)

        if debug_shapes and debug_counter["n"] < 5:
            print(
                "DEBUG router shapes:",
                "logits=", tuple(logits.shape),
                "probs=", tuple(probs.shape),
            )
            debug_counter["n"] += 1

        # Match the working English extraction:
        # old code did probs.squeeze(0).mean(dim=0).
        # This safe version handles either (1, T, E) or (T, E).
        if probs.ndim == 3:
            if probs.shape[0] != 1:
                raise RuntimeError(
                    f"Expected batch size 1 in router probs, got shape {tuple(probs.shape)}"
                )
            probs = probs.squeeze(0)

        if probs.ndim != 2:
            raise RuntimeError(
                f"Expected router probs shape (T, E), got {tuple(probs.shape)}"
            )

        probs = probs.mean(dim=0)

        if probs.ndim != 1:
            raise RuntimeError(
                f"Expected mean router probs shape (E,), got {tuple(probs.shape)}"
            )

        if probs.shape[0] != NUM_EXPERTS:
            raise RuntimeError(
                f"Expected {NUM_EXPERTS} experts, got {probs.shape[0]}"
            )

        routing_buffer["probs_list"].append(probs.detach().cpu().numpy())

    hook_handle = model.encoder.global_router.register_forward_hook(router_hook)

    resolver = TarAudioResolver(tar_path=tar_path, temp_dir=temp_dir)

    stats = {
        "from_tar": 0,
        "from_disk": 0,
        "skipped_audio": 0,
        "failed_transcribe": 0,
        "incomplete_routing": 0,
    }

    print(f"Processing {len(lines)} utterances [{age_group}/{dataset_source}]...")
    print(f"Manifest: {manifest_path}")
    print(f"Tar:      {tar_path}")

    try:
        for line in tqdm(lines):
            data = json.loads(line)

            manifest_audio_path = data.get("audio_filepath")
            resolved_audio_path, source = resolver.resolve(manifest_audio_path)

            if resolved_audio_path is None or not os.path.exists(resolved_audio_path):
                stats["skipped_audio"] += 1
                if stats["skipped_audio"] <= 10:
                    print(f"Skipping unresolved audio: {manifest_audio_path}")
                continue

            if source == "tar":
                stats["from_tar"] += 1
            elif source == "disk":
                stats["from_disk"] += 1

            duration_sec = data.get("duration", data.get("duration_sec"))
            try:
                duration_sec = float(duration_sec)
            except (TypeError, ValueError):
                duration_sec = np.nan

            routing_buffer["probs_list"] = []

            try:
                with torch.no_grad():
                    model.transcribe(
                        [resolved_audio_path],
                        batch_size=1,
                        verbose=False,
                    )
            except Exception as exc:
                stats["failed_transcribe"] += 1
                if stats["failed_transcribe"] <= 10:
                    print(f"Failed transcribe: {manifest_audio_path}")
                    print(f"Resolved path: {resolved_audio_path}")
                    print(f"Error: {exc}")
                continue

            if len(routing_buffer["probs_list"]) != NUM_LAYERS:
                stats["incomplete_routing"] += 1
                if stats["incomplete_routing"] <= 10:
                    print(
                        "Warning: incomplete routing captured "
                        f"({len(routing_buffer['probs_list'])}/{NUM_LAYERS}) for "
                        f"{manifest_audio_path}"
                    )
                continue

            probs_all_layers = np.stack(routing_buffer["probs_list"])

            for layer_idx in range(probs_all_layers.shape[0]):
                probs = probs_all_layers[layer_idx]

                # Normalize defensively. Should already sum to 1.
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum

                entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

                row = {
                    "utterance_id": manifest_audio_path,
                    "age_group": age_group,
                    "dataset_source": dataset_source,
                    "language": language,
                    "layer_idx": layer_idx,
                    "router_entropy": entropy,
                    "top1_expert": int(np.argmax(probs)),
                    "top2_expert": int(np.argsort(probs)[-2]),
                    "duration_sec": duration_sec,
                }

                for e in range(NUM_EXPERTS):
                    row[f"expert_{e}_prob"] = float(probs[e])

                records.append(row)

    finally:
        hook_handle.remove()
        resolver.close()

    df = pd.DataFrame(records)

    print(f"\nFinished {age_group}/{dataset_source}")
    print(f"  Routing rows collected: {len(df):,}")
    print(f"  Utterances represented: {df['utterance_id'].nunique() if not df.empty else 0:,}")
    print(f"  Resolved from tar:      {stats['from_tar']:,}")
    print(f"  Resolved from disk:     {stats['from_disk']:,}")
    print(f"  Skipped missing audio:  {stats['skipped_audio']:,}")
    print(f"  Failed transcribe:      {stats['failed_transcribe']:,}")
    print(f"  Incomplete routing:     {stats['incomplete_routing']:,}")

    if df.empty:
        raise RuntimeError(
            f"No routing records extracted for {age_group}/{dataset_source}. "
            "Check the manifest/tar path and audio_filepath values."
        )

    return df


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.max_utterances is not None and args.max_utterances < 0:
        max_utterances = None
    else:
        max_utterances = args.max_utterances

    model_paths = {}
    for model_name in args.models:
        if model_name not in MODEL_FILES:
            print(f"Skipping unknown model key: {model_name}")
            continue

        model_path = os.path.join(args.model_dir, MODEL_FILES[model_name])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model_paths[model_name] = model_path

    temp_root = tempfile.mkdtemp(prefix="yodas_routing_audio_")
    print(f"Temporary audio directory: {temp_root}")

    try:
        for model_name, model_path in model_paths.items():
            print("\n" + "=" * 80)
            print(f"Model: {model_name}")
            print(f"Path:  {model_path}")
            print("=" * 80)

            out_path = os.path.join(
                args.cache_dir,
                f"{model_name}_routing_yodas.csv",
            )

            print(f"Output CSV: {out_path}")

            model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                model_path,
                map_location=args.device,
            )

            df_yodas = extract_routing_per_utterance(
                model=model,
                manifest_path=args.yodas_manifest,
                tar_path=args.yodas_tar,
                age_group="adult",
                dataset_source="yodas",
                language=args.language,
                max_utterances=max_utterances,
                device=args.device,
                temp_dir=temp_root,
                debug_shapes=args.debug_shapes,
            )

            df_yodas["model"] = model_name

            prob_cols = [f"expert_{i}_prob" for i in range(NUM_EXPERTS)]
            prob_sums = df_yodas[prob_cols].sum(axis=1)

            print("\nProbability sanity check")
            print(prob_sums.describe())
            print("Min expert prob:", df_yodas[prob_cols].min().min())
            print("Max expert prob:", df_yodas[prob_cols].max().max())
            print(
                "Rows with prob sum not close to 1:",
                int(((prob_sums < 0.999) | (prob_sums > 1.001)).sum()),
            )

            df_yodas.to_csv(out_path, index=False)

            print(f"Saved: {out_path}")
            print(f"Rows:  {len(df_yodas):,}")

            del model
            torch.cuda.empty_cache()

    finally:
        if args.keep_temp_audio:
            print(f"Keeping temporary audio directory: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f"Removed temporary audio directory: {temp_root}")


if __name__ == "__main__":
    main()