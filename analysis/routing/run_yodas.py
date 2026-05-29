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


DEFAULT_YODAS_MANIFEST = (
    "/data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/"
    "ASR_updated/YODAS2/en/0_by_whisper/sharded_manifests_updated/"
    "manifest_53.json"
)

DEFAULT_YODAS_TAR = (
    "/data/granary-pz/yodas2/en/0_by_whisper/audio__OP_0..529_CL_.tar"
)

DEFAULT_OUTPUT_DIR = "/lp-dev/amelia/inclusive-asr-moe/analysis/routing/outputs/yodas"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract routing distributions for English MoE models on tarred YODAS."
    )

    parser.add_argument(
        "--model-dir",
        default="/lp-dev/amelia/inclusive-asr-moe/final_weights",
        help="Directory containing .nemo model files.",
    )
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for routing CSVs.",
    )
    parser.add_argument(
        "--yodas-manifest",
        default=DEFAULT_YODAS_MANIFEST,
        help="YODAS sharded manifest JSON file.",
    )
    parser.add_argument(
        "--yodas-tar",
        default=DEFAULT_YODAS_TAR,
        help="YODAS tarred audio file.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["adult_moe", "child_moe_lb_on", "child_moe_lb_off"],
        help="Model keys to run: adult_moe child_moe_lb_on child_moe_lb_off",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=300,
        help="Maximum number of YODAS utterances to process. Use -1 for all.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda or cuda:0.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language label written to the CSV.",
    )
    parser.add_argument(
        "--keep-temp-audio",
        action="store_true",
        help="Keep temporary extracted audio files after the script finishes.",
    )

    return parser.parse_args()


def build_model_paths(model_dir, model_names):
    paths = {}

    for model_name in model_names:
        if model_name not in MODEL_FILES:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Known names: {list(MODEL_FILES.keys())}"
            )

        model_path = os.path.join(model_dir, MODEL_FILES[model_name])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        paths[model_name] = model_path

    return paths


def load_manifest_lines(manifest_path, max_utterances=None):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_utterances is not None and max_utterances > 0:
        lines = lines[:max_utterances]

    return lines


def get_audio_path_candidates(audio_filepath):
    """
    Produce possible tar member names from a NeMo-style manifest audio_filepath.

    Different tarred manifests may store audio_filepath as:
    - the exact tar member name
    - a relative path
    - a path containing the tar name followed by the member path
    - only a basename
    """
    if not audio_filepath:
        return []

    raw = str(audio_filepath).strip()
    raw = raw.replace("\\", "/")

    candidates = []

    def add(x):
        if x and x not in candidates:
            candidates.append(x)

    add(raw)
    add(raw.lstrip("/"))
    add(os.path.basename(raw))

    # Handle things like "...tar/some/path/audio.wav"
    if ".tar/" in raw:
        after_tar = raw.split(".tar/", 1)[1]
        add(after_tar)
        add(after_tar.lstrip("/"))
        add(os.path.basename(after_tar))

    # Handle things like "tarfile.tar::member.wav"
    if "::" in raw:
        after_sep = raw.split("::", 1)[1]
        add(after_sep)
        add(after_sep.lstrip("/"))
        add(os.path.basename(after_sep))

    # Handle things like "tarfile.tar:member.wav"
    # Avoid breaking absolute paths such as /data/...
    if ".tar:" in raw:
        after_sep = raw.split(".tar:", 1)[1]
        add(after_sep)
        add(after_sep.lstrip("/"))
        add(os.path.basename(after_sep))

    return candidates


class TarAudioResolver:
    """
    Resolves manifest audio_filepath entries to actual temporary files.

    If audio_filepath already exists on disk, it returns it unchanged.
    Otherwise, it tries to find the corresponding member inside the YODAS tar.
    """

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
            print(f"Opening tar file: {self.tar_path}")
            self.tar = tarfile.open(self.tar_path, "r")
            self._build_index()

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def _build_index(self):
        """
        Build lookup tables for exact member names and basenames.

        This can take a little time for large tar files, but it prevents
        repeatedly scanning the tar for every utterance.
        """
        print("Indexing tar members...")
        count = 0

        for member in self.tar.getmembers():
            if not member.isfile():
                continue

            name = member.name.replace("\\", "/")
            basename = os.path.basename(name)

            self.member_by_name[name] = member
            self.member_by_name[name.lstrip("/")] = member

            # Basenames should usually be unique enough for these shards.
            # If duplicate basenames exist, we keep the first one.
            if basename and basename not in self.member_by_basename:
                self.member_by_basename[basename] = member

            count += 1

        print(f"Indexed {count} file members from tar.")

    def resolve(self, audio_filepath):
        """
        Return a real local path that model.transcribe can read.

        Returns:
            resolved_path, source
            source is either "disk" or "tar"
        """
        if audio_filepath and os.path.exists(audio_filepath):
            return audio_filepath, "disk"

        self.open()

        candidates = get_audio_path_candidates(audio_filepath)

        member = None

        for cand in candidates:
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

        member_name = member.name
        base = os.path.basename(member_name)
        ext = Path(base).suffix

        if not ext:
            ext = ".wav"

        digest = hashlib.md5(member_name.encode("utf-8")).hexdigest()[:12]
        out_path = self.temp_dir / f"yodas_{digest}{ext}"

        if not out_path.exists():
            extracted = self.tar.extractfile(member)
            if extracted is None:
                return None, None

            with open(out_path, "wb") as out_f:
                shutil.copyfileobj(extracted, out_f)

        return str(out_path), "tar"


def extract_routing_per_utterance(
    model,
    manifest_path,
    tar_path,
    age_group,
    dataset_source,
    language="en",
    max_utterances=300,
    device="cuda",
    temp_dir=None,
):
    """
    Extracts per-utterance routing distributions from OmniRouter.

    Output:
    - one row per utterance per MoE layer
    - soft expert probabilities
    - hard top-k expert selection probabilities
    - top-1 expert counts
    - entropy
    """
    model.eval()
    model = model.to(device)

    lines = load_manifest_lines(manifest_path, max_utterances=max_utterances)

    records = []
    skipped_missing_audio = 0
    failed_transcribe = 0
    incomplete_routing = 0
    resolved_from_tar = 0
    resolved_from_disk = 0

    routing_buffer = {"layers": []}

    resolver = TarAudioResolver(tar_path=tar_path, temp_dir=temp_dir)

    def router_hook(module, inputs, output):
        """
        Expected router output:
        output = (top_k_weights, top_k_indices, logits)
        """
        try:
            top_k_indices = output[1]
            logits = output[2]
        except Exception as exc:
            print(f"Router hook failed to unpack output: {exc}")
            return

        n_experts = logits.shape[-1]

        soft = torch.softmax(logits.float(), dim=-1).squeeze(0)
        soft_mean = soft.mean(dim=0)

        hard_idx = top_k_indices.squeeze(0)
        hard_counts = torch.zeros(n_experts, device=hard_idx.device)

        for e in range(n_experts):
            hard_counts[e] = (hard_idx == e).sum()

        hard_total = hard_counts.sum().item()
        hard_probs = hard_counts / hard_total if hard_total > 0 else hard_counts

        top1_idx = hard_idx[:, 0]
        top1_counts = torch.zeros(n_experts, device=hard_idx.device)

        for e in range(n_experts):
            top1_counts[e] = (top1_idx == e).sum()

        top1_total = top1_counts.sum().item()

        routing_buffer["layers"].append(
            {
                "soft_probs": soft_mean.detach().cpu().numpy(),
                "hard_probs": hard_probs.detach().cpu().numpy(),
                "hard_counts": hard_counts.detach().cpu().numpy(),
                "hard_total": hard_total,
                "top1_counts": top1_counts.detach().cpu().numpy(),
                "top1_total": top1_total,
            }
        )

    hook_handle = model.encoder.global_router.register_forward_hook(router_hook)

    print(f"Processing {len(lines)} utterances [{age_group}/{dataset_source}]...")
    print(f"Manifest: {manifest_path}")
    print(f"Tar:      {tar_path}")

    try:
        for line in tqdm(lines):
            data = json.loads(line)

            manifest_audio_path = data.get("audio_filepath")
            resolved_audio_path, source = resolver.resolve(manifest_audio_path)

            if resolved_audio_path is None or not os.path.exists(resolved_audio_path):
                skipped_missing_audio += 1
                if skipped_missing_audio <= 10:
                    print(f"Skipping unresolved audio: {manifest_audio_path}")
                continue

            if source == "tar":
                resolved_from_tar += 1
            elif source == "disk":
                resolved_from_disk += 1

            duration_sec = data.get("duration", data.get("duration_sec"))
            try:
                duration_sec = float(duration_sec)
            except (TypeError, ValueError):
                duration_sec = np.nan

            routing_buffer["layers"] = []

            try:
                with torch.no_grad():
                    model.transcribe(
                        [resolved_audio_path],
                        batch_size=1,
                        verbose=False,
                    )
            except Exception as exc:
                failed_transcribe += 1
                if failed_transcribe <= 10:
                    print(f"Failed transcribe: {manifest_audio_path}")
                    print(f"Resolved path: {resolved_audio_path}")
                    print(f"Error: {exc}")
                continue

            if len(routing_buffer["layers"]) != 17:
                incomplete_routing += 1
                if incomplete_routing <= 10:
                    print(
                        "Warning: incomplete routing captured "
                        f"({len(routing_buffer['layers'])}/17) for "
                        f"{manifest_audio_path}"
                    )
                continue

            for layer_idx, layer_info in enumerate(routing_buffer["layers"]):
                hard_probs = layer_info["hard_probs"]
                soft_probs = layer_info["soft_probs"]
                hard_counts = layer_info["hard_counts"]
                top1_counts = layer_info["top1_counts"]

                soft_entropy = float(
                    -np.sum(soft_probs * np.log(soft_probs + 1e-10))
                )
                hard_entropy = float(
                    -np.sum(hard_probs * np.log(hard_probs + 1e-10))
                )

                row = {
                    "utterance_id": manifest_audio_path,
                    "resolved_audio_path": resolved_audio_path,
                    "audio_source": source,
                    "age_group": age_group,
                    "dataset_source": dataset_source,
                    "language": language,
                    "layer_idx": layer_idx,
                    "duration_sec": duration_sec,
                    "soft_router_entropy": soft_entropy,
                    "hard_router_entropy": hard_entropy,
                    "hard_total_selections": layer_info["hard_total"],
                    "top1_total_frames": layer_info["top1_total"],
                    "top1_expert": int(np.argmax(top1_counts))
                    if top1_counts.sum() > 0
                    else -1,
                    "top_soft_expert": int(np.argmax(soft_probs))
                    if soft_probs.sum() > 0
                    else -1,
                }

                for e in range(len(hard_probs)):
                    row[f"hard_expert_{e}_prob"] = float(hard_probs[e])
                    row[f"soft_expert_{e}_prob"] = float(soft_probs[e])
                    row[f"hard_expert_{e}_count"] = int(hard_counts[e])
                    row[f"top1_expert_{e}_count"] = int(top1_counts[e])

                records.append(row)

    finally:
        hook_handle.remove()
        resolver.close()

    df = pd.DataFrame(records)

    print(f"\nFinished {age_group}/{dataset_source}")
    print(f"  routing rows collected: {len(df)}")
    print(f"  utterances represented: {df['utterance_id'].nunique() if not df.empty else 0}")
    print(f"  resolved from tar:      {resolved_from_tar}")
    print(f"  resolved from disk:     {resolved_from_disk}")
    print(f"  skipped missing audio:  {skipped_missing_audio}")
    print(f"  failed transcribe:      {failed_transcribe}")
    print(f"  incomplete routing:     {incomplete_routing}")

    if df.empty:
        raise RuntimeError(
            f"No routing records extracted for {age_group}/{dataset_source}. "
            "The manifest was read, but no usable audio was transcribed. "
            "Check that audio_filepath values match members inside the tar."
        )

    return df


def main():
    args = parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    if args.max_utterances is not None and args.max_utterances < 0:
        max_utterances = None
    else:
        max_utterances = args.max_utterances

    model_paths = build_model_paths(args.model_dir, args.models)

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
            )

            df_yodas["model"] = model_name
            df_yodas.to_csv(out_path, index=False)

            print(f"Saved: {out_path}")
            print(f"Rows:  {len(df_yodas)}")

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


# CUDA_VISIBLE_DEVICES=3 python analysis/routing/run_yodas.py \
#   --models adult_moe child_moe_lb_on child_moe_lb_off \
#   --yodas-manifest /data/granary-pz/granary/version_1_0/manifests/manifests_all_pnc/ASR_updated/YODAS2/en/0_by_whisper/sharded_manifests_updated/manifest_53.json \
#   --yodas-tar /data/granary-pz/yodas2/en/0_by_whisper/audio_53.tar \
#   --cache-dir /lp-dev/amelia/inclusive-asr-moe/analysis/routing/outputs/yodas \
#   --max-utterances 300 \
#   --device cuda