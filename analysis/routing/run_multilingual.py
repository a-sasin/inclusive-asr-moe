"""
Extract routing distributions for multilingual MoE models.

This version is matched to the working English/YODAS extraction logic.

Output schema:
    utterance_id, age_group, language, dataset_source, layer_idx,
    router_entropy, top1_expert, top2_expert, duration_sec,
    expert_0_prob, expert_1_prob, expert_2_prob, expert_3_prob, model

Usage:
    CUDA_VISIBLE_DEVICES=3 python run_multilingual_routing.py

Smoke test:
    CUDA_VISIBLE_DEVICES=3 python run_multilingual_routing.py \
        --models child_moe_lb_off \
        --max-utterances-per-dataset 10 \
        --overwrite
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nemo.collections.asr as nemo_asr


MODEL_FILES = {
    "adult_moe": "multilingual_adult_moe.nemo",
    "child_moe_lb_on": "multilingual_child_moe_lb_on.nemo",
    "child_moe_lb_off": "multilingual_child_moe_lb_off.nemo",
}

NUM_LAYERS = 17
NUM_EXPERTS = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multilingual MoE routing and cache CSVs."
    )

    parser.add_argument(
        "--model-dir",
        default="/lp-dev/amelia/inclusive-asr-moe/final_weights",
        help="Directory containing .nemo model files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/lp-dev/amelia/inclusive-asr-moe2/analysis/routing/routing_outputs_multilingual",
        help="Output directory for cached routing CSVs.",
    )
    parser.add_argument(
        "--max-utterances-per-dataset",
        type=int,
        default=None,
        help=(
            "Maximum utterances to process per manifest/dataset. "
            "Use -1 or omit for all utterances."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run inference, e.g. cuda or cuda:0.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["adult_moe", "child_moe_lb_on", "child_moe_lb_off"],
        help="Subset of model keys to run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing cached CSVs.",
    )
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help="Print router tensor shapes for the first few hook calls.",
    )

    return parser.parse_args()


def load_manifest_lines(manifest_path, max_utterances=None):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_utterances is not None and max_utterances > 0:
        lines = lines[:max_utterances]

    return lines


def extract_routing_multilingual(
    model,
    manifest_path,
    age_group,
    language,
    dataset_source,
    max_utterances=None,
    device="cuda",
    debug_shapes=False,
):
    """
    Extract per-utterance routing distributions.

    Matched to English/YODAS logic:
        logits -> softmax -> squeeze batch if present -> mean over frames
    """
    model.eval()
    model = model.to(device)

    lines = load_manifest_lines(
        manifest_path,
        max_utterances=max_utterances,
    )

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

        # Safe version of probs.squeeze(0).mean(dim=0).
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

    stats = {
        "skipped_audio": 0,
        "failed_transcribe": 0,
        "incomplete_routing": 0,
        "processed": 0,
    }

    print(
        f"Processing {len(lines)} utterances "
        f"[{age_group}/{language}/{dataset_source}]..."
    )

    try:
        for line in tqdm(lines, desc=f"{age_group}/{language}/{dataset_source}"):
            data = json.loads(line)

            audio_path = data.get("audio_filepath")
            if not audio_path or not os.path.exists(audio_path):
                stats["skipped_audio"] += 1
                if stats["skipped_audio"] <= 10:
                    print(f"Skipping missing audio: {audio_path}")
                continue

            duration_sec = data.get("duration", data.get("duration_sec"))
            try:
                duration_sec = float(duration_sec)
            except (TypeError, ValueError):
                duration_sec = np.nan

            routing_buffer["probs_list"] = []

            try:
                with torch.no_grad():
                    model.transcribe(
                        [audio_path],
                        batch_size=1,
                        verbose=False,
                    )
            except Exception as exc:
                stats["failed_transcribe"] += 1
                if stats["failed_transcribe"] <= 10:
                    print(f"Failed transcribe: {audio_path}")
                    print(f"Error: {exc}")
                continue

            if len(routing_buffer["probs_list"]) != NUM_LAYERS:
                stats["incomplete_routing"] += 1
                if stats["incomplete_routing"] <= 10:
                    print(
                        "Warning: incomplete routing captured "
                        f"({len(routing_buffer['probs_list'])}/{NUM_LAYERS}) "
                        f"for {audio_path}"
                    )
                continue

            probs_all_layers = np.stack(routing_buffer["probs_list"])
            stats["processed"] += 1

            for layer_idx, probs in enumerate(probs_all_layers):
                probs = probs.astype(float)

                # Defensive normalization. Should already sum to 1.
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum

                entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

                row = {
                    "utterance_id": audio_path,
                    "age_group": age_group,
                    "language": language,
                    "dataset_source": dataset_source,
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

    df = pd.DataFrame(records)

    print(f"\nFinished [{age_group}/{language}/{dataset_source}]")
    print(f"  Routing rows:       {len(df):,}")
    print(f"  Utterances:         {df['utterance_id'].nunique() if not df.empty else 0:,}")
    print(f"  Processed:          {stats['processed']:,}")
    print(f"  Skipped audio:      {stats['skipped_audio']:,}")
    print(f"  Failed transcribe:  {stats['failed_transcribe']:,}")
    print(f"  Incomplete routing: {stats['incomplete_routing']:,}")

    return df


def load_or_extract_multilingual(
    model_name,
    model_path,
    manifests,
    cache_dir,
    max_utterances_per_dataset=None,
    device="cuda",
    overwrite=False,
    debug_shapes=False,
):
    cache_path = os.path.join(cache_dir, f"{model_name}_routing.csv")

    if os.path.exists(cache_path) and not overwrite:
        print(f"Loading cached: {cache_path}")
        return pd.read_csv(cache_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("\n" + "=" * 80)
    print(f"Loading model: {model_name}")
    print(f"Model path:    {model_path}")
    print(f"Output CSV:    {cache_path}")
    print("=" * 80)

    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        model_path,
        map_location=device,
    )

    dfs = []

    for manifest_path, age_group, language, dataset_source in manifests:
        if not os.path.exists(manifest_path):
            print(f"Skipping missing manifest: {manifest_path}")
            continue

        df_part = extract_routing_multilingual(
            model=model,
            manifest_path=manifest_path,
            age_group=age_group,
            language=language,
            dataset_source=dataset_source,
            max_utterances=max_utterances_per_dataset,
            device=device,
            debug_shapes=debug_shapes,
        )

        if not df_part.empty:
            dfs.append(df_part)

    if not dfs:
        raise RuntimeError(f"No routing records extracted for {model_name}")

    df = pd.concat(dfs, ignore_index=True)
    df["model"] = model_name

    prob_cols = [f"expert_{i}_prob" for i in range(NUM_EXPERTS)]
    prob_sums = df[prob_cols].sum(axis=1)

    print("\nProbability sanity check")
    print(prob_sums.describe())
    print("Min expert prob:", df[prob_cols].min().min())
    print("Max expert prob:", df[prob_cols].max().max())
    print(
        "Rows with prob sum not close to 1:",
        int(((prob_sums < 0.999) | (prob_sums > 1.001)).sum()),
    )

    df.to_csv(cache_path, index=False)

    del model
    torch.cuda.empty_cache()

    print(f"\nSaved {len(df):,} rows -> {cache_path}")
    return df


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.max_utterances_per_dataset is not None and args.max_utterances_per_dataset < 0:
        max_utterances_per_dataset = None
    else:
        max_utterances_per_dataset = args.max_utterances_per_dataset

    models = {}
    for model_name, filename in MODEL_FILES.items():
        models[model_name] = os.path.join(args.model_dir, filename)

    manifests = [
        ("/data/cv/nemo/nl/test.json", "adult", "nl", "commonvoice"),
        ("/data/cv/nemo/de/test.json", "adult", "de", "commonvoice"),
        ("/data/cv/nemo/pl/test.json", "adult", "pl", "commonvoice"),
        ("/data/librispeech_nemo/test_clean.json", "adult", "en", "librispeech"),
        ("/lp-dev/amelia/data/jasmin/child/test.json", "child", "nl", "jasmin"),
        ("/lp-dev/amelia/data/kidstalc/cleaned/val.json", "child", "de", "kidstalc"),
        (
            "/lp-dev/amelia/data/pavsig/training/new_test_manifest_mono.jsonl",
            "child",
            "pl",
            "pavsig",
        ),
        ("/lp-dev/amelia/data/myst/test.json", "child", "en", "myst"),
    ]

    all_routing = {}

    for model_name in args.models:
        if model_name not in models:
            print(f"Skipping unknown model key: {model_name}")
            continue

        all_routing[model_name] = load_or_extract_multilingual(
            model_name=model_name,
            model_path=models[model_name],
            manifests=manifests,
            cache_dir=args.cache_dir,
            max_utterances_per_dataset=max_utterances_per_dataset,
            device=args.device,
            overwrite=args.overwrite,
            debug_shapes=args.debug_shapes,
        )

    if "child_moe_lb_off" in all_routing:
        df_main = all_routing["child_moe_lb_off"]
        layer0 = df_main[df_main["layer_idx"] == 0]

        print("\nFinal summary for child_moe_lb_off")
        print(f"Rows: {len(df_main):,}")
        print("\nAge groups:")
        print(layer0["age_group"].value_counts())
        print("\nLanguages:")
        print(layer0["language"].value_counts())
        print("\nDataset sources:")
        print(layer0["dataset_source"].value_counts())


if __name__ == "__main__":
    main()