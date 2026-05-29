import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nemo.collections.asr as nemo_asr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract multilingual MoE routing and cache CSVs so the notebook can read them."
        )
    )
    parser.add_argument(
        "--model-dir",
        default="/lp-dev/amelia/inclusive-asr-moe/final_weights",
        help="Directory containing .nemo model files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="routing_outputs_multilingual",
        help="Output directory for cached routing CSVs.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=None,
        help="Optional limit per manifest for smoke tests.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run inference (default: cuda).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["adult_moe", "child_moe_lb_on", "child_moe_lb_off"],
        help="Subset of model keys to run.",
    )
    return parser.parse_args()


def extract_routing_multilingual(
    model,
    manifest_path,
    age_group,
    language,
    dataset_source,
    n_layers,
    n_experts,
    max_utterances=None,
    device="cuda",
):
    """
    Same hook logic as the notebook.
    Adds language and dataset_source columns to every row.
    """
    model.eval()
    model = model.to(device)

    with open(manifest_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_utterances:
        lines = lines[:max_utterances]

    records = []
    layer_call_buffer = []

    def router_hook(module, inputs, output):
        logits = output[2]  # [1, T, 4]
        probs = torch.softmax(logits.float(), dim=-1)
        probs = probs.squeeze(0).mean(dim=0)  # [4]
        layer_call_buffer.append(probs.detach().cpu().numpy())

    h = model.encoder.global_router.register_forward_hook(router_hook)

    for line in tqdm(lines, desc=f"{age_group}/{language}/{dataset_source}"):
        data = json.loads(line)
        audio_path = data.get("audio_filepath")
        if not audio_path or not os.path.exists(audio_path):
            continue
        duration_sec = data.get("duration", data.get("duration_sec"))
        try:
            duration_sec = float(duration_sec)
        except (TypeError, ValueError):
            duration_sec = np.nan

        layer_call_buffer.clear()

        try:
            with torch.no_grad():
                model.transcribe([audio_path], batch_size=1, verbose=False)
        except Exception as e:
            print(f"Failed: {audio_path}: {e}")
            continue

        if len(layer_call_buffer) != n_layers:
            print(f"Warning: {len(layer_call_buffer)}/{n_layers} hooks — skipping")
            continue

        for layer_idx, probs in enumerate(layer_call_buffer):
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            row = {
                "utterance_id": audio_path,
                "age_group": age_group,
                "language": language,
                "dataset_source": dataset_source,
                "layer_idx": layer_idx,
                "router_entropy": entropy,
                "top1_expert": int(np.argmax(probs)),
                "duration_sec": duration_sec,
            }
            for e in range(n_experts):
                row[f"expert_{e}_prob"] = float(probs[e])
            records.append(row)

    h.remove()
    return pd.DataFrame(records)


def load_or_extract_multilingual(
    model_name,
    model_path,
    manifests,
    cache_dir,
    n_layers,
    n_experts,
    max_utterances=None,
    device="cuda",
):
    cache = f"{cache_dir}/{model_name}_routing.csv"
    if os.path.exists(cache):
        print(f"Loading cached: {cache}")
        return pd.read_csv(cache)

    print(f"\nLoading model: {model_name}")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        model_path, map_location=device
    )

    dfs = []
    for manifest_path, age_group, language, dataset_source in manifests:
        if not os.path.exists(manifest_path):
            print(f"  Skipping missing: {manifest_path}")
            continue
        df = extract_routing_multilingual(
            model,
            manifest_path,
            age_group,
            language,
            dataset_source,
            n_layers,
            n_experts,
            max_utterances=max_utterances,
            device=device,
        )
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df["model"] = model_name
    df.to_csv(cache, index=False)

    del model
    torch.cuda.empty_cache()
    print(f"Saved {len(df)} rows -> {cache}")
    return df


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    models = {
        "adult_moe": f"{args.model_dir}/multilingual_adult_moe.nemo",
        "child_moe_lb_on": f"{args.model_dir}/multilingual_child_moe_lb_on.nemo",
        "child_moe_lb_off": f"{args.model_dir}/multilingual_child_moe_lb_off.nemo",
    }

    manifests = [
        ("/data/cv/nemo/nl/test.json", "adult", "nl", "commonvoice"),
        ("/data/cv/nemo/de/test.json", "adult", "de", "commonvoice"),
        ("/data/cv/nemo/pl/test.json", "adult", "pl", "commonvoice"),
        ("/data/librispeech_nemo/test_clean.json", "adult", "en", "librispeech"),
        ("/lp-dev/amelia/data/jasmin/child/test.json", "child", "nl", "jasmin"),
        ("/lp-dev/amelia/data/kidstalc/cleaned/val.json", "child", "de", "kidstalc"),
        (
            "/lp-dev/amelia/data/pavsig/training/new_test_manifest.mono.jsonl",
            "child",
            "pl",
            "pavsig",
        ),
        ("/lp-dev/amelia/data/myst/test.json", "child", "en", "myst"),
    ]

    n_layers = 17
    n_experts = 4

    all_routing = {}
    for model_name in args.models:
        if model_name not in models:
            print(f"Skipping unknown model key: {model_name}")
            continue
        model_path = models[model_name]
        all_routing[model_name] = load_or_extract_multilingual(
            model_name,
            model_path,
            manifests,
            args.cache_dir,
            n_layers,
            n_experts,
            max_utterances=args.max_utterances,
            device=args.device,
        )

    if "child_moe_lb_off" in all_routing:
        df_main = all_routing["child_moe_lb_off"]
        print(f"\nRows: {len(df_main)}")
        print(
            "Age groups:\n"
            f"{df_main[df_main['layer_idx'] == 0]['age_group'].value_counts()}"
        )
        print(
            "\nLanguages:\n"
            f"{df_main[df_main['layer_idx'] == 0]['language'].value_counts()}"
        )
        print(
            "\nDataset sources:\n"
            f"{df_main[df_main['layer_idx'] == 0]['dataset_source'].value_counts()}"
        )


if __name__ == "__main__":
    main()
