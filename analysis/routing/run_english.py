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
        description="Extract English MoE routing and cache CSVs for the notebook."
    )
    parser.add_argument(
        "--model-dir",
        default="/lp-dev/amelia/inclusive-asr-moe/final_weights",
        help="Directory containing .nemo model files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="routing_outputs",
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


def extract_routing_per_utterance(
    model,
    manifest_path,
    age_group,
    dataset_source,
    language="en",
    max_utterances=None,
    device="cuda",
):
    """
    Extracts per-utterance expert routing distributions from OmniRouter.
    One row per utterance per layer in the output DataFrame.
    """
    model.eval()
    model = model.to(device)

    with open(manifest_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_utterances:
        lines = lines[:max_utterances]

    records = []
    routing_buffer = {"probs_list": []}

    def router_hook(module, inputs, output):
        logits = output[2]
        probs = torch.softmax(logits.float(), dim=-1)
        probs = probs.squeeze(0)
        probs = probs.mean(dim=0)
        routing_buffer["probs_list"].append(probs.detach().cpu().numpy())

    h = model.encoder.global_router.register_forward_hook(router_hook)

    print(f"Processing {len(lines)} utterances [{age_group}/{dataset_source}]...")

    for line in tqdm(lines):
        data = json.loads(line)
        audio_path = data.get("audio_filepath")
        if not audio_path or not os.path.exists(audio_path):
            continue
        duration_sec = data.get("duration", data.get("duration_sec"))
        try:
            duration_sec = float(duration_sec)
        except (TypeError, ValueError):
            duration_sec = np.nan

        routing_buffer["probs_list"] = []

        try:
            with torch.no_grad():
                model.transcribe([audio_path], batch_size=1, verbose=False)
        except Exception as e:
            print(f"Failed: {audio_path}: {e}")
            continue

        if len(routing_buffer["probs_list"]) != 17:
            print(
                "Warning: incomplete routing captured "
                f"({len(routing_buffer['probs_list'])}/17) for {audio_path}"
            )
            continue

        probs_all_layers = np.stack(routing_buffer["probs_list"])
        n_experts = probs_all_layers.shape[-1]

        for layer_idx in range(probs_all_layers.shape[0]):
            probs = probs_all_layers[layer_idx]
            entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
            row = {
                "utterance_id": audio_path,
                "age_group": age_group,
                "dataset_source": dataset_source,
                "language": language,
                "layer_idx": layer_idx,
                "router_entropy": entropy,
                "top1_expert": int(np.argmax(probs)),
                "top2_expert": int(np.argsort(probs)[-2]),
                "duration_sec": duration_sec,
            }
            for e in range(n_experts):
                row[f"expert_{e}_prob"] = float(probs[e])
            records.append(row)

    h.remove()
    return pd.DataFrame(records)


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    models = {
        "adult_moe": f"{args.model_dir}/en_adult_moe.nemo",
        "child_moe_lb_on": f"{args.model_dir}/en_child_moe_lb_on.nemo",
        "child_moe_lb_off": f"{args.model_dir}/en_child_moe_lb_off.nemo",
    }

    test_manifests = {
        "adult": "/data/librispeech_nemo/test_clean.json",
        "child": "/lp-dev/amelia/data/myst/test.json",
    }

    all_routing = {}
    for model_name in args.models:
        if model_name not in models:
            print(f"Skipping unknown model key: {model_name}")
            continue

        cache_path = f"{args.cache_dir}/{model_name}_routing.csv"
        if os.path.exists(cache_path):
            print(f"Loading cached routing for {model_name}")
            all_routing[model_name] = pd.read_csv(cache_path)
            continue

        print(f"\nLoading: {model_name}")
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            models[model_name], map_location=args.device
        )

        df_adult = extract_routing_per_utterance(
            model,
            test_manifests["adult"],
            age_group="adult",
            dataset_source="librispeech",
            language="en",
            max_utterances=args.max_utterances,
            device=args.device,
        )

        df_child = extract_routing_per_utterance(
            model,
            test_manifests["child"],
            age_group="child",
            dataset_source="myst",
            language="en",
            max_utterances=args.max_utterances,
            device=args.device,
        )

        df = pd.concat([df_adult, df_child], ignore_index=True)
        df["model"] = model_name
        df.to_csv(cache_path, index=False)
        all_routing[model_name] = df

        del model
        torch.cuda.empty_cache()
        print(f"Done - {len(df)} rows -> {cache_path}")

    if "child_moe_lb_off" in all_routing:
        df_main = all_routing["child_moe_lb_off"]
        print(f"\nRows: {len(df_main)}")
        print(
            "Age groups:\n"
            f"{df_main[df_main['layer_idx'] == 0]['age_group'].value_counts()}"
        )
        print(
            "\nDataset sources:\n"
            f"{df_main[df_main['layer_idx'] == 0]['dataset_source'].value_counts()}"
        )


if __name__ == "__main__":
    main()