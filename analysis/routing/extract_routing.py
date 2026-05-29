import argparse
import json
import os

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


def load_yodas_manifest_path(yaml_path):
    try:
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        if isinstance(data, list) and data:
            return data[0].get("manifest_filepath")
    except Exception:
        pass
    with open(yaml_path, "r") as f:
        for line in f:
            if "manifest_filepath:" in line:
                return line.split("manifest_filepath:", 1)[1].strip()
    return None


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
    Extracts per-utterance routing distributions from OmniRouter.

    Records both:
    - hard expert load (top-k selections)
    - soft router probabilities (mean softmax)
    """
    model.eval()
    model = model.to(device)

    with open(manifest_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if max_utterances:
        lines = lines[:max_utterances]

    records = []
    routing_buffer = {"layers": []}

    def router_hook(module, inputs, output):
        # output = (top_k_weights, top_k_indices, logits)
        top_k_indices = output[1]
        logits = output[2]
        n_experts = logits.shape[-1]

        soft = torch.softmax(logits.float(), dim=-1).squeeze(0)  # [T, E]
        soft_mean = soft.mean(dim=0)

        hard_idx = top_k_indices.squeeze(0)  # [T, k]
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

        routing_buffer["layers"] = []

        try:
            with torch.no_grad():
                model.transcribe([audio_path], batch_size=1, verbose=False)
        except Exception as e:
            print(f"Failed: {audio_path}: {e}")
            continue

        if len(routing_buffer["layers"]) != 17:
            print(
                f"Warning: incomplete routing captured ({len(routing_buffer['layers'])}/17) for {audio_path}"
            )
            continue

        for layer_idx, layer_info in enumerate(routing_buffer["layers"]):
            hard_probs = layer_info["hard_probs"]
            soft_probs = layer_info["soft_probs"]
            hard_counts = layer_info["hard_counts"]
            top1_counts = layer_info["top1_counts"]

            row = {
                "utterance_id": audio_path,
                "age_group": age_group,
                "dataset_source": dataset_source,
                "language": language,
                "layer_idx": layer_idx,
                "duration_sec": duration_sec,
                "hard_total_selections": layer_info["hard_total"],
                "top1_total_frames": layer_info["top1_total"],
                "top1_expert": int(np.argmax(top1_counts)) if top1_counts.sum() > 0 else -1,
            }

            for e in range(len(hard_probs)):
                row[f"hard_expert_{e}_prob"] = float(hard_probs[e])
                row[f"soft_expert_{e}_prob"] = float(soft_probs[e])
                row[f"hard_expert_{e}_count"] = int(hard_counts[e])
                row[f"top1_expert_{e}_count"] = int(top1_counts[e])

            records.append(row)

    h.remove()
    return pd.DataFrame(records)


def build_model_paths(model_dir, model_names):
    paths = {}
    for name in model_names:
        file_name = MODEL_FILES.get(name)
        if not file_name:
            raise ValueError(f"Unknown model name: {name}")
        paths[name] = os.path.join(model_dir, file_name)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Extract routing CSVs for English MoE models")
    parser.add_argument("--model-dir", default="/lp-dev/amelia/inclusive-asr-moe/final_weights")
    parser.add_argument("--cache-dir", default="/lp-dev/amelia/inclusive-asr-moe/analysis/routing/outputs/english")
    parser.add_argument("--adult-manifest", default="/data/librispeech_nemo/test_clean.json")
    parser.add_argument("--child-manifest", default="/lp-dev/amelia/data/myst/test.json")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        default="adult_moe,child_moe_lb_on,child_moe_lb_off",
        help="Comma-separated model names",
    )
    parser.add_argument("--domain-control", action="store_true")
    parser.add_argument("--yodas-only", action="store_true")
    parser.add_argument("--domain-model", default="child_moe_lb_off")
    parser.add_argument(
        "--yodas-yaml",
        default="/lp-dev/amelia/inclusive-asr-moe/analysis/routing/test_ood.yaml",
    )
    parser.add_argument("--domain-max-utterances", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    if args.yodas_only:
        yodas_manifest = load_yodas_manifest_path(args.yodas_yaml)
        if not yodas_manifest or not os.path.exists(yodas_manifest):
            raise FileNotFoundError(f"YODAS manifest not found: {yodas_manifest}")

        model_paths = build_model_paths(args.model_dir, [args.domain_model])
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            model_paths[args.domain_model], map_location=args.device
        )

        df_yodas = extract_routing_per_utterance(
            model,
            yodas_manifest,
            age_group="adult",
            dataset_source="yodas",
            language=args.language,
            max_utterances=args.domain_max_utterances,
            device=args.device,
        )

        out_path = os.path.join(args.cache_dir, "yodas_routing.csv")
        df_yodas.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

        del model
        torch.cuda.empty_cache()
        return

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    model_paths = build_model_paths(args.model_dir, model_names)

    for model_name, model_path in model_paths.items():
        print(f"\n=== {model_name} ===")
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            model_path, map_location=args.device
        )

        df_adult = extract_routing_per_utterance(
            model,
            args.adult_manifest,
            age_group="adult",
            dataset_source="librispeech",
            language=args.language,
            max_utterances=args.max_utterances,
            device=args.device,
        )
        df_child = extract_routing_per_utterance(
            model,
            args.child_manifest,
            age_group="child",
            dataset_source="myst",
            language=args.language,
            max_utterances=args.max_utterances,
            device=args.device,
        )

        df_all = pd.concat([df_adult, df_child], ignore_index=True)
        out_path = os.path.join(args.cache_dir, f"{model_name}_routing.csv")
        df_all.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

        del model
        torch.cuda.empty_cache()

    if args.domain_control:
        domain_model_path = build_model_paths(args.model_dir, [args.domain_model])[args.domain_model]
        model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
            domain_model_path, map_location=args.device
        )

        df_librispeech = extract_routing_per_utterance(
            model,
            args.adult_manifest,
            age_group="adult",
            dataset_source="librispeech",
            language=args.language,
            max_utterances=args.domain_max_utterances,
            device=args.device,
        )
        df_myst = extract_routing_per_utterance(
            model,
            args.child_manifest,
            age_group="child",
            dataset_source="myst",
            language=args.language,
            max_utterances=args.domain_max_utterances,
            device=args.device,
        )
        df_yodas = pd.DataFrame()
        yodas_manifest = load_yodas_manifest_path(args.yodas_yaml)
        if yodas_manifest and os.path.exists(yodas_manifest):
            df_yodas = extract_routing_per_utterance(
                model,
                yodas_manifest,
                age_group="adult",
                dataset_source="yodas",
                language=args.language,
                max_utterances=args.domain_max_utterances,
                device=args.device,
            )
        else:
            print(f"YODAS manifest not found: {yodas_manifest}")

        df_domain = pd.concat([df_librispeech, df_myst, df_yodas], ignore_index=True)
        out_domain = os.path.join(args.cache_dir, "domain_control_routing.csv")
        df_domain.to_csv(out_domain, index=False)
        print(f"Saved {out_domain}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
