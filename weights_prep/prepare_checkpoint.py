#!/usr/bin/env python3
"""Unified FastConformer preparation utility.

Supported modes:
1) inspect    : print architecture + feedforward match sanity
2) ff-reset   : reset dense feedforward linear weights only
3) to-moe     : convert dense checkpoint to MoE checkpoint using a MoE config
4) to-fast    : convert dense checkpoint to FastConformer checkpoint using a Fast config
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


DEFAULT_MODEL_PATH = (
	"/lp-dev/amelia/inclusive-asr-moe/baseline_weights/stt_en_fastconformer_ctc_large.nemo"
)
DEFAULT_MOE_CFG = "/lp-dev/amelia/inclusive-asr-moe/configs/english/adult_moe.yaml"
DEFAULT_FAST_CFG = "/lp-dev/amelia/inclusive-asr-moe/configs/english/adult_dense.yaml"


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Prepare FastConformer checkpoints.")
	parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
	parser.add_argument(
		"--mode",
		choices=["inspect", "ff-reset", "to-moe", "to-fast"],
		default="inspect",
		help="Operation mode. Legacy flags can still auto-select mode.",
	)

	# Inspection / FF reset args
	parser.add_argument("--show-arch", action="store_true")
	parser.add_argument(
		"--max-module-lines",
		type=int,
		default=200,
		help="Maximum number of module lines to print when --show-arch is used.",
	)
	parser.add_argument(
		"--reset-feedforward",
		action="store_true",
		help="Legacy flag for ff-reset mode.",
	)
	parser.add_argument(
		"--ff-regex",
		type=str,
		default=r"(^|\.)(feed_?forward\d*|ffn\d*|ff_?layer\d*|dense_?relu_?dense)(\.|$)",
		help="Regex used to identify feedforward Linear layer names.",
	)

	# MoE conversion args
	parser.add_argument("--to-moe", action="store_true", help="Legacy flag for to-moe mode.")
	parser.add_argument("--moe-config", type=str, default=DEFAULT_MOE_CFG)
	parser.add_argument("--fast-config", type=str, default=DEFAULT_FAST_CFG)
	parser.add_argument(
		"--tokenizer-dir",
		type=str,
		default="",
		help="Optional override for model.tokenizer.dir when building MoE model.",
	)
	parser.add_argument(
		"--apply-ff-reset",
		action="store_true",
		help="After config-based conversion, reset feedforward linear layers in the target model.",
	)

	# Shared output args
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--output-path", type=str, default="")
	parser.add_argument("--overwrite", action="store_true")
	parser.add_argument("--dry-run", action="store_true")
	return parser


def choose_mode(args: argparse.Namespace) -> str:
	if args.to_moe:
		return "to-moe"
	if args.reset_feedforward:
		return "ff-reset"
	if args.mode != "inspect":
		return args.mode
	if args.show_arch:
		return "inspect"
	return "inspect"


def load_nemo_model(model_path: str):
	from nemo.collections.asr.models import EncDecCTCModelBPE

	return EncDecCTCModelBPE.restore_from(restore_path=model_path)


def summarize_architecture(model: nn.Module, max_module_lines: int = 200) -> None:
	print("\n=== MODEL CLASS ===")
	print(type(model))
	if hasattr(model, "encoder"):
		print("\n=== ENCODER CLASS ===")
		print(type(model.encoder))
	print("\n=== TOP LEVEL MODEL (truncated) ===")
	model_lines = str(model).splitlines()
	for line in model_lines[:max_module_lines]:
		print(line)
	if len(model_lines) > max_module_lines:
		print(f"... truncated {len(model_lines) - max_module_lines} lines ...")


def get_feedforward_linear_layers(model: nn.Module, ff_regex: str) -> List[Tuple[str, nn.Linear]]:
	ff_name_pattern = re.compile(ff_regex, re.IGNORECASE)
	matches: List[Tuple[str, nn.Linear]] = []
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear) and ff_name_pattern.search(name):
			matches.append((name, module))
	return matches


def print_linear_name_samples(model: nn.Module, max_items: int = 80) -> None:
	print("\nNo feedforward layers matched. Showing Linear module names for debugging:")
	shown = 0
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear):
			print(f"  - {name}")
			shown += 1
			if shown >= max_items:
				break
	if shown == 0:
		print("  (no nn.Linear modules found)")


def summarize_ff_matches(model: nn.Module, ff_layers: List[Tuple[str, nn.Linear]]) -> None:
	layer_pattern = re.compile(r"encoder\.layers\.(\d+)\.")
	counts = collections.Counter()
	for name, _ in ff_layers:
		m = layer_pattern.search(name)
		if m:
			counts[int(m.group(1))] += 1

	n_layers = None
	if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
		try:
			n_layers = len(model.encoder.layers)
		except TypeError:
			n_layers = None

	print("\n=== FEEDFORWARD MATCH SANITY ===")
	if n_layers is not None:
		expected = n_layers * 4
		print(f"encoder_layers={n_layers} expected_ff_linear_matches={expected}")
	print(f"found_ff_linear_matches={len(ff_layers)}")
	if counts:
		print("per_encoder_layer_ff_linear_counts:")
		for idx in sorted(counts):
			print(f"  layer_{idx}: {counts[idx]}")


def reset_linear(linear: nn.Linear) -> None:
	nn.init.kaiming_uniform_(linear.weight, a=5**0.5)
	if linear.bias is not None:
		nn.init.zeros_(linear.bias)


def reset_feedforward_layers(ff_layers: List[Tuple[str, nn.Linear]], dry_run: bool = False) -> Dict[str, int]:
	total_params = 0
	for layer_name, linear in ff_layers:
		layer_params = linear.weight.numel() + (linear.bias.numel() if linear.bias is not None else 0)
		total_params += layer_params
		print(
			f"[FF] {layer_name}: weight={tuple(linear.weight.shape)} "
			f"bias={'yes' if linear.bias is not None else 'no'} params={layer_params}"
		)
		if not dry_run:
			reset_linear(linear)
	return {"num_layers": len(ff_layers), "num_params": total_params}


def resolve_output_path(input_path: Path, output_path_arg: str, mode: str) -> Path:
	if output_path_arg:
		return Path(output_path_arg)
	if mode == "to-moe":
		return input_path.with_name(f"{input_path.stem}_to_moe_init.nemo")
	if mode == "to-fast":
		return input_path.with_name(f"{input_path.stem}_to_fast_from_config.nemo")
	return input_path.with_name(f"{input_path.stem}_ff_reset.nemo")


def copy_shape_compatible_weights(
	dense_state: Dict[str, torch.Tensor], moe_state: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
	stats = {"direct_copied": 0, "missing_in_target": 0, "shape_mismatch": 0}
	updated = dict(moe_state)
	for key, src_tensor in dense_state.items():
		if key not in updated:
			stats["missing_in_target"] += 1
			continue
		if updated[key].shape != src_tensor.shape:
			stats["shape_mismatch"] += 1
			continue
		updated[key] = src_tensor.detach().clone()
		stats["direct_copied"] += 1
	return updated, stats


def map_dense_ff2_to_moe(
	dense_state: Dict[str, torch.Tensor],
	moe_state: Dict[str, torch.Tensor],
	n_layers: int,
) -> Dict[str, int]:
	stats = {"layers_mapped": 0, "expert_params_written": 0, "layers_skipped": 0}

	for i in range(n_layers):
		dense_prefix = f"encoder.layers.{i}"
		moe_prefix = f"encoder.layers.{i}"

		norm2_w = f"{dense_prefix}.norm_feed_forward2.weight"
		norm2_b = f"{dense_prefix}.norm_feed_forward2.bias"
		ff2_l1_w = f"{dense_prefix}.feed_forward2.linear1.weight"
		ff2_l1_b = f"{dense_prefix}.feed_forward2.linear1.bias"
		ff2_l2_w = f"{dense_prefix}.feed_forward2.linear2.weight"
		ff2_l2_b = f"{dense_prefix}.feed_forward2.linear2.bias"

		tgt_norm_w = f"{moe_prefix}.norm_feed_forward.weight"
		tgt_norm_b = f"{moe_prefix}.norm_feed_forward.bias"
		tgt_w1 = f"{moe_prefix}.feed_forward.w1"
		tgt_w2 = f"{moe_prefix}.feed_forward.w2"
		tgt_b1 = f"{moe_prefix}.feed_forward.b1"
		tgt_b2 = f"{moe_prefix}.feed_forward.b2"

		required_dense = [norm2_w, norm2_b, ff2_l1_w, ff2_l2_w]
		required_moe = [tgt_norm_w, tgt_norm_b, tgt_w1, tgt_w2]
		if not all(k in dense_state for k in required_dense) or not all(k in moe_state for k in required_moe):
			stats["layers_skipped"] += 1
			continue

		moe_state[tgt_norm_w] = dense_state[norm2_w].detach().clone()
		moe_state[tgt_norm_b] = dense_state[norm2_b].detach().clone()

		src_l1_w = dense_state[ff2_l1_w].detach()
		src_l2_w = dense_state[ff2_l2_w].detach()
		moe_state[tgt_w1] = src_l1_w.unsqueeze(0).repeat(moe_state[tgt_w1].shape[0], 1, 1).clone()
		moe_state[tgt_w2] = src_l2_w.unsqueeze(0).repeat(moe_state[tgt_w2].shape[0], 1, 1).clone()
		stats["expert_params_written"] += moe_state[tgt_w1].numel() + moe_state[tgt_w2].numel()

		if ff2_l1_b in dense_state and tgt_b1 in moe_state and moe_state[tgt_b1] is not None:
			src_l1_b = dense_state[ff2_l1_b].detach()
			moe_state[tgt_b1] = src_l1_b.unsqueeze(0).repeat(moe_state[tgt_b1].shape[0], 1).clone()
			stats["expert_params_written"] += moe_state[tgt_b1].numel()

		if ff2_l2_b in dense_state and tgt_b2 in moe_state and moe_state[tgt_b2] is not None:
			src_l2_b = dense_state[ff2_l2_b].detach()
			moe_state[tgt_b2] = src_l2_b.unsqueeze(0).repeat(moe_state[tgt_b2].shape[0], 1).clone()
			stats["expert_params_written"] += moe_state[tgt_b2].numel()

		stats["layers_mapped"] += 1

	return stats


def run_inspect_or_ff_reset(args: argparse.Namespace, mode: str) -> None:
	model_path = Path(args.model_path)
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")

	model = load_nemo_model(str(model_path))
	if args.show_arch or mode == "inspect":
		summarize_architecture(model, max_module_lines=args.max_module_lines)

	ff_layers = get_feedforward_linear_layers(model, ff_regex=args.ff_regex)
	print("\n=== FEEDFORWARD LINEAR LAYERS FOUND ===")
	print(f"count={len(ff_layers)}")
	summarize_ff_matches(model, ff_layers)
	if len(ff_layers) == 0:
		print_linear_name_samples(model)

	if mode != "ff-reset":
		return

	output_path = resolve_output_path(model_path, args.output_path, mode=mode)
	if output_path.resolve() == model_path.resolve():
		raise ValueError("Refusing in-place overwrite: output checkpoint path must differ from model path.")
	if output_path.exists() and not args.overwrite:
		raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

	stats = reset_feedforward_layers(ff_layers, dry_run=args.dry_run)
	print(f"\nReset summary: layers={stats['num_layers']} params={stats['num_params']} dry_run={args.dry_run}")
	if not args.dry_run:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		model.save_to(str(output_path))
		print(f"Saved updated model to: {output_path}")


def run_dense_to_moe(args: argparse.Namespace) -> None:
	from omegaconf import OmegaConf
	import nemo.collections.asr as nemo_asr

	src_nemo = Path(args.model_path)
	if not src_nemo.exists():
		raise FileNotFoundError(f"Source checkpoint not found: {src_nemo}")

	moe_cfg = Path(args.moe_config)
	if not moe_cfg.exists():
		raise FileNotFoundError(f"MoE config not found: {moe_cfg}")

	output_path = resolve_output_path(src_nemo, args.output_path, mode="to-moe")
	if output_path.resolve() == src_nemo.resolve():
		raise ValueError("Refusing in-place overwrite: output path must differ from source checkpoint.")
	if output_path.exists() and not args.overwrite:
		raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

	dense_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(src_nemo))
	cfg = OmegaConf.load(str(moe_cfg))
	if args.tokenizer_dir:
		cfg.model.tokenizer.dir = args.tokenizer_dir
		cfg.model.tokenizer.type = "bpe"
	moe_model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg.model)

	dense_state = dense_model.state_dict()
	moe_state = moe_model.state_dict()
	moe_state, direct_stats = copy_shape_compatible_weights(dense_state, moe_state)
	source_n_layers = len(dense_model.encoder.layers)
	n_layers = len(moe_model.encoder.layers)
	ff_map_stats = map_dense_ff2_to_moe(dense_state, moe_state, n_layers=n_layers)
	missing, unexpected = moe_model.load_state_dict(moe_state, strict=False)

	print("\n=== CONVERSION SUMMARY ===")
	print(f"source: {src_nemo}")
	print(f"target-config: {moe_cfg}")
	print(f"source_encoder_layers: {source_n_layers}")
	print(f"target-encoder-class: {type(moe_model.encoder)}")
	print(f"encoder_layers: {n_layers}")
	if source_n_layers != n_layers:
		print(
			"WARNING: source and target encoder layer counts differ; unmatched layers are not transferable."
		)
	print(f"direct_copied: {direct_stats['direct_copied']}")
	print(f"shape_mismatch_skipped: {direct_stats['shape_mismatch']}")
	print(f"missing_in_target_skipped: {direct_stats['missing_in_target']}")
	print(f"ff_layers_mapped: {ff_map_stats['layers_mapped']}")
	print(f"ff_layers_skipped: {ff_map_stats['layers_skipped']}")
	print(f"ff_expert_params_written: {ff_map_stats['expert_params_written']}")
	print(f"post_load_missing_keys: {len(missing)}")
	print(f"post_load_unexpected_keys: {len(unexpected)}")

	if args.dry_run:
		print("\nDry run enabled: not saving checkpoint.")
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)
	moe_model.save_to(str(output_path))
	print(f"Saved converted MoE checkpoint: {output_path}")


def run_dense_to_fast_config(args: argparse.Namespace) -> None:
	from omegaconf import OmegaConf
	import nemo.collections.asr as nemo_asr

	src_nemo = Path(args.model_path)
	if not src_nemo.exists():
		raise FileNotFoundError(f"Source checkpoint not found: {src_nemo}")

	fast_cfg = Path(args.fast_config)
	if not fast_cfg.exists():
		raise FileNotFoundError(f"FastConformer config not found: {fast_cfg}")

	output_path = resolve_output_path(src_nemo, args.output_path, mode="to-fast")
	if output_path.resolve() == src_nemo.resolve():
		raise ValueError("Refusing in-place overwrite: output path must differ from source checkpoint.")
	if output_path.exists() and not args.overwrite:
		raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")

	dense_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(str(src_nemo))
	cfg = OmegaConf.load(str(fast_cfg))
	if args.tokenizer_dir:
		cfg.model.tokenizer.dir = args.tokenizer_dir
		cfg.model.tokenizer.type = "bpe"
	target_model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg.model)

	src_state = dense_model.state_dict()
	tgt_state = target_model.state_dict()
	tgt_state, copy_stats = copy_shape_compatible_weights(src_state, tgt_state)
	source_n_layers = len(dense_model.encoder.layers)
	target_n_layers = len(target_model.encoder.layers)
	missing, unexpected = target_model.load_state_dict(tgt_state, strict=False)

	ff_stats = None
	if args.apply_ff_reset:
		ff_layers = get_feedforward_linear_layers(target_model, ff_regex=args.ff_regex)
		ff_stats = reset_feedforward_layers(ff_layers, dry_run=False)

	print("\n=== FAST-CONFORMER CONFIG CONVERSION SUMMARY ===")
	print(f"source: {src_nemo}")
	print(f"target-config: {fast_cfg}")
	print(f"source_encoder_layers: {source_n_layers}")
	print(f"target-encoder-class: {type(target_model.encoder)}")
	print(f"encoder_layers: {target_n_layers}")
	if source_n_layers != target_n_layers:
		print(
			"WARNING: source and target encoder layer counts differ; unmatched layers are not transferable."
		)
	print(f"direct_copied: {copy_stats['direct_copied']}")
	print(f"shape_mismatch_skipped: {copy_stats['shape_mismatch']}")
	print(f"missing_in_target_skipped: {copy_stats['missing_in_target']}")
	print(f"post_load_missing_keys: {len(missing)}")
	print(f"post_load_unexpected_keys: {len(unexpected)}")
	if ff_stats is not None:
		print(
			f"ff_reset_applied: layers={ff_stats['num_layers']} params={ff_stats['num_params']}"
		)

	if args.dry_run:
		print("\nDry run enabled: not saving checkpoint.")
		return

	output_path.parent.mkdir(parents=True, exist_ok=True)
	target_model.save_to(str(output_path))
	print(f"Saved converted FastConformer checkpoint: {output_path}")


def main() -> None:
	args = build_parser().parse_args()
	torch.manual_seed(args.seed)
	mode = choose_mode(args)

	if mode in ("inspect", "ff-reset"):
		run_inspect_or_ff_reset(args, mode=mode)
		return

	if mode == "to-moe":
		run_dense_to_moe(args)
		return

	if mode == "to-fast":
		run_dense_to_fast_config(args)
		return

	raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
	main()