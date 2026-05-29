# Weights Preparation

Utilities for converting the NVIDIA pretrained FastConformer-CTC checkpoint into the initialisation checkpoints used for training.

## Scripts

| Script | Purpose |
|---|---|
| `prepare_checkpoint.py` | English / single-language variants |
| `prepare_checkpoint_multilingual.py` | Multilingual variants (different default config paths) |

Both scripts share the same modes; the multilingual variant uses configs under `configs/multilingual/`.

## Modes

### `inspect` — Print architecture summary

```bash
python weights_prep/prepare_checkpoint.py --mode inspect \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large.nemo
```

### `ff-reset` — Re-initialise feedforward layers

Resets all FF linear weights to Kaiming uniform before training. Used for both dense and MoE initialisations to prevent the pretrained FF representations from dominating early training.

```bash
python weights_prep/prepare_checkpoint.py --mode ff-reset \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large.nemo \
    --output-path baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo
```

### `to-moe` — Convert dense checkpoint → MoE initialisation

Copies all shape-compatible weights (attention, convolution, layer norms) from the dense checkpoint into a freshly constructed MoE model. The single dense FF weights are replicated across all N experts as a symmetric starting point.

```bash
# English (4 experts)
python weights_prep/prepare_checkpoint.py --mode to-moe \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo \
    --moe-config configs/english/adult_moe.yaml \
    --output-path baseline_weights/stt_en_fastconformer_ctc_large_to_moe_init.nemo

# Multilingual (8 experts)
python weights_prep/prepare_checkpoint_multilingual.py --mode to-moe \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo \
    --moe-config configs/multilingual/adult_moe.yaml \
    --output-path baseline_weights/stt_en_fastconformer_ctc_large_to_moe_init_ml.nemo
```

Use `--dry-run` to validate without writing, `--overwrite` to replace an existing output.

### `to-fast` — Re-initialise from a different FastConformer config

```bash
python weights_prep/prepare_checkpoint.py --mode to-fast \
    --model-path baseline_weights/source.nemo \
    --fast-config configs/english/adult_dense.yaml \
    --output-path baseline_weights/source_fast_init.nemo
```

## Expected Baseline Weights

Place the following in `baseline_weights/` before running any preparation:

```
baseline_weights/
└── stt_en_fastconformer_ctc_large.nemo    ← NVIDIA pretrained (download from NGC)
```
