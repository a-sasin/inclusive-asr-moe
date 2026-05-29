# Training

Shell scripts for each stage of the training pipeline. All scripts use the NeMo `speech_to_text_ctc_bpe.py` entry point driven by Hydra configs.

## Pipeline Stages

```
NVIDIA pretrained checkpoint
        │
        ▼  weights_prep/prepare_checkpoint.py  (ff-reset + optional MoE conversion)
Initialised checkpoint  (baseline_weights/)
        │
        ├── English track ──────────────────────────────────────────────────────
        │   ├── stage2_adult_dense/moe.sh    Fine-tune on LibriSpeech (960 h)
        │   └── stage3_child_dense/moe.sh    Fine-tune on MyST + LibriSpeech mix
        │
        └── Multilingual track ──────────────────────────────────────────────────
            ├── stage1_adult_dense/moe.sh    Fine-tune on CV (NL/DE/PL) + LS (EN)
            └── stage3_child_dense/moe.sh    Fine-tune on child + adult mix (4 languages)
```

Note: the English track has no "Stage 1" because the NVIDIA pretrained weights already encode large-scale English ASR knowledge. Stage 2 adapts those weights to clean LibriSpeech before child fine-tuning.

## Usage

```bash
# Required: set GPU assignment before any training run
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Required for child fine-tuning stages: path to the stage 2/1 checkpoint
export PRETRAINED_NEMO=/path/to/stage2_or_stage1_checkpoint.nemo

bash training/english/stage2_adult_dense.sh
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | — | **Required.** GPU indices to use |
| `PRETRAINED_NEMO` | Hardcoded path in script | Override pretrained checkpoint |
| `EXPERIMENT_NAME` | Auto-generated timestamp | Override experiment directory name |

## Path Configuration

Each script has a `WORKSPACE`, `NEMO_DIR`, and `PROJECT_DIR` variable near the top. Update these before running:

```bash
WORKSPACE=/lp-dev/amelia                        # server workspace root
NEMO_DIR=${WORKSPACE}/NeMo                      # local NeMo fork with ConformerMoEEncoder
PROJECT_DIR=${WORKSPACE}/inclusive-asr-moe      # this repository
```

## Outputs

Each run writes to:
```
experiments/<track>/<stage>/<EXPERIMENT_NAME>/
├── checkpoints/          Best and last .nemo checkpoints
├── logs/
│   ├── train.log         Full training stdout
│   ├── run_info.txt      Metadata snapshot
│   └── config_snapshot.yaml
└── wandb_logs/
```

WandB project: `fastconformer-asr`

## Load-Balancing Variants

`stage3_child_moe_lb_on.sh` trains with the auxiliary load-balancing loss enabled (λ=0.002). `stage3_child_moe.sh` trains with it disabled (λ=0). See Section 3.3 of the thesis for the load-balancing formulation.
