#!/bin/bash
# ============================================================================
# Fine-tune Dense FastConformer with Feedforward Layers Reinitialized
# ============================================================================
# Workflow:
#   1) Load pretrained dense FastConformer .nemo
#   2) Reinitialize feedforward linear layers only
#   3) Save as a NEW .nemo checkpoint (never in-place)
#   4) Fine-tune from this new checkpoint via +init_from_nemo_model
#
# Usage:
#   ./scripts/finetune/english/finetune_fastconformer_ffreset.sh
#
# Optional overrides:
#   PRETRAINED_NEMO=/path/base.nemo RESET_NEMO=/path/reset.nemo \
#   CUDA_VISIBLE_DEVICES=0,1 ./scripts/finetune/english/finetune_fastconformer_ffreset.sh
# ============================================================================

set -euo pipefail

source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate nemo_asr

PYTHON=/home/nvidia/miniconda3/envs/nemo_asr/bin/python3

PROJECT_ROOT=/lp-dev/amelia
NEMO_DIR=${PROJECT_ROOT}/NeMo
CONFIG_PATH=${PROJECT_ROOT}/inclusive-asr-moe/configs/NEW/english
CONFIG_NAME=fast-conformer_ctc_bpe.yaml
TOKENIZER_DIR=${PROJECT_ROOT}/inclusive-asr-moe/tokenizers/granary_en_bpe_4096/tokenizer_spe_bpe_v4096

PRETRAINED_NEMO=${PRETRAINED_NEMO:-"${PROJECT_ROOT}/inclusive-asr-moe/baseline_weights/stt_en_fastconformer_ctc_large.nemo"}
RESET_NEMO=${RESET_NEMO:-"${PROJECT_ROOT}/inclusive-asr-moe/baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EXPERIMENT_NAME="fastconformer_dense_ffreset_finetune_${TIMESTAMP}"
EXP_DIR=${PROJECT_ROOT}/inclusive-asr-moe/experiments/english/finetune/${EXPERIMENT_NAME}
LOG_DIR=${EXP_DIR}/logs

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "FastConformer Dense FF-reset Fine-tuning"
echo "========================================"
echo "Pretrained: ${PRETRAINED_NEMO}"
echo "FF-reset:   ${RESET_NEMO}"
echo "Config:     ${CONFIG_PATH}/${CONFIG_NAME}"
echo "Exp dir:    ${EXP_DIR}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo ""

if [ ! -f "${PRETRAINED_NEMO}" ]; then
    echo "ERROR: Pretrained .nemo not found at ${PRETRAINED_NEMO}"
    exit 1
fi

if [ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]; then
    echo "ERROR: tokenizer.model not found at ${TOKENIZER_DIR}"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}/${CONFIG_NAME}" ]; then
    echo "ERROR: Config not found at ${CONFIG_PATH}/${CONFIG_NAME}"
    exit 1
fi

# Install local NeMo fork
${PYTHON} -m pip install -e "${NEMO_DIR}" 2>&1 | tail -3

# Step 1: Prepare a NEW checkpoint with randomized feedforward layers
if [ -f "${RESET_NEMO}" ]; then
    echo "FF-reset checkpoint already exists: ${RESET_NEMO}"
    echo "Reusing existing file (set RESET_NEMO to a new path to regenerate)."
else
    echo "Preparing FF-reset checkpoint..."
    ${PYTHON} ${PROJECT_ROOT}/inclusive-asr-moe/weights_prep/prep_fast_conformer_english.py \
        --model-path "${PRETRAINED_NEMO}" \
        --reset-feedforward \
        --output-path "${RESET_NEMO}" \
        2>&1 | tee "${LOG_DIR}/ff_reset_prep.log"
fi

if [ ! -f "${RESET_NEMO}" ]; then
    echo "ERROR: FF-reset checkpoint was not created at ${RESET_NEMO}"
    exit 1
fi

# Metadata snapshot
{
    echo "date: $(date -Is)"
    echo "hostname: $(hostname)"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "pretrained_nemo: ${PRETRAINED_NEMO}"
    echo "ff_reset_nemo: ${RESET_NEMO}"
    echo "config: ${CONFIG_PATH}/${CONFIG_NAME}"
    echo "python: $(${PYTHON} --version 2>&1)"
} > "${LOG_DIR}/run_info.txt"

cp -f "${CONFIG_PATH}/${CONFIG_NAME}" "${LOG_DIR}/config_snapshot.yaml"

# Step 2: Fine-tune from the FF-reset checkpoint
export HYDRA_FULL_ERROR=1
cd "${PROJECT_ROOT}"

${PYTHON} ${NEMO_DIR}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    exp_manager.exp_dir="${EXP_DIR}" \
    model.tokenizer.dir="${TOKENIZER_DIR}" \
    trainer.use_distributed_sampler=false \
    trainer.devices=${NUM_GPUS} \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    +init_from_nemo_model="${RESET_NEMO}" \
    exp_manager.wandb_logger_kwargs.name="${EXPERIMENT_NAME}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "========================================"
echo "FF-reset dense fine-tuning finished"
echo "Experiment: ${EXP_DIR}"
echo "Log:        ${LOG_DIR}/train.log"
echo "========================================"
