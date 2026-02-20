#!/bin/bash
# ============================================================================
# Conformer-CTC-BPE Training (English) — 4 GPU robust script
# Tokenizer already trained
# ============================================================================

set -euo pipefail

############################################
# ENVIRONMENT
############################################
source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate nemo_asr

PYTHON=/home/nvidia/miniconda3/envs/nemo_asr/bin/python3

PROJECT_ROOT=/lp-dev/amelia/
NEMO_DIR=${PROJECT_ROOT}/NeMo
CONFIG_PATH=${PROJECT_ROOT}/inclusive-asr-moe/configs/english
CONFIG_NAME=conformer_ctc_bpe_english.yaml



TOKENIZER_DIR=${PROJECT_ROOT}/inclusive-asr-moe/tokenizers/granary_en_bpe_4096/tokenizer_spe_bpe_v4096

# Choose 4 GPUs explicitly (edit if you want different IDs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

EXPERIMENT_NAME="conformer_en_$(date +%Y-%m-%d_%H-%M-%S)"
EXP_DIR=${PROJECT_ROOT}/experiments/Conformer/${EXPERIMENT_NAME}
LOG_DIR=${EXP_DIR}/logs

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Conformer CTC Training (English)"
echo "========================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Exp dir:    ${EXP_DIR}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo ""

############################################
# VERIFY SETUP
############################################

# Check tokenizer
if [ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]; then
    echo "ERROR: tokenizer.model not found at ${TOKENIZER_DIR}/tokenizer.model"
    exit 1
fi
echo "✓ Tokenizer OK: ${TOKENIZER_DIR}"

# Check config
CONFIG_FILE="${CONFIG_PATH}/${CONFIG_NAME}"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config not found at ${CONFIG_FILE}"
    exit 1
fi
echo "✓ Config OK: ${CONFIG_FILE}"

# Optional: show GPU
echo ""
nvidia-smi
echo ""

############################################
# SNAPSHOT EXPERIMENT METADATA (optional but useful)
############################################
{
  echo "date: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "pwd: $(pwd)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "python: $(${PYTHON} --version 2>&1)"
  echo "which python: $(which python)"
  echo "pip freeze (first 30):"
  ${PYTHON} -m pip freeze | head -n 30
  echo ""
  echo "git (NeMo):"
  (cd "${NEMO_DIR}" && git rev-parse HEAD 2>/dev/null || true)
  echo "git (project):"
  (cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || true)
} > "${LOG_DIR}/run_info.txt"

cp -f "${CONFIG_FILE}" "${LOG_DIR}/config_snapshot.yaml"

############################################
# TRAINING
############################################
export HYDRA_FULL_ERROR=1

cd "${PROJECT_ROOT}"

${PYTHON} ${NEMO_DIR}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    exp_manager.exp_dir="${EXP_DIR}" \
    model.tokenizer.dir="${TOKENIZER_DIR}" \
    trainer.devices=4 \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "========================================"
echo "Training finished"
echo "Experiment: ${EXP_DIR}"
echo "Log:        ${LOG_DIR}/train.log"
echo "========================================"
