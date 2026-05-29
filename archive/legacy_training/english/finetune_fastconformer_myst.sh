#!/bin/bash
# ============================================================================
# Fine-tune Dense FastConformer-CTC-BPE on MyST Child Speech
# ============================================================================
#
# Prerequisites:
#   - Pretrained Dense FastConformer .nemo checkpoint
#   - MyST data at /home/nvidia/amelia/data/myst/
#
# Usage:
#   ./finetune_fastconformer_myst.sh
#
#   # Override pretrained checkpoint:
#   PRETRAINED_NEMO=/path/to/other.nemo ./finetune_fastconformer_myst.sh
#
#   # Override GPUs:
#   CUDA_VISIBLE_DEVICES=0,1 ./finetune_fastconformer_myst.sh
#
# ============================================================================

set -euo pipefail

############################################
# ENVIRONMENT
############################################
source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate nemo_asr

PYTHON=/home/nvidia/miniconda3/envs/nemo_asr/bin/python3

PROJECT_ROOT=/lp-dev/amelia
NEMO_DIR=${PROJECT_ROOT}/NeMo
CONFIG_PATH=${PROJECT_ROOT}/inclusive-asr-moe/configs/finetune_child
CONFIG_NAME=fastconformer_finetune_myst.yaml
TOKENIZER_DIR=${PROJECT_ROOT}/inclusive-asr-moe/tokenizers/granary_en_bpe_4096/tokenizer_spe_bpe_v4096

# ---- Pretrained model (override with env var) ----
PRETRAINED_NEMO=${PRETRAINED_NEMO:-"/lp-dev/amelia/inclusive-asr-moe/experiments/english/finetune_librispeech/dense_finetune_librispeech_2026-03-25_13-52-12/FastConformer-CTC-BPE-Finetune-LibriSpeech-Dense/2026-03-25_13-52-26/checkpoints/FastConformer-CTC-BPE-Finetune-LibriSpeech-Dense.nemo"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6,7}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)

############################################
# EXPERIMENT NAMING
############################################
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EXPERIMENT_NAME="fastconformer_dense_finetune_librispeech_myst_${TIMESTAMP}"
EXP_DIR=${PROJECT_ROOT}/inclusive-asr-moe/experiments/english/finetune/${EXPERIMENT_NAME}
LOG_DIR=${EXP_DIR}/logs

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Dense FastConformer → MyST Child Fine-tuning"
echo "========================================"
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Pretrained:  ${PRETRAINED_NEMO}"
echo "Exp dir:     ${EXP_DIR}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo ""

############################################
# VERIFY SETUP
############################################
if [ ! -f "${PRETRAINED_NEMO}" ]; then
    echo "ERROR: Pretrained .nemo not found at ${PRETRAINED_NEMO}"
    echo "       Set PRETRAINED_NEMO=/path/to/your.nemo"
    exit 1
fi
echo "✓ Pretrained model OK"

if [ ! -f "${TOKENIZER_DIR}/tokenizer.model" ]; then
    echo "ERROR: tokenizer.model not found at ${TOKENIZER_DIR}"
    exit 1
fi
echo "✓ Tokenizer OK"

CONFIG_FILE="${CONFIG_PATH}/${CONFIG_NAME}"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config not found at ${CONFIG_FILE}"
    exit 1
fi
echo "✓ Config OK"

echo ""
nvidia-smi
echo ""

############################################
# INSTALL LOCAL NeMo FORK
############################################
echo "Installing local NeMo fork..."
${PYTHON} -m pip install -e ${NEMO_DIR} 2>&1 | tail -3

############################################
# SNAPSHOT METADATA
############################################
{
    echo "=== Fine-tune: Dense FastConformer → MyST Child Speech ==="
  echo "date: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "pretrained_nemo: ${PRETRAINED_NEMO}"
  echo "config: ${CONFIG_FILE}"
  echo "python: $(${PYTHON} --version 2>&1)"
  echo ""
  echo "git (NeMo): $(cd "${NEMO_DIR}" && git rev-parse HEAD 2>/dev/null || echo 'n/a')"
  echo "git (project): $(cd "${PROJECT_ROOT}" && git rev-parse HEAD 2>/dev/null || echo 'n/a')"
} > "${LOG_DIR}/run_info.txt"

cp -f "${CONFIG_FILE}" "${LOG_DIR}/config_snapshot.yaml"

############################################
# FINE-TUNE
############################################
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
    +init_from_nemo_model="${PRETRAINED_NEMO}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "========================================"
echo "Dense FastConformer fine-tuning finished"
echo "Experiment: ${EXP_DIR}"
echo "Log:        ${LOG_DIR}/train.log"
echo "========================================"
