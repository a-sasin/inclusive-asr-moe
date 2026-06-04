#!/bin/bash
# ============================================================================
# Dense FastConformer Training (NEW config)
# Fair-comparison companion to train_fastconformer_moe.sh
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
CONFIG_PATH=${PROJECT_ROOT}/inclusive-asr-moe/configs/multilingual
CONFIG_NAME=fast-conformer_ctc_bpe.yaml

echo "→ Installing NeMo from ${NEMO_DIR} (editable)..."
${PYTHON} -m pip install -e "${NEMO_DIR}" --quiet
echo "✓ NeMo editable install OK"


TOKENIZER_DIR=${PROJECT_ROOT}/inclusive-asr-moe/tokenizers/granary_multilingual_bpe_16384/tokenizer_spe_bpe_v16384

PRETRAINED_NEMO=${PRETRAINED_NEMO:-"/lp-dev/amelia/inclusive-asr-moe/baseline_weights/stt_ml_fastconformer_ctc_large_to_fast_from_config_ffreset.nemo"}

# User controls GPU routing from shell, e.g.:
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/NEW/train_fast-conformer.sh
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES is not set."
    echo "Set it explicitly, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3"
    exit 1
fi

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')
if [[ "${NUM_GPUS}" -lt 1 ]]; then
    echo "ERROR: Could not infer GPU count from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exit 1
fi


TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"dense_fastconformer_multilingual_${TIMESTAMP}"}
EXPERIMENT_ROOT=${PROJECT_ROOT}/inclusive-asr-moe/experiments/NEW/multilingual/dense
EXP_DIR=${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}
LOG_DIR=${EXP_DIR}/logs

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Dense FastConformer Training (NEW multilingual)"
echo "========================================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Exp dir:    ${EXP_DIR}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs:   ${NUM_GPUS}"
echo "Config:     ${CONFIG_PATH}/${CONFIG_NAME}"
echo "Init nemo:  ${PRETRAINED_NEMO}"
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

# Check pretrained checkpoint
if [ ! -f "${PRETRAINED_NEMO}" ]; then
    echo "ERROR: Pretrained .nemo not found at ${PRETRAINED_NEMO}"
    exit 1
fi
echo "✓ Pretrained model OK"

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_ROOT}"

${PYTHON} ${NEMO_DIR}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path="${CONFIG_PATH}" \
    --config-name="${CONFIG_NAME}" \
    exp_manager.exp_dir="${EXPERIMENT_ROOT}" \
    exp_manager.name="${EXPERIMENT_NAME}" \
    model.tokenizer.dir="${TOKENIZER_DIR}" \
    trainer.devices=${NUM_GPUS} \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    exp_manager.wandb_logger_kwargs.name="${EXPERIMENT_NAME}" \
    +init_from_nemo_model="${PRETRAINED_NEMO}" \
    2>&1 | tee "${LOG_DIR}/train.log"


echo ""
echo "========================================"
echo "Training finished"
echo "Experiment: ${EXP_DIR}"
echo "Log:        ${LOG_DIR}/train.log"
echo "========================================"
