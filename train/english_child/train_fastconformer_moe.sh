set -euo pipefail

# ============================================================================
# MoE FastConformer Training (NEW config)
# Fair-comparison companion to train_fast-conformer.sh
# ============================================================================

############################################
# ENVIRONMENT
############################################
source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate nemo_asr

PYTHON=/home/nvidia/miniconda3/envs/nemo_asr/bin/python3

PROJECT_ROOT=/lp-dev/amelia
NEMO_DIR=${PROJECT_ROOT}/NeMo
CONFIG_PATH=${PROJECT_ROOT}/inclusive-asr-moe/configs/english_child
CONFIG_NAME=moe-fast-conformer_ctc_bpe.yaml
TOKENIZER_DIR=${PROJECT_ROOT}/inclusive-asr-moe/tokenizers/granary_en_bpe_4096/tokenizer_spe_bpe_v4096


PRETRAINED_NEMO=${PRETRAINED_NEMO:-"/lp-dev/amelia/inclusive-asr-moe/experiments/NEW/english/moe/moe_fastconformer_librispeech_2026-04-16_18-03-09/2026-04-16_18-03-22/checkpoints/moe_fastconformer_librispeech_2026-04-16_18-03-09.nemo"}

# User controls GPU routing from shell, e.g.:
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/NEW/train_fastconformer_moe.sh
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

############################################
# EXPERIMENT NAMING
############################################
RUN_TAG=${RUN_TAG:-$(date +%Y-%m-%d_%H-%M-%S)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"moe_fastconformer_child_myst_${RUN_TAG}"}
EXPERIMENT_ROOT=${PROJECT_ROOT}/inclusive-asr-moe/experiments/NEW/english_child/moe
EXP_DIR=${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}
LOG_DIR=${EXP_DIR}/logs

mkdir -p "${LOG_DIR}"

echo "========================================"
echo "MoE FastConformer Child Fine-tuning"
echo "========================================"
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "Pretrained:  ${PRETRAINED_NEMO}"
echo "Exp dir:     ${EXP_DIR}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Config:      ${CONFIG_PATH}/${CONFIG_NAME}"
echo "NOTE:        load_balance_loss_weight = 0.0 (disabled for strict fairness)"
echo ""

############################################
# VERIFY SETUP
############################################
if [ ! -f "${PRETRAINED_NEMO}" ]; then
    echo "ERROR: Pretrained MoE .nemo not found at ${PRETRAINED_NEMO}"
    echo "       Set PRETRAINED_NEMO=/path/to/your_moe.nemo"
    exit 1
fi
echo "✓ Pretrained MoE model OK"

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
# INSTALL LOCAL NeMo FORK (needed for ConformerMoEEncoder)
############################################
echo "Installing local NeMo fork (required for MoE encoder)..."
${PYTHON} -m pip install -e ${NEMO_DIR} 2>&1 | tail -3

############################################
# SNAPSHOT METADATA
############################################
{
    echo "=== Fine-tune: MoE FastConformer → Child MyST ==="
  echo "date: $(date -Is)"
  echo "hostname: $(hostname)"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
  echo "pretrained_nemo: ${PRETRAINED_NEMO}"
  echo "config: ${CONFIG_FILE}"
    echo "load_balance_loss_weight: 0.0 (disabled for strict fairness)"
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
    exp_manager.exp_dir="${EXPERIMENT_ROOT}" \
    exp_manager.name="${EXPERIMENT_NAME}" \
    model.tokenizer.dir="${TOKENIZER_DIR}" \
    trainer.use_distributed_sampler=false \
    trainer.devices=${NUM_GPUS} \
    trainer.num_nodes=1 \
    trainer.strategy=ddp \
    model.encoder.load_balance_loss_weight=0.0 \
    exp_manager.wandb_logger_kwargs.name="${EXPERIMENT_NAME}" \
    +init_from_nemo_model="${PRETRAINED_NEMO}" \
    2>&1 | tee "${LOG_DIR}/train.log"

echo ""
echo "========================================"
echo "MoE Fine-tuning finished"
echo "Experiment: ${EXP_DIR}"
echo "Log:        ${LOG_DIR}/train.log"
echo "========================================"
