#!/bin/bash
# Run NeMo OOMptimizer to find optimal bucket batch sizes.
# Adjust CUDA_VISIBLE_DEVICES and --memory-fraction as needed.

set -euo pipefail

PROJECT_ROOT=/lp-dev/amelia
NEMO_DIR=${PROJECT_ROOT}/NeMo

# Dense FastConformer (English)
CUDA_VISIBLE_DEVICES=0 python ${NEMO_DIR}/scripts/speech_recognition/oomptimizer.py \
  --config-path ${PROJECT_ROOT}/inclusive-asr-moe/configs/optimization/fastconformer_oom.yaml \
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[0.5,6.24,6.96,7.6,8.16,8.64,9.2,9.68,10.16,10.64,11.12,11.68,12.24,12.8,13.36,14.0,14.64,15.36,16.08,16.88,17.76,18.64,19.68,20.72,21.92,23.28,24.78,26.4,28.32,31.76]' \
  --memory-fraction 0.80

# MoE FastConformer (English)
CUDA_VISIBLE_DEVICES=0 python ${NEMO_DIR}/scripts/speech_recognition/oomptimizer.py \
  --config-path ${PROJECT_ROOT}/inclusive-asr-moe/configs/optimization/fastconformer_moe_oom.yaml \
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[0.5,6.255,8.55,10.135,11.19,11.89,12.365,12.75,13.065,13.33,13.555,13.755,13.94,14.11,14.265,14.41,14.55,14.685,14.815,14.94,15.065,15.185,15.305,15.42,15.54,15.655,15.775,15.9,16.07,16.355]' \
  --memory-fraction 0.80
