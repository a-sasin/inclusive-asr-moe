#!/usr/bin/env bash
# ==========================================================================
# PATHS — update these for your environment
# ==========================================================================
WORKSPACE=/lp-dev/amelia
NEMO_DIR=${WORKSPACE}/NeMo
PROJECT_DIR=${WORKSPACE}/inclusive-asr-moe

python3 ${NEMO_DIR}/scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest=/data/cv/nemo/train_merged.json \
  --data_root="${PROJECT_DIR}/tokenizers/granary_multilingual_bpe_16384" \
  --vocab_size=16384 \
  --tokenizer="spe" \
  --spe_type="bpe" \
  --spe_character_coverage=1.0 \
  --spe_sample_size=-1 \
  --spe_train_extremely_large_corpus \
  --no_lower_case \
  --log
