cd /lp-dev/amelia/NeMo


python scripts/speech_recognition/estimate_duration_bins.py -b 30 /home/nvidia/amelia/inclusive-asr-moe/scripts/experiments/train/train_granary_myst_en.yaml



export CUDA_VISIBLE_DEVICES=4


python scripts/speech_recognition/oomptimizer.py \
  --config-path /home/nvidia/amelia/inclusive-asr-moe/scripts/experiments/train/conformer_ctc_bpe_oom.yaml \
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[3.42,5.24,6.3,7.12,7.92,8.64,9.31,10.0,10.72,11.52,12.28,13.12,14.0,14.98,16.0,17.12,18.4,19.7,21.16,22.68,24.32,26.0,27.84,29.82,30.0,32.0,35.58,37.38,38.84]' \
  --memory-fraction 0.90
