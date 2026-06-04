cd /lp-dev/amelia/NeMo


python scripts/speech_recognition/estimate_duration_bins.py -b 30 /home/nvidia/amelia/inclusive-asr-moe/scripts/experiments/train/train_granary_myst_en.yaml

python scripts/speech_recognition/estimate_duration_bins.py -b 30 /lp-dev/amelia/inclusive-asr-moe/data/english/finetune_librispeech.yaml

export CUDA_VISIBLE_DEVICES=4


python scripts/speech_recognition/oomptimizer.py \
  --config-path /home/nvidia/amelia/inclusive-asr-moe/scripts/experiments/train/conformer_ctc_bpe_oom.yaml \
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[3.42,5.24,6.3,7.12,7.92,8.64,9.31,10.0,10.72,11.52,12.28,13.12,14.0,14.98,16.0,17.12,18.4,19.7,21.16,22.68,24.32,26.0,27.84,29.82,30.0,32.0,35.58,37.38,38.84]' \
  --memory-fraction 0.90


python /lp-dev/amelia/NeMo/scripts/speech_recognition/oomptimizer.py \
  --config-path /lp-dev/amelia/inclusive-asr-moe/configs/optimization/conformer_ctc_bpe_oom.yaml\
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[0.5,6.24,6.96,7.6,8.16,8.64,9.2,9.68,10.16,10.64,11.12,11.68,12.24,12.8,13.36,14.0,14.64,15.36,16.08,16.88,17.76,18.64,19.68,20.72,21.92,23.28,24.78,26.4,28.32,31.76]' \
  --memory-fraction 0.80





CUDA_VISIBLE_DEVICES=0 python /lp-dev/amelia/NeMo/scripts/speech_recognition/oomptimizer.py \
  --config-path /lp-dev/amelia/inclusive-asr-moe/configs/optimization/fastconformer_oom.yaml\
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[10.96,12.915,13.805,14.41,14.91,15.36,15.805]' \
  --memory-fraction 0.80


 CUDA_VISIBLE_DEVICES=2  python /lp-dev/amelia/NeMo/scripts/speech_recognition/oomptimizer.py \
  --config-path /lp-dev/amelia/inclusive-asr-moe/configs/optimization/fastconformer_moe_oom.yaml\
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[0.5,6.255,8.55,10.135,11.19,11.89,12.365,12.75,13.065,13.33,13.555,13.755,13.94,14.11,14.265,14.41,14.55,14.685,14.815,14.94,15.065,15.185,15.305,15.42,15.54,15.655,15.775,15.9,16.07,16.355]' \
  --memory-fraction 0.80


  /lp-dev/amelia/inclusive-asr-moe/configs/optimization/fastconformer_moe_oom.yaml

 CUDA_VISIBLE_DEVICES=2  python /lp-dev/amelia/NeMo/scripts/speech_recognition/oomptimizer.py \
  --config-path  /lp-dev/amelia/inclusive-asr-moe/configs/optimization/conformer_ctc_bpe_oom.yaml\
  --module-name nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE \
  --buckets '[0.5,6.255,8.55,10.135,11.19,11.89,12.365,12.75,13.065,13.33,13.555,13.755,13.94,14.11,14.265,14.41,14.55,14.685,14.815,14.94,15.065,15.185,15.305,15.42,15.54,15.655,15.775,15.9,16.07,16.355]' \
  --memory-fraction 0.80



 