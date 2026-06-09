# Tokenizers

This directory is gitignored. The trained tokenizer files must be reproduced locally or obtained from the training server.

## Reproducing

### English BPE-4096

```bash
bash tokenizer/prepare_english_tokenizer.sh
```

Trains a 4096-unit SentencePiece BPE tokenizer on Granary English text + MyST transcriptions with lower-casing. Output: `tokenizers/granary_en_bpe_4096/tokenizer_spe_bpe_v4096/`.

### Multilingual BPE-16384

```bash
bash tokenizer/prepare_multilingual_tokenizer.sh
```

Trains a 16384-unit SentencePiece BPE tokenizer on CommonVoice (NL/DE/PL) + LibriSpeech (EN) + all four child speech corpora without lower-casing. Output: `tokenizers/granary_multilingual_bpe_16384/tokenizer_spe_bpe_v16384/`.

Both scripts call `NeMo/scripts/tokenizers/process_asr_text_tokenizer.py`. Update the `NEMO_DIR` and manifest paths inside the scripts before running.

## Expected Paths Referenced by Configs

```
tokenizers/
├── granary_en_bpe_4096/
│   └── tokenizer_spe_bpe_v4096/
│       ├── tokenizer.model
│       └── vocab.txt
└── granary_multilingual_bpe_16384/
    └── tokenizer_spe_bpe_v16384/
        ├── tokenizer.model
        └── vocab.txt
```
