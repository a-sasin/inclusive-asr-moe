# Mixture-of-Experts Architectures for Inclusive and Robust Automatic Speech Recognition

MSc thesis code, configs, and results for studying sparse Mixture-of-Experts (MoE) architectures in ASR, with a focus on reducing the recognition gap between adult and child speech across multiple languages.

---

## Research Question

Can sparse MoE feed-forward layers in a FastConformer-CTC encoder reduce the adult–child WER gap when fine-tuned on child speech, and does expert routing learn to specialise by speaker age or language without explicit supervision?

---

## Architecture

Both the dense baseline and the MoE variant share the same FastConformer-CTC skeleton (17 Conformer blocks, d=512, 8 attention heads, 8× depthwise strided subsampling, CTC decoder). The only structural difference is that the second feed-forward sublayer in every encoder block is replaced by a sparse MoE-FFN.

| Configuration | Encoder | Experts | Active params | Total params |
|---|---|---|---|---|
| Dense FastConformer-CTC | `ConformerEncoder` | — | ~120 M | ~120 M |
| MoE FastConformer-CTC (EN) | `ConformerMoEEncoder` | 4 | ~120 M | ~227 M |
| MoE FastConformer-CTC (ML) | `ConformerMoEEncoder` | 8 | ~120 M | ~370 M |

**Router:** shared linear layer (top-2 gating, σ=0.005 noise during training). Optional load-balancing auxiliary loss λ=0.002.

`ConformerMoEEncoder` is implemented in a local NeMo fork — see [Prerequisites](#prerequisites).

---

## Repository Structure

```
inclusive-asr-moe/
├── configs/               Model configs (Hydra/NeMo YAML)
│   ├── english/           Adult EN fine-tuning (LibriSpeech)
│   ├── english_child/     Child EN fine-tuning (MyST) — dense + moe + moe_lb_on
│   ├── multilingual/      Adult multilingual fine-tuning (CV25 + LibriSpeech)
│   ├── multilingual_child/ Child multilingual fine-tuning — same variants
│   └── optimization/      OOMptimizer configs for bucket size tuning
├── train/                 Training shell scripts
│   ├── english/           Adult EN: train_fast-conformer.sh, train_fastconformer_moe.sh
│   ├── english_child/     Child EN: dense + moe + moe_lb_on variants
│   ├── multilingual/      Adult ML: dense + moe
│   ├── multilingual_child/ Child ML: dense + moe + moe_lb_on variants
│   └── optimization/      bucket_optimization.sh (OOMptimizer runs)
├── evaluation/            Evaluation Python scripts
│   ├── evaluate_english_experiment.py
│   └── evaluate_multilingual_experiment.py
├── analysis/
│   ├── routing/           Expert routing extraction + JSD / age-specificity analysis
│   └── wer_analysis/      Per-language WER breakdown notebooks (DE, PL)
├── tokenizers/            BPE tokenizer training scripts
│   ├── prepare_granary_tokenizer.sh          (EN, 4096 units)
│   └── prepare_granary_tokenizer_multilingual.sh  (ML, 16384 units)
└── data/                  Lhotse manifest YAML files (point to server-local audio)
    ├── english/           finetune_librispeech*.yaml, finetune_myst*.yaml
    └── multilingual/      train_cv.yaml, train_child.yaml, val_*.yaml
```

---

## Prerequisites

### 1. NeMo fork

`ConformerMoEEncoder` is not in public NeMo. The implementation lives at **https://github.com/a-sasin/NeMo**. Clone and install:

```bash
git clone https://github.com/a-sasin/NeMo /path/to/NeMo
conda env create -f environment.yml
conda activate nemo_asr
pip install -e /path/to/NeMo
```

### 2. Pretrained checkpoint

Download the NVIDIA NeMo pretrained FastConformer-CTC Large checkpoint and place it in `baseline_weights/`:

```bash
# From NVIDIA NGC or NeMo model hub:
# stt_en_fastconformer_ctc_large.nemo  →  baseline_weights/
```

### 3. Data

Datasets are not included in this repository. Each corpus must be obtained separately:

| Corpus | Language | Access |
|---|---|---|
| LibriSpeech | EN | Public — openslr.org/12 |
| CommonVoice 25.0 | NL / DE / PL | Public — commonvoice.mozilla.org |
| MyST | EN | Request — talkbank.org/access/CABank/MyST.html |
| JASMIN | NL | Request — cls.ru.nl/jasmin |
| KidsTalc | DE | Contact authors |
| PAVSig | PL | Contact authors |

Preprocessing instructions and expected manifest paths are in [preprocessing/README.md](preprocessing/README.md).

### 4. Environment variables

All training scripts read server-local absolute paths. Before running, update the `PROJECT_ROOT` and `NEMO_DIR` variables at the top of each script in `training/`.

---

## Reproducing the Experiments

### Step 1 — Train BPE tokenizers

```bash
bash tokenizers/prepare_granary_tokenizer.sh               # EN: 4096-unit
bash tokenizers/prepare_granary_tokenizer_multilingual.sh  # ML: 16384-unit
```

### Step 2 — Train models

`CUDA_VISIBLE_DEVICES` must be set explicitly. `PRETRAINED_NEMO` defaults are hardcoded in each script and can be overridden.

**English track (adult → child):**

```bash
# Adult LibriSpeech fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/english/train_fast-conformer.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/english/train_fastconformer_moe.sh

# Child fine-tuning
PRETRAINED_NEMO=/path/to/adult_dense.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/english_child/train_fast-conformer.sh

PRETRAINED_NEMO=/path/to/adult_moe.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/english_child/train_fastconformer_moe.sh

PRETRAINED_NEMO=/path/to/adult_moe.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/english_child/train_fastconformer_moe_load_balancing_on.sh
```

**Multilingual track (adult → child):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/multilingual/train_fast-conformer.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/multilingual/train_fastconformer_moe.sh

PRETRAINED_NEMO=/path/to/adult_dense_ml.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/multilingual_child/train_fast-conformer.sh

PRETRAINED_NEMO=/path/to/adult_moe_ml.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash train/multilingual_child/train_fastconformer_moe.sh
```

Checkpoints are written to `experiments/NEW/<track>/<model>/<name>/checkpoints/`.

### Step 3 — Evaluate

```bash
# English models (adult + child WER, Absolute Bias)
python evaluation/evaluate_english_experiment.py \
    --dense_pretrained /path/to/dense_ls.nemo \
    --dense_finetuned  /path/to/dense_myst.nemo \
    --moe_pretrained   /path/to/moe_ls.nemo \
    --moe_finetuned_lb_off /path/to/moe_myst_lb_off.nemo

# Multilingual + child speech (per-language WER)
python evaluation/evaluate_multilingual_experiment.py \
    --checkpoint /path/to/model.nemo \
    --output results/multilingual/my_model.json
```

---

## Key Results

### English Track

Adult = LibriSpeech test-clean; Child = MyST test partition. Abs. Bias = Child WER − Adult WER (pp).

| Model | Adult WER | Child WER | Abs. Bias |
|---|---|---|---|
| Dense — LibriSpeech only | 2.98 % | 26.99 % | +24.01 pp |
| MoE — LibriSpeech only | 2.56 % | 24.16 % | +21.60 pp |
| Dense — child fine-tuned | 2.94 % | 14.65 % | +11.71 pp |
| MoE — child fine-tuned (LB on) | 2.62 % | 14.08 % | +11.46 pp |
| **MoE — child fine-tuned (LB off)** | **2.66 %** | **14.05 %** | **+11.39 pp** |

### Multilingual Track

Adult test sets: CommonVoice 25.0 (NL/DE/PL), LibriSpeech test-clean (EN). Child test sets: JASMIN (NL), KidsTALC (DE), PAVSig† (PL), MyST (EN). Abs. Bias = Child WER − Adult WER (pp).

**Adult fine-tuning only**

| Model | Lang | Adult WER | Child WER | Abs. Bias |
|---|---|---|---|---|
| Dense | EN | 3.14 % | 26.67 % | +23.53 pp |
| | NL | 2.71 % | 81.38 % | +78.67 pp |
| | DE | 6.64 % | 67.72 % | +61.08 pp |
| | PL | 7.52 % | 118.32 %† | +110.80 pp |
| MoE | EN | 3.49 % | 29.08 % | +25.60 pp |
| | NL | 2.78 % | 89.59 % | +86.81 pp |
| | DE | 6.12 % | 69.50 % | +63.38 pp |
| | PL | 7.22 % | 128.77 %† | +121.55 pp |

**After child fine-tuning**

| Model | Lang | Adult WER | Child WER | Abs. Bias |
|---|---|---|---|---|
| Dense | EN | 3.54 % | 16.05 % | +12.51 pp |
| | NL | 3.51 % | 22.59 % | +19.08 pp |
| | DE | 8.00 % | 37.99 % | +29.99 pp |
| | PL | 8.36 % | 99.66 %† | +91.30 pp |
| **MoE (LB off)** | **EN** | **3.37 %** | **14.50 %** | **+11.13 pp** |
| | **NL** | **2.72 %** | **20.54 %** | **+17.82 pp** |
| | DE | 7.00 % | 48.86 % | +41.87 pp |
| | **PL** | **7.14 %** | **96.06 %†** | **+88.92 pp** |
| MoE (LB on) | EN | 3.44 % | 14.50 % | +11.06 pp |
| | NL | 2.72 % | 20.91 % | +18.19 pp |
| | DE | 7.10 % | 49.34 % | +42.24 pp |
| | PL | 7.20 % | 101.03 %† | +93.83 pp |

> † PAVSig contains children with sigmatism (a phonological disorder) and is not directly comparable to the other child corpora.
> German child results: MoE underperforms dense on child WER due to the small size and domain mismatch of KidsTALC (38 speakers, 8.8 h).

Full per-language WER and routing analysis outputs are in `results/multilingual/`.

---

## Expert Routing Analysis

Routing analyses (JSD, age-specificity, participation ratio) are in `analysis/routing/`. See [analysis/routing/README.md](analysis/routing/README.md) for how to reproduce the plots.

---

## Citation

```bibtex
@mastersthesis{sasin2026moe,
  author  = {Amelia Sasin},
  title   = {Mixture-of-Experts Architectures for Inclusive and Robust Automatic Speech Recognition},
  school  = {TU Delft},
  year    = {2026},
}
```
