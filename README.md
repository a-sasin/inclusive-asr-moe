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
├── configs/            Model configs (Hydra/NeMo YAML)
│   ├── english/        English track: adult_dense, adult_moe, child_dense, child_moe, child_moe_lb_on
│   ├── multilingual/   Multilingual track: same set of variants
│   └── optimization/   OOMptimizer configs for bucket size tuning
├── training/           Training shell scripts
│   ├── english/        stage2_adult_*.sh  →  stage3_child_*.sh
│   └── multilingual/   stage1_adult_*.sh  →  stage3_child_*.sh
├── evaluation/         Evaluation Python scripts + run_all_evaluations.sh
├── preprocessing/      Per-corpus data preparation scripts
│   ├── myst/           MyST English child speech
│   ├── common_voice/   CommonVoice (NL, DE, PL)
│   ├── jasmin/         JASMIN Dutch child speech
│   ├── kidstalc/       KidsTalc German child speech
│   └── pavsig/         PAVSig Polish pathological child speech
├── analysis/
│   ├── routing/        Expert routing extraction + JSD / age-specificity analysis
│   └── corpus/         Corpus statistics scripts
├── weights_prep/       Checkpoint preparation utilities (ff-reset, dense→MoE conversion)
├── tokenizer/          BPE tokenizer training scripts
├── data/               Lhotse manifest YAML files (point to server-local audio)
├── results/            Evaluation JSON outputs (tracked in git)
│   ├── english/        English track WER results
│   └── multilingual/   Multilingual track WER results
└── archive/            Legacy training scripts (superseded, kept for reference)
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

### Step 0 — Prepare checkpoints

```bash
# Dense → ff-reset (reset FF weights before training)
python weights_prep/prepare_checkpoint.py --mode ff-reset \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large.nemo \
    --output-path baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo

# Dense → MoE init (English, 4 experts)
python weights_prep/prepare_checkpoint.py --mode to-moe \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo \
    --moe-config configs/english/adult_moe.yaml \
    --output-path baseline_weights/stt_en_fastconformer_ctc_large_to_moe_init.nemo

# Dense → MoE init (Multilingual, 8 experts)
python weights_prep/prepare_checkpoint_multilingual.py --mode to-moe \
    --model-path baseline_weights/stt_en_fastconformer_ctc_large_ff_reset.nemo \
    --moe-config configs/multilingual/adult_moe.yaml \
    --output-path baseline_weights/stt_en_fastconformer_ctc_large_to_moe_init_ml.nemo
```

### Step 1 — Train BPE tokenizers

```bash
bash tokenizer/prepare_english_tokenizer.sh       # 4096-unit EN tokenizer
bash tokenizer/prepare_multilingual_tokenizer.sh  # 16384-unit ML tokenizer
```

### Step 2 — Train models

**English track (Stage 2 → Stage 3):**

```bash
# Adult LibriSpeech fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/english/stage2_adult_dense.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/english/stage2_adult_moe.sh

# Child fine-tuning (set PRETRAINED_NEMO to Stage 2 checkpoint)
PRETRAINED_NEMO=/path/to/stage2_dense.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/english/stage3_child_dense.sh

PRETRAINED_NEMO=/path/to/stage2_moe.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/english/stage3_child_moe.sh
```

**Multilingual track (Stage 1 → Stage 3):**

```bash
# Adult CommonVoice + LibriSpeech
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/multilingual/stage1_adult_dense.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/multilingual/stage1_adult_moe.sh

# Multilingual child fine-tuning
PRETRAINED_NEMO=/path/to/stage1_dense.nemo \
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/multilingual/stage3_child_dense.sh
```

Checkpoints are written to `experiments/NEW/<track>/<model>/<name>/checkpoints/`.

### Step 3 — Evaluate

```bash
# Single model
python evaluation/evaluate_multilingual.py \
    --checkpoint /path/to/model.nemo \
    --output results/multilingual/my_model.json

# All thesis models (~6–8 h on 1× A100)
bash evaluation/run_all_evaluations.sh
```

### Step 4 — Compile results table

```bash
python evaluation/compile_results.py \
    --results_dir results/multilingual \
    --output results/multilingual/wer_table.txt
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
