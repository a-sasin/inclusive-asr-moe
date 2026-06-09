# Preprocessing

Per-corpus scripts that convert raw dataset downloads into NeMo-format JSONL manifests. Each manifest entry has three fields: `audio_filepath`, `duration` (seconds), `text` (orthographic transcription).

All preprocessing scripts write to paths under `/lp-dev/amelia/data/` by default. Update the output path constants at the top of each script for your environment.

## Corpora

### MyST — English child speech (`myst/`)

```bash
bash preprocessing/myst/run.sh
```

Pipeline: Whisper text normalisation → utterance filtering (non-speech, clipping, Whisper-oracle WER > 50%) → audio concatenation to reduce short clips. Uses Ray for distributed Whisper inference across multiple GPUs.

**Input:** Raw MyST corpus directory  
**Output:** `train.json`, `val.json`, `test.json`  
**Requirements:** `ray`, `whisper-normalizer`, `openai-whisper`, `inflect`, `soundfile`

---

### CommonVoice — NL / DE / PL (`common_voice/`)

```bash
python preprocessing/common_voice/prepare.py --lang nl
python preprocessing/common_voice/prepare.py --lang de
python preprocessing/common_voice/prepare.py --lang pl
```

Converts CommonVoice TSV splits to NeMo manifests with speaker-independent 80/10/10 splitting. Audio is resampled to 16 kHz mono WAV.

**Input:** CommonVoice 25.0 download directory  
**Output:** `train.json`, `val.json`, `test.json` per language  
**Requirements:** `datasets` or direct TSV reading, `librosa`

---

### JASMIN — Dutch child speech (`jasmin/`)

```bash
python preprocessing/jasmin/prepare.py
```

Filters to Group 1 (native Dutch children, ages 6–13, Netherlands dialect region). Applies transcription cleaning (truncation markers, non-speech tokens).

**Input:** JASMIN corpus directory with `recordings.txt`  
**Output:** Filtered manifests preserving the original train/val/test split

---

### KidsTalc — German child speech (`kidstalc/`)

```bash
python preprocessing/kidstalc/prepare.py
```

Minimal filtering (empty transcriptions, < 0.5 s). Preserves German capitalisation.

**Input:** KidsTalc corpus directory  
**Output:** `train.json`, `val.json`, `test.json`

---

### PAVSig — Polish pathological child speech (`pavsig/`)

```bash
# 1. Fix multi-channel audio to mono
python preprocessing/pavsig/fix_audio.py

# 2. Convert IPA transcriptions to orthographic Polish
python preprocessing/pavsig/orthographic.py

# 3. Generate word- and speaker-disjoint train/val/test split
python preprocessing/pavsig/generate_splits.py
```

**Special note:** PAVSig targets sigmatism (lisp). WER on this corpus is not directly comparable to other child corpora. See thesis Section 3.4 for details.

**Input:** PAVSig corpus with audio files and IPA manifests  
**Output:** `train.json`, `val.json`, `test.json`

---

## Shared Data Format

All manifests follow the NeMo JSONL format:
```json
{"audio_filepath": "/absolute/path/to/audio.wav", "duration": 3.14, "text": "the transcription"}
```

Audio is 16 kHz mono. Duration is in seconds as a float.
