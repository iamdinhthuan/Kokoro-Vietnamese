# Training Kokoro Vietnamese

Vietnamese training uses the same frontend as inference: `vig2p` converts text
to Kokoro-compatible phonemes before StyleTTS2 training.

## Install

```bash
git clone https://github.com/iamdinhthuan/Kokoro-Vietnamese.git
cd Kokoro-Vietnamese
pip install -e ".[training]"
```

Install the PyTorch CUDA build that matches your GPU before training if needed.

System packages:

```bash
sudo apt-get install ffmpeg libsndfile1
```

## What Is Included

- `StyleTTS2/`: patched training code.
- `kokoro/`: patched Kokoro runtime used by training previews.
- `configs/`: Vietnamese stage 1/stage 2 configs.
- `scripts/prepare_vietnamese_dataset.py`: dataset preprocessing.
- `scripts/audit_vietnamese_frontend.py`: phoneme/frontend audit.
- `scripts/extract_voicepack.py`: voicepack extraction.
- `scripts/gradio_vietnamese_checkpoint.py`: local checkpoint testing.
- `scripts/modal_vietnamese_*.py`: Modal training launchers.

Datasets, train/val lists, checkpoints, voicepacks, logs, caches, and generated
audio are intentionally ignored.

## Dataset Format

Single speaker:

```text
audio/000001.wav|Nội dung transcript tiếng Việt
audio/000002.wav|Câu tiếp theo
```

Multi-speaker:

```text
audio/000001.wav|Câu của speaker A|speaker_a
audio/000002.wav|Câu của speaker B|speaker_b
```

Audio should be clean mono WAV, ideally 24 kHz, with transcripts matching the
audio exactly.

## Preprocess

```bash
python scripts/prepare_vietnamese_dataset.py \
  --source-root /path/to/source_dataset \
  --metadata /path/to/source_dataset/metadata.csv \
  --target-duration-hours 30 \
  --output-dataset dataset_vi_30h \
  --output-training training/vi_30h \
  --min-duration 0.6 \
  --max-duration 30 \
  --clean
```

Run the frontend audit before spending GPU time:

```bash
python scripts/audit_vietnamese_frontend.py \
  --metadata /path/to/source_dataset/metadata.csv \
  --fail-on-minimal-pairs \
  --fail-on-adapter-collisions
```

## Base Checkpoint

Put the Kokoro base checkpoint at:

```text
training/kokoro_base.pth
```

This file is not committed.

## Train

Stage 1:

```bash
cd StyleTTS2
accelerate launch train_first.py \
  --config_path ../configs/config_vietnamese_30h_stage1.yml
```

Stage 2:

```bash
cd StyleTTS2
accelerate launch train_second.py \
  --config_path ../configs/config_vietnamese_30h_stage2.yml
```

Multi-speaker configs:

```bash
cd StyleTTS2
accelerate launch train_first.py \
  --config_path ../configs/config_vietnamese_30h_multispeaker_stage1.yml
accelerate launch train_second.py \
  --config_path ../configs/config_vietnamese_30h_multispeaker_stage2.yml
```

Checkpoints are written under `StyleTTS2/logs/` and are ignored by git.

## Test Checkpoints

```bash
python scripts/gradio_vietnamese_checkpoint.py \
  --log-dir StyleTTS2/logs/kokoro-vi-30h-ms \
  --cache-dir gradio_cache/vi_30h_ms \
  --audio-dir dataset_vi_30h/audio \
  --speakers-path training/vi_30h/speakers.json \
  --device cuda \
  --share
```

## Modal

The Modal launchers expect local preprocessing to be finished first. They upload
preprocessed metadata/lists and train on Modal volumes without uploading local
datasets into the image.

```bash
modal run scripts/modal_vietnamese_larvoice_train.py \
  --stage both \
  --epochs-stage1 10 \
  --epochs-stage2 10 \
  --batch-size-stage1 20 \
  --batch-size-stage2 16 \
  --clean-log
```

Edit the dataset paths and volume names inside the Modal scripts for your run.

## Verification

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
