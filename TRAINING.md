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
  --output-dataset dataset_vi \
  --output-training training/vi \
  --min-duration 0.6 \
  --max-duration 30 \
  --clean
```

The command uses every valid metadata row by default. Only pass
`--target-duration-hours` when you intentionally want to cap a dataset.

Run the frontend audit before spending GPU time:

```bash
python scripts/audit_vietnamese_frontend.py \
  --metadata /path/to/source_dataset/metadata.csv \
  --fail-on-minimal-pairs \
  --fail-on-adapter-collisions
```

## Base Checkpoint

Training configs point to:

```text
training/kokoro_base.pth
```

This file is not committed. If it is missing, `train_first.py` downloads the
Kokoro base checkpoint from `hexgrad/Kokoro-82M` automatically before loading
the pretrained weights. To use another base checkpoint, set `pretrained_model`
to your local path or set `pretrained_model_url` in the config.

## Train

Use the provided config closest to your run, or copy one under `configs/` and
update its `data_params` paths to match your preprocessed dataset.

Stage 1:

```bash
cd StyleTTS2
accelerate launch train_first.py \
  --config_path ../configs/your_stage1_config.yml
```

Stage 2:

```bash
cd StyleTTS2
accelerate launch train_second.py \
  --config_path ../configs/your_stage2_config.yml
```

Checkpoints are written under `StyleTTS2/logs/` and are ignored by git.

## Test Checkpoints

Point these paths at the log directory, cache directory, audio directory, and
speaker registry from your own run.

```bash
python scripts/gradio_vietnamese_checkpoint.py \
  --log-dir StyleTTS2/logs/your-run \
  --cache-dir gradio_cache/your-run \
  --audio-dir dataset_vi/audio \
  --speakers-path training/vi/speakers.json \
  --device cuda \
  --share
```

## Verification

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
