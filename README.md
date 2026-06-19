# Kokoro Vietnamese

Inference-only project for Vietnamese Kokoro TTS.

This repository intentionally does not include finetuning code, datasets,
training configs, checkpoints, or StyleTTS2. Vietnamese phonemes are generated
with the PyPI package `vig2p`.

## Install

```bash
pip install -e .
```

For CUDA, install the PyTorch build that matches your machine first.

## Model Files

The CLI downloads these files from
[`contextboxai/Kokoro-Vietnamese`](https://huggingface.co/contextboxai/Kokoro-Vietnamese)
when local paths are not provided:

- `kokoro_vi.pth`
- `kokoro_vi_voicepack.pt`
- `config.json`

Extra voicepacks:

- `voicepacks/diem_trinh.pt`
- `voicepacks/mai_linh.pt`
- `voicepacks/mai_loan.pt`
- `voicepacks/storyvert.pt`

## Usage

```bash
kokoro-vietnamese \
  --text "Tường nhà khách." \
  --output outputs/sample.wav \
  --device cuda \
  --print-phonemes
```

Or run as a module:

```bash
python -m kokoro_vietnamese \
  --text "Xin chào, hôm nay tôi đang kiểm tra giọng đọc tiếng Việt." \
  --output outputs/sample.wav
```

Use a different voicepack:

```bash
kokoro-vietnamese \
  --text "Xin chào." \
  --voicepack voicepacks/mai_linh.pt \
  --output outputs/mai_linh.wav
```

## Python API

```python
from kokoro_vietnamese import KokoroVietnamese

tts = KokoroVietnamese(device="cuda")
audio, phonemes = tts.synthesize("Tường nhà khách.")
```
