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

Named voices:

| Voice | HF file |
| --- | --- |
| `diem_trinh` | `voicepacks/diem_trinh.pt` |
| `hung_thinh` | `voicepacks/hung_thinh.pt` |
| `mai_linh` | `voicepacks/mai_linh.pt` |
| `mai_loan` | `voicepacks/mai_loan.pt` |
| `manh_dung` | `voicepacks/manh_dung.pt` |
| `my_yen` | `voicepacks/my_yen.pt` |
| `ngoc_huyen` | `voicepacks/ngoc_huyen.pt` |
| `phat_tai` | `voicepacks/phat_tai.pt` |
| `thanh_dat` | `voicepacks/thanh_dat.pt` |
| `thuc_trinh` | `voicepacks/thuc_trinh.pt` |
| `tuan_ngoc` | `voicepacks/tuan_ngoc.pt` |
| `storyvert` | `voicepacks/storyvert.pt` |
| `duc_an` | `voicepacks/duc_an.pt` |
| `duc_duy` | `voicepacks/duc_duy.pt` |

## Usage

```bash
kokoro-vietnamese \
  --text "Giữa một buổi chiều yên tĩnh, cô ấy kể lại câu chuyện bằng một giọng nói ấm áp và chậm rãi." \
  --output outputs/sample.wav \
  --voice diem_trinh \
  --device cuda
```

```bash
kokoro-vietnamese --list-voices
```

```bash
kokoro-vietnamese \
  --batch-file texts.txt \
  --voice diem_trinh \
  --output-dir outputs/batch
```

## Python API

```python
from kokoro_vietnamese import KokoroVietnamese

tts = KokoroVietnamese(device="cuda")
audio, phonemes = tts.synthesize("Giữa một buổi chiều yên tĩnh, cô ấy kể lại câu chuyện bằng một giọng nói ấm áp và chậm rãi.")
```
