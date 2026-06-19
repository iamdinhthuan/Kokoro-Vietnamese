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

## Python API

```python
import soundfile as sf

from kokoro_vietnamese import KokoroVietnamese

tts = KokoroVietnamese(device="cuda", voice="diem_trinh")

audio, phonemes = tts.synthesize(
    "Giữa một buổi chiều yên tĩnh, cô ấy kể lại câu chuyện bằng một giọng nói ấm áp và chậm rãi."
)

sf.write("outputs/sample.wav", audio, 24000)
print(phonemes)
```

Use another voice:

```python
tts = KokoroVietnamese(device="cuda", voice="mai_linh")
audio, phonemes = tts.synthesize("Hôm nay trời trong xanh, gió thổi nhẹ qua hiên nhà.")
```

Use local files instead of downloading from Hugging Face:

```python
tts = KokoroVietnamese(
    device="cuda",
    model_path="kokoro_vi.pth",
    voicepack_path="voicepacks/diem_trinh.pt",
    config_path="config.json",
)
```

## CLI

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
