# Kokoro Vietnamese

Vietnamese G2P is handled by [`vig2p`](https://pypi.org/project/vig2p/).

This repository includes both the lightweight inference package and the
Vietnamese fine-tuning recipe. Training instructions are in
[`TRAINING.md`](TRAINING.md).

## Install

```bash
git clone https://github.com/iamdinhthuan/Kokoro-Vietnamese.git
cd Kokoro-Vietnamese
pip install -e .
```

For CUDA, install the PyTorch build that matches your machine first.

For training utilities:

```bash
pip install -e ".[training]"
```

## Model Files

The CLI downloads these files from
[`contextboxai/Kokoro-Vietnamese`](https://huggingface.co/contextboxai/Kokoro-Vietnamese)
when local paths are not provided:

- `kokoro_vi.pth`
- `kokoro_vi.onnx`
- `kokoro_vi_voicepack.pt`
- `config.json`

## Voices

Use these names with `voice=...` in Python or `--voice ...` in the CLI.

| Voice | Name |
| --- | --- |
| `diem_trinh` | Diễm Trinh |
| `hung_thinh` | Hưng Thịnh |
| `mai_linh` | Mai Linh |
| `mai_loan` | Mai Loan |
| `manh_dung` | Mạnh Dũng |
| `my_yen` | Mỹ Yến |
| `ngoc_huyen` | Ngọc Huyền |
| `phat_tai` | Phát Tài |
| `thanh_dat` | Thành Đạt |
| `thuc_trinh` | Thục Trinh |
| `tuan_ngoc` | Tuấn Ngọc |
| `storyvert` | storyvert |
| `duc_an` | Đức An |
| `duc_duy` | đức duy |

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

The package uses the same Kokoro runtime path as the original Gradio predictor.
Pass `normalize_peak=0.95` only if you want optional peak limiting before
writing WAV files.

```python
audio, phonemes = tts.synthesize("Tường nhà khách.", normalize_peak=0.95)
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

## ONNX Runtime

Install ONNX dependencies:

```bash
pip install -e ".[onnx]"
```

Export the PyTorch checkpoint to ONNX:

```bash
kokoro-vietnamese-export-onnx \
  --output outputs/kokoro_vi.onnx
```

Run ONNX Runtime inference:

```bash
kokoro-vietnamese-onnx \
  --text "Tường nhà khách đã được sơn lại." \
  --onnx outputs/kokoro_vi.onnx \
  --output outputs/onnx.wav \
  --device cpu \
  --print-phonemes
```

When `--onnx` is omitted, the command downloads `kokoro_vi.onnx` from the
Hugging Face model repo. Install `onnxruntime-gpu` and pass `--device cuda` to
use CUDAExecutionProvider when available.

## Gradio

```bash
kokoro-vietnamese-gradio --device cuda --share
```

If `AlbertModel` fails to import in an old environment:

```bash
pip install -U git+https://github.com/iamdinhthuan/Kokoro-Vietnamese.git "transformers>=4.48,<5"
```
