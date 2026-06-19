from __future__ import annotations

import re
from pathlib import Path

import numpy as np


DEFAULT_HF_REPO_ID = 'contextboxai/Kokoro-Vietnamese'
DEFAULT_MODEL_FILE = 'kokoro_vi.pth'
DEFAULT_VOICEPACK_FILE = 'kokoro_vi_voicepack.pt'
DEFAULT_CONFIG_FILE = 'config.json'
SAMPLE_RATE = 24000
DEFAULT_CROSSFADE_MS = 50


def split_text(text: str) -> list[str]:
    normalized = re.sub(r'\s+', ' ', text.strip())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    for match in re.finditer(r'[.!?…]+(?:["”’)]*)', normalized):
        end = match.end()
        if end < len(normalized) and not normalized[end].isspace():
            continue
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end

    remainder = normalized[start:].strip()
    if remainder:
        chunks.append(remainder)
    return chunks


def merge_audio_chunks(chunks: list[np.ndarray], crossfade_samples: int) -> np.ndarray:
    valid_chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks if len(chunk) > 0]
    if not valid_chunks:
        return np.array([], dtype=np.float32)

    merged = valid_chunks[0]
    for chunk in valid_chunks[1:]:
        overlap = min(int(crossfade_samples), len(merged), len(chunk))
        if overlap <= 0:
            merged = np.concatenate([merged, chunk])
            continue

        fade_out = np.linspace(1.0, 0.0, overlap + 2, dtype=np.float32)[1:-1]
        fade_in = 1.0 - fade_out
        crossfaded = (merged[-overlap:] * fade_out) + (chunk[:overlap] * fade_in)
        merged = np.concatenate([merged[:-overlap], crossfaded, chunk[overlap:]])
    return merged.astype(np.float32, copy=False)


def phonemize(text: str) -> str:
    from vig2p import phonemize_text

    return phonemize_text(text)


def _download_or_resolve(repo_id: str, filename: str, local_path: str | Path | None) -> Path:
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if path.exists():
            return path
        filename = str(local_path)

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


class KokoroVietnamese:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_HF_REPO_ID,
        model_path: str | Path | None = None,
        voicepack_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = 'cuda',
    ) -> None:
        import torch
        from kokoro import KModel

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.model_path = _download_or_resolve(repo_id, DEFAULT_MODEL_FILE, model_path)
        self.voicepack_path = _download_or_resolve(repo_id, DEFAULT_VOICEPACK_FILE, voicepack_path)
        self.config_path = _download_or_resolve(repo_id, DEFAULT_CONFIG_FILE, config_path)

        self.model = KModel(
            repo_id='hexgrad/Kokoro-82M',
            config=str(self.config_path),
            model=str(self.model_path),
        ).to(device).eval()
        self.voicepack = torch.load(self.voicepack_path, map_location='cpu', weights_only=True)

    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    ) -> tuple[np.ndarray, str]:
        import torch

        audio_chunks: list[np.ndarray] = []
        phoneme_chunks: list[str] = []
        for index, text_chunk in enumerate(split_text(text), start=1):
            ps = phonemize(text_chunk)
            if not ps:
                continue
            if len(ps) > 510:
                raise ValueError(f'Phoneme chunk too long ({len(ps)} > 510): {text_chunk[:80]}')

            with torch.no_grad():
                ref_s = self.voicepack[len(ps) - 1]
                audio = self.model(ps, ref_s, float(speed))
            phoneme_chunks.append(f'[{index}] {ps}')
            audio_chunks.append(audio.detach().cpu().numpy())

        crossfade_samples = round(SAMPLE_RATE * int(crossfade_ms) / 1000)
        return merge_audio_chunks(audio_chunks, crossfade_samples), '\n'.join(phoneme_chunks)
