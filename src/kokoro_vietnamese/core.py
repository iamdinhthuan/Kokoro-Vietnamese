from __future__ import annotations

import re
from pathlib import Path
from importlib import metadata

import numpy as np


DEFAULT_HF_REPO_ID = 'contextboxai/Kokoro-Vietnamese'
DEFAULT_MODEL_FILE = 'kokoro_vi.pth'
DEFAULT_VOICEPACK_FILE = 'kokoro_vi_voicepack.pt'
DEFAULT_CONFIG_FILE = 'config.json'
DEFAULT_VOICE = 'diem_trinh'
SAMPLE_RATE = 24000
DEFAULT_CROSSFADE_MS = 50
DEFAULT_NORMALIZE_PEAK = 0.95
VOICES = {
    'diem_trinh': {'label': 'Diễm Trinh', 'filename': 'voicepacks/diem_trinh.pt'},
    'hung_thinh': {'label': 'Hưng Thịnh', 'filename': 'voicepacks/hung_thinh.pt'},
    'mai_linh': {'label': 'Mai Linh', 'filename': 'voicepacks/mai_linh.pt'},
    'mai_loan': {'label': 'Mai Loan', 'filename': 'voicepacks/mai_loan.pt'},
    'manh_dung': {'label': 'Mạnh Dũng', 'filename': 'voicepacks/manh_dung.pt'},
    'my_yen': {'label': 'Mỹ Yến', 'filename': 'voicepacks/my_yen.pt'},
    'ngoc_huyen': {'label': 'Ngọc Huyền', 'filename': 'voicepacks/ngoc_huyen.pt'},
    'phat_tai': {'label': 'Phát Tài', 'filename': 'voicepacks/phat_tai.pt'},
    'thanh_dat': {'label': 'Thành Đạt', 'filename': 'voicepacks/thanh_dat.pt'},
    'thuc_trinh': {'label': 'Thục Trinh', 'filename': 'voicepacks/thuc_trinh.pt'},
    'tuan_ngoc': {'label': 'Tuấn Ngọc', 'filename': 'voicepacks/tuan_ngoc.pt'},
    'storyvert': {'label': 'storyvert', 'filename': 'voicepacks/storyvert.pt'},
    'duc_an': {'label': 'Đức An', 'filename': 'voicepacks/duc_an.pt'},
    'duc_duy': {'label': 'đức duy', 'filename': 'voicepacks/duc_duy.pt'},
}


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


def normalize_audio(audio: np.ndarray, peak: float | None = DEFAULT_NORMALIZE_PEAK) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if peak is None or peak <= 0 or len(audio) == 0:
        return audio

    max_abs = float(np.max(np.abs(audio)))
    if max_abs > peak:
        audio = audio * (float(peak) / max_abs)
    return audio.astype(np.float32, copy=False)


def phonemize(text: str) -> str:
    from vig2p import phonemize_text

    return phonemize_text(text)


def list_voices() -> list[str]:
    return sorted(VOICES)


def resolve_voicepack_filename(voice: str | None, voicepack: str | Path | None) -> str:
    if voicepack:
        return str(voicepack)
    voice_name = voice or DEFAULT_VOICE
    try:
        return VOICES[voice_name]['filename']
    except KeyError as exc:
        available = ', '.join(list_voices())
        raise ValueError(f'Unknown voice {voice_name!r}. Available voices: {available}') from exc


def _download_or_resolve(repo_id: str, filename: str, local_path: str | Path | None) -> Path:
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if path.exists():
            return path
        filename = str(local_path)

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def prepare_transformers_for_kokoro() -> None:
    try:
        import transformers
        import transformers.utils.import_utils as import_utils
        from packaging.version import Version
    except Exception as exc:
        raise RuntimeError('Install transformers>=4.48,<5 for Kokoro.') from exc

    version = Version(metadata.version('transformers'))
    if version.major >= 5:
        raise RuntimeError(
            f'transformers {version} is not supported by this package. '
            'Run: pip install "transformers>=4.48,<5"'
        )

    # Kokoro only uses ALBERT text modules. Broken optional vision/audio packages
    # such as an incompatible torchvision can make transformers lazy imports fail.
    for flag in ('_torchvision_available', '_librosa_available', '_cv2_available'):
        if hasattr(import_utils, flag):
            setattr(import_utils, flag, False)
    if hasattr(import_utils, '_torchvision_version'):
        import_utils._torchvision_version = 'N/A'

    try:
        from transformers import AlbertModel  # noqa: F401
    except Exception as exc:
        details = []
        for package in ('transformers', 'torch', 'torchvision', 'librosa', 'numpy'):
            try:
                details.append(f'{package}={metadata.version(package)}')
            except metadata.PackageNotFoundError:
                details.append(f'{package}=not-installed')
        raise RuntimeError(
            'Failed to import transformers.AlbertModel for Kokoro. '
            f'Environment: {", ".join(details)}. '
            'Try: pip uninstall -y torchvision librosa && '
            'pip install "transformers>=4.48,<5"'
        ) from exc


class KokoroVietnamese:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_HF_REPO_ID,
        voice: str | None = DEFAULT_VOICE,
        model_path: str | Path | None = None,
        voicepack_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = 'cuda',
    ) -> None:
        import torch
        prepare_transformers_for_kokoro()
        from kokoro import KModel

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self.device = device
        self.model_path = _download_or_resolve(repo_id, DEFAULT_MODEL_FILE, model_path)
        voicepack_filename = resolve_voicepack_filename(voice, voicepack_path)
        self.voicepack_path = _download_or_resolve(repo_id, DEFAULT_VOICEPACK_FILE, voicepack_filename)
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
        normalize_peak: float | None = DEFAULT_NORMALIZE_PEAK,
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
        audio = merge_audio_chunks(audio_chunks, crossfade_samples)
        return normalize_audio(audio, normalize_peak), '\n'.join(phoneme_chunks)
