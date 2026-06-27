from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def phonemes_to_input_ids(
    phonemes: str,
    vocab: dict[str, int],
    *,
    context_length: int = 512,
) -> np.ndarray:
    input_ids = [vocab[p] for p in phonemes if p in vocab]
    if len(input_ids) + 2 > context_length:
        raise ValueError(
            f'Phoneme sequence too long: {len(input_ids) + 2} > {context_length}'
        )
    return np.asarray([[0, *input_ids, 0]], dtype=np.int64)


def select_voice_style(voicepack: Any, phoneme_count: int) -> np.ndarray:
    if phoneme_count <= 0:
        raise ValueError('phoneme_count must be positive')

    if hasattr(voicepack, 'detach'):
        voicepack = voicepack.detach().cpu().numpy()
    voicepack = np.asarray(voicepack, dtype=np.float32)
    if voicepack.ndim != 3 or voicepack.shape[1:] != (1, 256):
        raise ValueError(
            'Expected voicepack shape [max_phonemes, 1, 256], '
            f'got {tuple(voicepack.shape)}'
        )

    index = min(phoneme_count, voicepack.shape[0]) - 1
    return np.asarray(voicepack[index], dtype=np.float32)


def speed_input(speed: float) -> np.ndarray:
    if speed <= 0:
        raise ValueError('speed must be greater than 0')
    return np.asarray(float(speed), dtype=np.float32)
