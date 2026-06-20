#!/usr/bin/env python3
"""Modal runner for the local 100h Vietnamese multi-speaker Kokoro training."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import modal
import yaml


APP_NAME = 'kokoro-vi-100h-rtxpro6000'
VOLUME_NAME = 'kokoro-vi-100h-rtxpro6000'
GPU = 'RTX-PRO-6000'

REMOTE_ROOT = Path('/data')
REMOTE_REPO = Path('/repo')
REMOTE_UPLOADS = REMOTE_ROOT / 'uploads'
REMOTE_SOURCE = REMOTE_ROOT / 'source'
REMOTE_TRAINING = REMOTE_ROOT / 'training'
REMOTE_WORK = REMOTE_ROOT / 'work'
REMOTE_LOG_DIR = REMOTE_WORK / 'logs/kokoro-vi-100h-ms'
REMOTE_CONFIG = REMOTE_WORK / 'config_vietnamese_100h_multispeaker_stage1_modal.yml'

SOURCE_TAR_NAME = 'vivoice_kokoro_100h_clean.tar'


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _ignore_repo_path(path: Path) -> bool:
    parts = set(path.parts)
    name = path.name
    if name in {'.git', '.venv', '__pycache__'}:
        return True
    if 'StyleTTS2' in parts and 'logs' in parts:
        return True
    if path.match('StyleTTS2/Utils/ASR/epoch_00080.pth'):
        return False
    if name.endswith(('.pth', '.pt', '.wav', '.mp3', '.flac')):
        return True
    if path.match('dataset_vi_*') or path.match('training/vi_*'):
        return True
    return False


image = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('ffmpeg', 'libsndfile1', 'git')
    .uv_pip_install(
        'torch==2.11.0',
        'torchaudio==2.11.0',
        index_url='https://download.pytorch.org/whl/cu128',
    )
    .uv_pip_install(
        'accelerate',
        'einops',
        'einops-exts',
        'gradio',
        'huggingface_hub>=0.28,<1',
        'librosa',
        'loguru',
        'matplotlib',
        'misaki[en]>=0.9.4',
        'munch',
        'nltk',
        'numpy',
        'pydub',
        'pyyaml',
        'vig2p>=0.1.0',
        'soundfile',
        'tensorboard',
        'tqdm',
        'transformers>=4.48,<5',
        'git+https://github.com/resemble-ai/monotonic_align.git',
    )
    .add_local_dir('.', str(REMOTE_REPO), ignore=_ignore_repo_path)
)


def make_remote_train_config(
    base_config: dict[str, Any],
    *,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['epochs_1st'] = epochs
    config['log_dir'] = str(REMOTE_LOG_DIR)
    config['pretrained_model'] = str(REMOTE_TRAINING / 'kokoro_base.pth')
    config['data_params']['root_path'] = str(REMOTE_SOURCE)
    config['data_params']['train_data'] = str(REMOTE_TRAINING / 'vi_100h/train_list.txt')
    config['data_params']['val_data'] = str(REMOTE_TRAINING / 'vi_100h/val_list.txt')
    config['data_params']['OOD_data'] = str(REMOTE_TRAINING / 'vi_100h/OOD_texts.txt')
    return config


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
)
def prepare_volume(force_unpack: bool = False) -> dict[str, Any]:
    source_tar = REMOTE_UPLOADS / SOURCE_TAR_NAME
    if not source_tar.exists():
        raise FileNotFoundError(f'Missing upload: {source_tar}')

    if force_unpack and REMOTE_SOURCE.exists():
        shutil.rmtree(REMOTE_SOURCE)
    if not REMOTE_SOURCE.exists():
        REMOTE_SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source_tar, 'r') as tar:
            tar.extractall(REMOTE_SOURCE)

    _copy_tree(REMOTE_UPLOADS / 'training/vi_100h', REMOTE_TRAINING / 'vi_100h')
    _copy_tree(REMOTE_UPLOADS / 'dataset_vi_100h_meta', REMOTE_WORK / 'dataset_vi_100h_meta')

    stats_path = REMOTE_WORK / 'dataset_vi_100h_meta/stats.json'
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    train_lines = sum(1 for _ in (REMOTE_TRAINING / 'vi_100h/train_list.txt').open(encoding='utf-8'))
    val_lines = sum(1 for _ in (REMOTE_TRAINING / 'vi_100h/val_list.txt').open(encoding='utf-8'))
    audio_count = sum(1 for _ in (REMOTE_SOURCE / 'audio').iterdir())

    volume.commit()
    return {
        'audio_count': audio_count,
        'train_lines': train_lines,
        'val_lines': val_lines,
        'speaker_count': stats['speaker_count'],
        'duration_h': stats['total_duration_h'],
    }


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
    ephemeral_disk=600000,
)
def train_stage1(batch_size: int = 20, epochs: int = 20) -> dict[str, Any]:
    base_config_path = REMOTE_REPO / 'configs/config_vietnamese_100h_multispeaker_stage1.yml'
    base_config = yaml.safe_load(base_config_path.read_text(encoding='utf-8'))
    config = make_remote_train_config(base_config, batch_size=batch_size, epochs=epochs)

    REMOTE_WORK.mkdir(parents=True, exist_ok=True)
    REMOTE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_CONFIG.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    volume.commit()

    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    cmd = [
        'accelerate',
        'launch',
        'train_first.py',
        '--config_path',
        str(REMOTE_CONFIG),
    ]
    subprocess.run(cmd, cwd=REMOTE_REPO / 'StyleTTS2', env=env, check=True)
    volume.commit()

    first_stage = REMOTE_LOG_DIR / 'first_stage.pth'
    return {
        'log_dir': str(REMOTE_LOG_DIR),
        'config': str(REMOTE_CONFIG),
        'first_stage_exists': first_stage.exists(),
        'first_stage_size': first_stage.stat().st_size if first_stage.exists() else 0,
    }


@app.local_entrypoint()
def main(
    force_unpack: bool = False,
    batch_size: int = 20,
    epochs: int = 20,
    train: bool = True,
):
    prep = prepare_volume.remote(force_unpack=force_unpack)
    print(json.dumps({'prepare': prep}, ensure_ascii=False, indent=2))
    if train:
        result = train_stage1.remote(batch_size=batch_size, epochs=epochs)
        print(json.dumps({'train_stage1': result}, ensure_ascii=False, indent=2))
