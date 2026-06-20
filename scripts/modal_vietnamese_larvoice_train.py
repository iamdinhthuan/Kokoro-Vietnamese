#!/usr/bin/env python3
"""Modal runner for Vietnamese LarVoice multi-speaker Kokoro training."""

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


APP_NAME = 'kokoro-vi-larvoice-rtxpro6000'
VOLUME_NAME = 'kokoro-vi-larvoice-rtxpro6000'
GPU = 'RTX-PRO-6000'

REMOTE_ROOT = Path('/data')
REMOTE_REPO = Path('/repo')
REMOTE_UPLOADS = REMOTE_ROOT / 'uploads'
REMOTE_SOURCE = REMOTE_ROOT / 'source'
REMOTE_TRAINING = REMOTE_ROOT / 'training'
REMOTE_WORK = REMOTE_ROOT / 'work'
REMOTE_LOG_DIR = REMOTE_WORK / 'logs/kokoro-vi-larvoice-ms'
REMOTE_STAGE1_CONFIG = REMOTE_WORK / 'config_vietnamese_larvoice_multispeaker_stage1_modal.yml'
REMOTE_STAGE2_CONFIG = REMOTE_WORK / 'config_vietnamese_larvoice_multispeaker_stage2_modal.yml'

SOURCE_TAR_NAME = 'dataset_voxcpm_larvoice.tar'
TRAINING_NAME = 'vi_larvoice'
META_NAME = 'dataset_vi_larvoice_meta'
PRETRAINED_NAME = 'kokoro_base.pth'


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _ignore_repo_path(path: Path) -> bool:
    parts = path.parts
    part_set = set(parts)
    name = path.name
    if name in {'.git', '.venv', '__pycache__'}:
        return True
    if any(part.startswith('dataset_vi_') for part in parts):
        return True
    if any(part == 'training' for part in parts) and any(
        part.startswith('vi_') for part in parts
    ):
        return True
    if any(part in {'gradio_cache', 'outputs'} for part in parts):
        return True
    if 'StyleTTS2' in part_set and 'logs' in part_set:
        return True
    if path.match('StyleTTS2/Utils/ASR/epoch_00080.pth'):
        return False
    if name.endswith(('.pth', '.pt', '.wav', '.mp3', '.flac')):
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
    config['epochs_2nd'] = epochs
    config['log_dir'] = str(REMOTE_LOG_DIR)
    config['pretrained_model'] = str(REMOTE_TRAINING / PRETRAINED_NAME)
    config['data_params']['root_path'] = str(REMOTE_SOURCE)
    config['data_params']['train_data'] = str(REMOTE_TRAINING / f'{TRAINING_NAME}/train_list.txt')
    config['data_params']['val_data'] = str(REMOTE_TRAINING / f'{TRAINING_NAME}/val_list.txt')
    config['data_params']['OOD_data'] = str(REMOTE_TRAINING / f'{TRAINING_NAME}/OOD_texts.txt')
    return config


def _run_training(config: dict[str, Any], config_path: Path, script_name: str) -> None:
    REMOTE_WORK.mkdir(parents=True, exist_ok=True)
    REMOTE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
    )
    volume.commit()

    env = os.environ.copy()
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', '1')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    cmd = [
        'accelerate',
        'launch',
        script_name,
        '--config_path',
        str(config_path),
    ]
    subprocess.run(cmd, cwd=REMOTE_REPO / 'StyleTTS2', env=env, check=True)
    volume.commit()


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _prepare_volume_impl(force_unpack: bool = False) -> dict[str, Any]:
    source_tar = REMOTE_UPLOADS / SOURCE_TAR_NAME
    if not source_tar.exists():
        raise FileNotFoundError(f'Missing upload: {source_tar}')

    if force_unpack and REMOTE_SOURCE.exists():
        shutil.rmtree(REMOTE_SOURCE)
    if not REMOTE_SOURCE.exists():
        REMOTE_SOURCE.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source_tar, 'r') as tar:
            tar.extractall(REMOTE_SOURCE)

    _copy_tree(REMOTE_UPLOADS / f'training/{TRAINING_NAME}', REMOTE_TRAINING / TRAINING_NAME)
    _copy_tree(REMOTE_UPLOADS / META_NAME, REMOTE_WORK / META_NAME)

    stats_path = REMOTE_WORK / f'{META_NAME}/stats.json'
    stats = json.loads(stats_path.read_text(encoding='utf-8'))
    train_lines = sum(1 for _ in (REMOTE_TRAINING / f'{TRAINING_NAME}/train_list.txt').open(encoding='utf-8'))
    val_lines = sum(1 for _ in (REMOTE_TRAINING / f'{TRAINING_NAME}/val_list.txt').open(encoding='utf-8'))
    audio_count = sum(1 for _ in (REMOTE_SOURCE / 'audio').iterdir())

    volume.commit()
    return {
        'audio_count': audio_count,
        'train_lines': train_lines,
        'val_lines': val_lines,
        'speaker_count': stats['speaker_count'],
        'duration_h': stats['total_duration_h'],
        'max_phoneme_length': stats['max_phoneme_length'],
    }


def _train_stage1_impl(batch_size: int = 20, epochs: int = 20) -> dict[str, Any]:
    base_config_path = REMOTE_REPO / 'configs/config_vietnamese_larvoice_multispeaker_stage1.yml'
    base_config = yaml.safe_load(base_config_path.read_text(encoding='utf-8'))
    config = make_remote_train_config(base_config, batch_size=batch_size, epochs=epochs)
    _run_training(config, REMOTE_STAGE1_CONFIG, 'train_first.py')

    first_stage = REMOTE_LOG_DIR / 'first_stage.pth'
    return {
        'log_dir': str(REMOTE_LOG_DIR),
        'config': str(REMOTE_STAGE1_CONFIG),
        'first_stage_exists': first_stage.exists(),
        'first_stage_size': first_stage.stat().st_size if first_stage.exists() else 0,
    }


def _train_stage2_impl(
    batch_size: int = 20,
    epochs: int = 5,
    resume_checkpoint: str = '',
) -> dict[str, Any]:
    first_stage = REMOTE_LOG_DIR / 'first_stage.pth'
    if not first_stage.exists():
        raise FileNotFoundError(f'Missing Stage 1 checkpoint: {first_stage}')

    base_config_path = REMOTE_REPO / 'configs/config_vietnamese_larvoice_multispeaker_stage2.yml'
    base_config = yaml.safe_load(base_config_path.read_text(encoding='utf-8'))
    config = make_remote_train_config(base_config, batch_size=batch_size, epochs=epochs)
    config['first_stage_path'] = first_stage.name
    config['second_stage_load_pretrained'] = False
    if resume_checkpoint:
        resume_path = REMOTE_LOG_DIR / resume_checkpoint
        if not resume_path.exists():
            raise FileNotFoundError(f'Missing Stage 2 resume checkpoint: {resume_path}')
        config['second_stage_resume_path'] = resume_checkpoint
    _run_training(config, REMOTE_STAGE2_CONFIG, 'train_second.py')

    checkpoints = sorted(REMOTE_LOG_DIR.glob('epoch_2nd_*.pth'))
    latest = checkpoints[-1] if checkpoints else None
    return {
        'log_dir': str(REMOTE_LOG_DIR),
        'config': str(REMOTE_STAGE2_CONFIG),
        'latest_stage2': latest.name if latest else None,
        'latest_stage2_size': latest.stat().st_size if latest else 0,
    }


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
)
def prepare_volume(force_unpack: bool = False) -> dict[str, Any]:
    return _prepare_volume_impl(force_unpack=force_unpack)


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
    ephemeral_disk=600000,
)
def train_stage1(batch_size: int = 20, epochs: int = 20, clean_log: bool = False) -> dict[str, Any]:
    if clean_log and REMOTE_LOG_DIR.exists():
        shutil.rmtree(REMOTE_LOG_DIR)
    return _train_stage1_impl(batch_size=batch_size, epochs=epochs)


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
    ephemeral_disk=600000,
)
def train_stage2(
    batch_size: int = 20,
    epochs: int = 5,
    resume_checkpoint: str = '',
) -> dict[str, Any]:
    return _train_stage2_impl(
        batch_size=batch_size,
        epochs=epochs,
        resume_checkpoint=resume_checkpoint,
    )


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(REMOTE_ROOT): volume},
    timeout=24 * 60 * 60,
    cpu=8,
    memory=65536,
    ephemeral_disk=600000,
)
def train_both(
    batch_size_stage1: int = 20,
    batch_size_stage2: int = 16,
    epochs_stage1: int = 10,
    epochs_stage2: int = 10,
    clean_log: bool = False,
) -> dict[str, Any]:
    if clean_log and REMOTE_LOG_DIR.exists():
        shutil.rmtree(REMOTE_LOG_DIR)
    stage1 = _train_stage1_impl(batch_size=batch_size_stage1, epochs=epochs_stage1)
    stage2 = _train_stage2_impl(batch_size=batch_size_stage2, epochs=epochs_stage2)
    return {
        'stage1': stage1,
        'stage2': stage2,
    }


@app.local_entrypoint()
def main(
    force_unpack: bool = False,
    batch_size: int = 20,
    epochs: int = 5,
    batch_size_stage1: int = 20,
    batch_size_stage2: int = 16,
    epochs_stage1: int = 10,
    epochs_stage2: int = 10,
    clean_log: bool = False,
    resume_checkpoint: str = '',
    stage: str = 'stage1',
    train: bool = True,
):
    prep = prepare_volume.remote(force_unpack=force_unpack)
    print(json.dumps({'prepare': prep}, ensure_ascii=False, indent=2))
    if train:
        if stage == 'stage1':
            result = train_stage1.remote(
                batch_size=batch_size,
                epochs=epochs,
                clean_log=clean_log,
            )
            print(json.dumps({'train_stage1': result}, ensure_ascii=False, indent=2))
        elif stage == 'stage2':
            result = train_stage2.remote(
                batch_size=batch_size,
                epochs=epochs,
                resume_checkpoint=resume_checkpoint,
            )
            print(json.dumps({'train_stage2': result}, ensure_ascii=False, indent=2))
        elif stage == 'both':
            result = train_both.remote(
                batch_size_stage1=batch_size_stage1,
                batch_size_stage2=batch_size_stage2,
                epochs_stage1=epochs_stage1,
                epochs_stage2=epochs_stage2,
                clean_log=clean_log,
            )
            print(json.dumps({'train_both': result}, ensure_ascii=False, indent=2))
        else:
            raise ValueError("stage must be 'stage1', 'stage2', or 'both'")
