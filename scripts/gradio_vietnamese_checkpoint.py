#!/usr/bin/env python3
"""Gradio app for Vietnamese Kokoro checkpoints.

The app resolves the best checkpoint from the configured log directory,
converts it to Kokoro KModel format, caches a voicepack, and runs
Vietnamese text through the same `KPipeline(lang_code='v')` frontend used by
the training preview.

Default device is CPU so this app can run while Stage 1/2 training uses the GPU.
Use `--device cuda` after training if you want faster inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
KOKORO_ROOT = REPO_ROOT / 'kokoro'
if str(KOKORO_ROOT) not in sys.path:
    sys.path.insert(0, str(KOKORO_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LOG_DIR = REPO_ROOT / 'StyleTTS2/logs/kokoro-vi-larvoice-ms-clean10'
DEFAULT_CACHE_DIR = REPO_ROOT / 'gradio_cache/vi_larvoice_clean10_latest'
DEFAULT_AUDIO_DIR = REPO_ROOT / 'dataset_vi_larvoice/audio'
DEFAULT_METADATA = REPO_ROOT / 'dataset_vi_larvoice/metadata.csv'
DEFAULT_SPEAKERS = REPO_ROOT / 'training/vi_larvoice/speakers.json'
DEFAULT_CONFIG = REPO_ROOT / 'training/config.json'
SAMPLE_RATE = 24000
DEFAULT_CROSSFADE_MS = 50


def checkpoint_stage(checkpoint_path: Path) -> int:
    name = checkpoint_path.name
    if name.startswith('epoch_2nd_'):
        return 2
    if name.startswith('epoch_1st_') or name == 'first_stage.pth':
        return 1
    return 0


def checkpoint_val_loss(metadata: dict[str, Any]) -> float | None:
    value = metadata.get('val_loss')
    if value is None:
        return None
    if hasattr(value, 'item'):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return {
        'epoch': checkpoint.get('epoch'),
        'val_loss': checkpoint.get('val_loss'),
    }


def find_latest_checkpoint(log_dir: Path, stage: int | None = None) -> Path:
    candidates = sorted(log_dir.glob('*.pth'))
    if not candidates:
        raise FileNotFoundError(f'No .pth checkpoint found in {log_dir}')
    if stage is not None:
        candidates = [path for path in candidates if checkpoint_stage(path) == stage]
        if not candidates:
            raise FileNotFoundError(f'No stage {stage} checkpoint found in {log_dir}')
    best_stage = stage if stage is not None else max(checkpoint_stage(path) for path in candidates)
    stage_candidates = [path for path in candidates if checkpoint_stage(path) == best_stage]
    return max(stage_candidates, key=lambda p: (p.stat().st_mtime, p.name))


def stage2_checkpoint_choices(log_dir: Path) -> list[str]:
    checkpoints = [path for path in log_dir.glob('epoch_2nd_*.pth') if path.is_file()]
    return [
        path.name
        for path in sorted(checkpoints, key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    ]


def find_best_checkpoint(
    log_dir: Path,
    metadata_loader: Callable[[Path], dict[str, Any]] | None = None,
    stage: int | None = None,
) -> Path:
    candidates = sorted(log_dir.glob('*.pth'))
    if not candidates:
        raise FileNotFoundError(f'No .pth checkpoint found in {log_dir}')

    if stage is None:
        stages = {checkpoint_stage(path) for path in candidates}
        stage = 2 if 2 in stages else 1 if 1 in stages else 0

    stage_candidates = [path for path in candidates if checkpoint_stage(path) == stage]
    if not stage_candidates:
        raise FileNotFoundError(f'No stage {stage} checkpoint found in {log_dir}')

    metadata_loader = metadata_loader or load_checkpoint_metadata
    rows = []
    for path in stage_candidates:
        try:
            metadata = metadata_loader(path)
        except Exception:
            continue
        rows.append((path, checkpoint_val_loss(metadata), path.stat().st_mtime))
    if not rows:
        raise FileNotFoundError(f'No readable stage {stage} checkpoint found in {log_dir}')

    with_loss = [row for row in rows if row[1] is not None]
    if with_loss:
        best_path, _, _ = min(with_loss, key=lambda row: (row[1], -row[2], row[0].name))
        return best_path

    best_path, _, _ = max(rows, key=lambda row: (row[2], row[0].name))
    return best_path


def needs_refresh(source: Path, target: Path) -> bool:
    return not target.exists() or target.stat().st_mtime < source.stat().st_mtime


def needs_voicepack_refresh(
    checkpoint_path: Path,
    target: Path,
    style_encoder_checkpoint: Path | None = None,
) -> bool:
    if needs_refresh(checkpoint_path, target):
        return True
    return (
        style_encoder_checkpoint is not None
        and target.stat().st_mtime < style_encoder_checkpoint.stat().st_mtime
    )


def speaker_slug(speaker: str) -> str:
    normalized = unicodedata.normalize('NFKD', str(speaker))
    ascii_text = normalized.encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', ascii_text).strip('_').lower()
    return slug or 'speaker'


def speaker_choices(speakers_path: Path | None) -> list[str]:
    if speakers_path is None or not speakers_path.exists():
        return ['0']
    data = json.loads(speakers_path.read_text(encoding='utf-8'))
    labels_by_id = data.get('labels_by_id') or {}
    if labels_by_id:
        return [
            labels_by_id[key]
            for key in sorted(labels_by_id, key=lambda value: int(value) if str(value).isdigit() else str(value))
        ]
    ids_by_label = data.get('ids_by_label') or {}
    if ids_by_label:
        return [
            label
            for label, _ in sorted(ids_by_label.items(), key=lambda item: item[1])
        ]
    return ['0']


def split_text_for_prediction(text: str) -> list[str]:
    normalized = re.sub(r'\s+', ' ', text.strip())
    if not normalized:
        return []

    chunks = []
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


def merge_audio_chunks_with_crossfade(
    chunks: list[np.ndarray],
    crossfade_samples: int,
) -> np.ndarray:
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


def converted_model_path(checkpoint_path: Path, cache_dir: Path) -> Path:
    return cache_dir / f'{checkpoint_path.stem}_kokoro.pth'


def voicepack_path(
    checkpoint_path: Path,
    cache_dir: Path,
    style_encoder_checkpoint: Path | None = None,
    speaker: str | None = None,
) -> Path:
    speaker_part = f'_speaker_{speaker_slug(speaker)}' if speaker is not None and str(speaker) else ''
    if style_encoder_checkpoint is not None:
        return cache_dir / (
            f'{checkpoint_path.stem}_style_{style_encoder_checkpoint.stem}{speaker_part}_voicepack.pt'
        )
    return cache_dir / f'{checkpoint_path.stem}{speaker_part}_voicepack.pt'


def convert_checkpoint(checkpoint_path: Path, output_path: Path) -> Path:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    net = checkpoint['net']

    def ensure_module_prefix(state_dict):
        return {
            (f'module.{key}' if not key.startswith('module.') else key): value
            for key, value in state_dict.items()
        }

    kokoro_weights = {}
    for key in ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder']:
        if key not in net:
            raise KeyError(f'{checkpoint_path} is missing inference component {key!r}')
        kokoro_weights[key] = ensure_module_prefix(net[key])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kokoro_weights, output_path)
    return output_path


def ensure_voicepack(
    checkpoint_path: Path,
    output_path: Path,
    audio_dir: Path,
    num_samples: int,
    device: str,
    style_encoder_checkpoint: Path | None = None,
    metadata_path: Path | None = None,
    speaker: str | None = None,
) -> Path:
    from scripts.extract_voicepack import extract_voicepack

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_voicepack(
        model_path=str(checkpoint_path),
        audio_dir=str(audio_dir),
        output_path=str(output_path),
        num_samples=num_samples,
        device=device,
        style_encoder_model=str(style_encoder_checkpoint) if style_encoder_checkpoint else None,
        metadata_path=str(metadata_path) if metadata_path and metadata_path.exists() else None,
        speaker=speaker,
    )
    return output_path


@dataclass
class LoadedCheckpoint:
    checkpoint_path: Path
    converted_path: Path
    voicepack_path: Path
    style_encoder_checkpoint: Path | None
    speaker: str | None
    status: str


class VietnameseCheckpointPredictor:
    def __init__(
        self,
        log_dir: Path,
        cache_dir: Path,
        audio_dir: Path,
        metadata_path: Path | None,
        speakers_path: Path | None,
        config_path: Path,
        device: str,
        num_voice_samples: int,
        checkpoint_strategy: str = 'best',
    ):
        self.log_dir = log_dir
        self.cache_dir = cache_dir
        self.audio_dir = audio_dir
        self.metadata_path = metadata_path
        self.speakers_path = speakers_path
        self.config_path = config_path
        self.device = device
        self.num_voice_samples = num_voice_samples
        self.checkpoint_strategy = checkpoint_strategy
        self._lock = threading.Lock()
        self._loaded: LoadedCheckpoint | None = None
        self._model = None
        self._pipeline = None
        self._voice = None

    def available_speakers(self) -> list[str]:
        return speaker_choices(self.speakers_path)

    def available_stage2_checkpoints(self) -> list[str]:
        return stage2_checkpoint_choices(self.log_dir)

    def _select_checkpoint(self, selected_checkpoint: str | None = None) -> tuple[Path, Path | None, str]:
        selected_checkpoint = (selected_checkpoint or '').strip()
        if selected_checkpoint:
            checkpoint_path = (self.log_dir / selected_checkpoint).resolve()
            if checkpoint_path.parent != self.log_dir.resolve():
                raise ValueError(f'Checkpoint must be inside {self.log_dir}: {selected_checkpoint}')
            if checkpoint_stage(checkpoint_path) != 2 or not checkpoint_path.exists():
                raise FileNotFoundError(f'Stage 2 checkpoint not found: {checkpoint_path}')
            strategy = 'selected'
        elif self.checkpoint_strategy == 'latest':
            checkpoint_path = find_latest_checkpoint(self.log_dir)
            strategy = 'latest'
        else:
            checkpoint_path = find_best_checkpoint(self.log_dir)
            strategy = 'best'

        style_encoder_checkpoint = None
        if checkpoint_stage(checkpoint_path) == 2:
            try:
                if self.checkpoint_strategy == 'latest':
                    style_encoder_checkpoint = find_latest_checkpoint(self.log_dir, stage=1)
                else:
                    style_encoder_checkpoint = find_best_checkpoint(self.log_dir, stage=1)
            except FileNotFoundError:
                style_encoder_checkpoint = None
        return checkpoint_path, style_encoder_checkpoint, strategy

    def load_checkpoint(
        self,
        force: bool = False,
        speaker: str | None = None,
        selected_checkpoint: str | None = None,
    ) -> LoadedCheckpoint:
        import torch
        from kokoro import KModel, KPipeline

        with self._lock:
            checkpoint_path, style_encoder_checkpoint, strategy = self._select_checkpoint(selected_checkpoint)
            converted_path = converted_model_path(checkpoint_path, self.cache_dir)
            vp_path = voicepack_path(
                checkpoint_path,
                self.cache_dir,
                style_encoder_checkpoint,
                speaker=speaker,
            )

            if needs_refresh(checkpoint_path, converted_path):
                convert_checkpoint(checkpoint_path, converted_path)
            if needs_voicepack_refresh(checkpoint_path, vp_path, style_encoder_checkpoint):
                ensure_voicepack(
                    checkpoint_path=checkpoint_path,
                    output_path=vp_path,
                    audio_dir=self.audio_dir,
                    num_samples=self.num_voice_samples,
                    device=self.device,
                    style_encoder_checkpoint=style_encoder_checkpoint,
                    metadata_path=self.metadata_path,
                    speaker=speaker,
                )

            already_loaded = (
                self._loaded is not None
                and self._loaded.checkpoint_path == checkpoint_path
                and self._loaded.style_encoder_checkpoint == style_encoder_checkpoint
                and self._loaded.speaker == speaker
                and not force
            )
            if already_loaded:
                return self._loaded

            self._model = None
            self._pipeline = None
            self._voice = None
            gc.collect()
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()

            kmodel = KModel(
                repo_id='hexgrad/Kokoro-82M',
                config=str(self.config_path),
                model=str(converted_path),
            ).to(self.device).eval()
            self._model = kmodel
            self._pipeline = KPipeline(
                lang_code='v',
                repo_id='hexgrad/Kokoro-82M',
                model=kmodel,
            )
            self._voice = torch.load(vp_path, map_location='cpu', weights_only=True)

            status = (
                f'Loaded {strategy} checkpoint {checkpoint_path.name} on {self.device}; '
                f'converted={converted_path.name}; voicepack={vp_path.name}'
            )
            if style_encoder_checkpoint is not None:
                status += f'; style_encoder={style_encoder_checkpoint.name}'
            if speaker is not None:
                status += f'; speaker={speaker}'
            self._loaded = LoadedCheckpoint(
                checkpoint_path=checkpoint_path,
                converted_path=converted_path,
                voicepack_path=vp_path,
                style_encoder_checkpoint=style_encoder_checkpoint,
                speaker=speaker,
                status=status,
            )
            return self._loaded

    def load_latest(self, force: bool = False) -> LoadedCheckpoint:
        return self.load_checkpoint(force=force)

    def predict(
        self,
        text: str,
        speed: float,
        reload_latest: bool,
        speaker: str | None = None,
        selected_checkpoint: str | None = None,
    ) -> tuple[tuple[int, np.ndarray] | None, str, str]:
        if not text or not text.strip():
            return None, 'Enter Vietnamese text first.', ''

        loaded = self.load_checkpoint(
            force=reload_latest,
            speaker=speaker,
            selected_checkpoint=selected_checkpoint,
        )
        assert self._pipeline is not None
        assert self._voice is not None

        chunks = []
        phoneme_chunks = []
        text_chunks = split_text_for_prediction(text)
        for index, text_chunk in enumerate(text_chunks, start=1):
            for _, phonemes, audio in self._pipeline(
                text_chunk,
                voice=self._voice,
                speed=float(speed),
                split_pattern=None,
            ):
                if phonemes:
                    phoneme_chunks.append(f'[{index}] {phonemes}')
                if audio is not None:
                    chunks.append(audio.detach().cpu().numpy())

        if not chunks:
            return None, f'{loaded.status}; no audio generated.', '\n'.join(phoneme_chunks)

        crossfade_samples = round(SAMPLE_RATE * DEFAULT_CROSSFADE_MS / 1000)
        audio = merge_audio_chunks_with_crossfade(chunks, crossfade_samples)
        status = f'{loaded.status}; chunks={len(text_chunks)}; crossfade={DEFAULT_CROSSFADE_MS}ms'
        return (SAMPLE_RATE, audio), status, '\n'.join(phoneme_chunks)


def build_demo(predictor: VietnameseCheckpointPredictor):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            'Gradio is not installed. Install it with: '
            'uv pip install --python .venv/bin/python gradio'
        ) from exc

    with gr.Blocks(title='Kokoro Vietnamese Checkpoint Predictor') as demo:
        gr.Markdown('# Kokoro Vietnamese Checkpoint Predictor')
        speakers = predictor.available_speakers()
        stage2_checkpoints = predictor.available_stage2_checkpoints()
        with gr.Row():
            text = gr.Textbox(
                label='Vietnamese text',
                value='Xin chào, hôm nay tôi đang kiểm tra giọng đọc tiếng Việt.',
                lines=4,
            )
        with gr.Row():
            speed = gr.Slider(0.6, 1.4, value=1.0, step=0.05, label='Speed')
            speaker = gr.Dropdown(
                choices=speakers,
                value=speakers[0] if speakers else None,
                label='Speaker',
            )
            checkpoint = gr.Dropdown(
                choices=stage2_checkpoints,
                value=stage2_checkpoints[0] if stage2_checkpoints else None,
                label='Stage 2 checkpoint',
            )
            reload_latest = gr.Checkbox(value=True, label='Reload checkpoint before predict')
        with gr.Row():
            predict_btn = gr.Button('Predict', variant='primary')
            reload_btn = gr.Button('Reload checkpoint')
        audio = gr.Audio(label='Generated audio', type='numpy')
        status = gr.Textbox(label='Status', lines=2)
        phonemes = gr.Textbox(label='Phonemes', lines=6)

        predict_btn.click(
            predictor.predict,
            inputs=[text, speed, reload_latest, speaker, checkpoint],
            outputs=[audio, status, phonemes],
        )
        reload_btn.click(
            lambda selected_speaker, selected_checkpoint: predictor.load_checkpoint(
                force=True,
                speaker=selected_speaker,
                selected_checkpoint=selected_checkpoint,
            ).status,
            inputs=[speaker, checkpoint],
            outputs=[status],
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve Vietnamese checkpoint with Gradio')
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument('--audio-dir', type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument('--metadata', type=Path, default=DEFAULT_METADATA)
    parser.add_argument('--speakers', type=Path, default=DEFAULT_SPEAKERS)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--num-voice-samples', type=int, default=80)
    parser.add_argument('--checkpoint-strategy', choices=['best', 'latest'], default='latest')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = VietnameseCheckpointPredictor(
        log_dir=args.log_dir.resolve(),
        cache_dir=args.cache_dir.resolve(),
        audio_dir=args.audio_dir.resolve(),
        metadata_path=args.metadata.resolve() if args.metadata else None,
        speakers_path=args.speakers.resolve() if args.speakers else None,
        config_path=args.config.resolve(),
        device=args.device,
        num_voice_samples=args.num_voice_samples,
        checkpoint_strategy=args.checkpoint_strategy,
    )
    demo = build_demo(predictor)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
