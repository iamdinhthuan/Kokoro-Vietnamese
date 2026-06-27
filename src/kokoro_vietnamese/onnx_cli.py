from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_CROSSFADE_MS,
    DEFAULT_HF_REPO_ID,
    DEFAULT_ONNX_FILE,
    DEFAULT_VOICE,
    DEFAULT_VOICEPACK_FILE,
    SAMPLE_RATE,
    _download_or_resolve,
    merge_audio_chunks,
    phonemize,
    resolve_voicepack_filename,
    split_text,
)
from .onnx_utils import load_config, phonemes_to_input_ids, select_voice_style, speed_input


def choose_providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    if device == 'cuda' and 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


class KokoroVietnameseONNX:
    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_HF_REPO_ID,
        voice: str | None = DEFAULT_VOICE,
        onnx_path: str | Path | None = None,
        voicepack_path: str | Path | None = None,
        config_path: str | Path | None = None,
        device: str = 'cpu',
    ) -> None:
        import onnxruntime as ort
        import torch

        self.onnx_path = _download_or_resolve(repo_id, DEFAULT_ONNX_FILE, onnx_path)
        voicepack_filename = resolve_voicepack_filename(voice, voicepack_path)
        self.voicepack_path = _download_or_resolve(repo_id, DEFAULT_VOICEPACK_FILE, voicepack_filename)
        self.config_path = _download_or_resolve(repo_id, DEFAULT_CONFIG_FILE, config_path)

        self.config = load_config(self.config_path)
        self.context_length = self.config['plbert']['max_position_embeddings']
        self.voicepack = torch.load(self.voicepack_path, map_location='cpu', weights_only=True)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=choose_providers(device))

    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    ) -> tuple[np.ndarray, str]:
        audio_chunks: list[np.ndarray] = []
        phoneme_chunks: list[str] = []
        speed_value = speed_input(speed)
        for index, text_chunk in enumerate(split_text(text), start=1):
            ps = phonemize(text_chunk)
            if not ps:
                continue
            phoneme_chunks.append(f'[{index}] {ps}')
            input_ids = phonemes_to_input_ids(
                ps,
                self.config['vocab'],
                context_length=self.context_length,
            )
            ref_s = select_voice_style(self.voicepack, len(ps))
            waveform, _duration = self.session.run(
                None,
                {
                    'input_ids': input_ids,
                    'ref_s': ref_s,
                    'speed': speed_value,
                },
            )
            audio_chunks.append(np.asarray(waveform, dtype=np.float32).reshape(-1))

        crossfade_samples = round(SAMPLE_RATE * int(crossfade_ms) / 1000)
        return merge_audio_chunks(audio_chunks, crossfade_samples), '\n'.join(phoneme_chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Vietnamese Kokoro ONNX inference')
    parser.add_argument('--text', required=True, help='Vietnamese text to synthesize')
    parser.add_argument('--output', default='outputs/kokoro_vi_onnx.wav', help='Output wav path')
    parser.add_argument('--repo-id', default=DEFAULT_HF_REPO_ID, help='Hugging Face model repo')
    parser.add_argument('--voice', default=DEFAULT_VOICE, help='Named voicepack from the HF repo')
    parser.add_argument('--onnx', help=f'Local ONNX path or HF filename. Downloads {DEFAULT_ONNX_FILE} when omitted.')
    parser.add_argument('--voicepack', help=f'Local voicepack path or HF filename. Downloads {DEFAULT_VOICEPACK_FILE} when omitted.')
    parser.add_argument('--config', help=f'Local config path or HF filename. Downloads {DEFAULT_CONFIG_FILE} when omitted.')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'], help='ONNX Runtime device')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--crossfade-ms', type=int, default=DEFAULT_CROSSFADE_MS, help='Sentence merge crossfade')
    parser.add_argument('--print-phonemes', action='store_true', help='Print generated phonemes')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tts = KokoroVietnameseONNX(
        repo_id=args.repo_id,
        voice=args.voice,
        onnx_path=args.onnx,
        voicepack_path=args.voicepack,
        config_path=args.config,
        device=args.device,
    )
    audio, phonemes = tts.synthesize(
        args.text,
        speed=args.speed,
        crossfade_ms=args.crossfade_ms,
    )
    if len(audio) == 0:
        raise RuntimeError('No audio generated.')

    import soundfile as sf

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, SAMPLE_RATE)
    if args.print_phonemes and phonemes:
        print(phonemes)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
