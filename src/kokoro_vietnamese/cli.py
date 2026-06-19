from __future__ import annotations

import argparse
from pathlib import Path

from .core import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_CROSSFADE_MS,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILE,
    DEFAULT_VOICEPACK_FILE,
    SAMPLE_RATE,
    KokoroVietnamese,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Vietnamese Kokoro text-to-speech inference')
    parser.add_argument('--text', required=True, help='Vietnamese text to synthesize')
    parser.add_argument('--output', default='outputs/kokoro_vi.wav', help='Output wav path')
    parser.add_argument('--repo-id', default=DEFAULT_HF_REPO_ID, help='Hugging Face model repo')
    parser.add_argument('--model', help=f'Local model path or HF filename. Downloads {DEFAULT_MODEL_FILE} when omitted.')
    parser.add_argument('--voicepack', help=f'Local voicepack path or HF filename. Downloads {DEFAULT_VOICEPACK_FILE} when omitted.')
    parser.add_argument('--config', help=f'Local config path or HF filename. Downloads {DEFAULT_CONFIG_FILE} when omitted.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Inference device')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--crossfade-ms', type=int, default=DEFAULT_CROSSFADE_MS, help='Sentence merge crossfade')
    parser.add_argument('--print-phonemes', action='store_true', help='Print generated phonemes')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tts = KokoroVietnamese(
        repo_id=args.repo_id,
        model_path=args.model,
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
