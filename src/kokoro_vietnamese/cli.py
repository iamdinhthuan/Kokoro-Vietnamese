from __future__ import annotations

import argparse
from pathlib import Path

from .core import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_CROSSFADE_MS,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILE,
    DEFAULT_NORMALIZE_PEAK,
    DEFAULT_VOICE,
    DEFAULT_VOICEPACK_FILE,
    SAMPLE_RATE,
    KokoroVietnamese,
    list_voices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Vietnamese Kokoro text-to-speech inference')
    parser.add_argument('--text', help='Vietnamese text to synthesize')
    parser.add_argument('--output', default='outputs/kokoro_vi.wav', help='Output wav path')
    parser.add_argument('--repo-id', default=DEFAULT_HF_REPO_ID, help='Hugging Face model repo')
    parser.add_argument('--voice', default=DEFAULT_VOICE, choices=list_voices(), help='Named voicepack from the HF repo')
    parser.add_argument('--model', help=f'Local model path or HF filename. Downloads {DEFAULT_MODEL_FILE} when omitted.')
    parser.add_argument('--voicepack', help=f'Local voicepack path or HF filename. Downloads {DEFAULT_VOICEPACK_FILE} when omitted.')
    parser.add_argument('--config', help=f'Local config path or HF filename. Downloads {DEFAULT_CONFIG_FILE} when omitted.')
    parser.add_argument('--text-file', help='Read synthesis text from a UTF-8 text file instead of --text')
    parser.add_argument('--batch-file', help='Read one utterance per line and write numbered wav files under --output-dir')
    parser.add_argument('--output-dir', default='outputs', help='Directory for --batch-file outputs')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Inference device')
    parser.add_argument('--speed', type=float, default=1.0, help='Speech speed multiplier')
    parser.add_argument('--crossfade-ms', type=int, default=DEFAULT_CROSSFADE_MS, help='Sentence merge crossfade')
    parser.add_argument(
        '--normalize-peak',
        type=float,
        default=DEFAULT_NORMALIZE_PEAK,
        help='Optionally scale audio down when peak amplitude exceeds this value. Disabled by default.',
    )
    parser.add_argument('--print-phonemes', action='store_true', help='Print generated phonemes')
    parser.add_argument('--list-voices', action='store_true', help='Print available voice names and exit')
    return parser


def _read_text_argument(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding='utf-8')
    if args.text:
        return args.text
    raise SystemExit('Pass --text, --text-file, --batch-file, or --list-voices.')


def _write_wav(path: Path, audio) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_voices:
        for voice in list_voices():
            print(voice)
        return 0

    tts = KokoroVietnamese(
        repo_id=args.repo_id,
        voice=args.voice,
        model_path=args.model,
        voicepack_path=args.voicepack,
        config_path=args.config,
        device=args.device,
    )

    if args.batch_file:
        output_dir = Path(args.output_dir)
        rows = [
            line.strip()
            for line in Path(args.batch_file).read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        for index, text in enumerate(rows, start=1):
            audio, phonemes = tts.synthesize(
                text,
                speed=args.speed,
                crossfade_ms=args.crossfade_ms,
                normalize_peak=args.normalize_peak or None,
            )
            if len(audio) == 0:
                raise RuntimeError(f'No audio generated for batch row {index}.')
            output = output_dir / f'{index:04d}.wav'
            _write_wav(output, audio)
            if args.print_phonemes and phonemes:
                print(f'#{index}\n{phonemes}')
            print(output)
        return 0

    audio, phonemes = tts.synthesize(
        _read_text_argument(args),
        speed=args.speed,
        crossfade_ms=args.crossfade_ms,
        normalize_peak=args.normalize_peak or None,
    )
    if len(audio) == 0:
        raise RuntimeError('No audio generated.')

    output = Path(args.output)
    _write_wav(output, audio)
    if args.print_phonemes and phonemes:
        print(phonemes)
    print(output)
    return 0
