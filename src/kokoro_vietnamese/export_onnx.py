from __future__ import annotations

import argparse
from pathlib import Path

from .core import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILE,
    DEFAULT_ONNX_FILE,
    DEFAULT_VOICEPACK_FILE,
    _download_or_resolve,
    phonemize,
    prepare_transformers_for_kokoro,
)
from .onnx_utils import load_config, phonemes_to_input_ids, select_voice_style, speed_input


DEFAULT_SAMPLE_TEXT = 'Xin chào, tôi đang kiểm tra ONNX.'


def sample_phonemes(text: str) -> str:
    phonemes = phonemize(text)
    if not phonemes:
        raise RuntimeError('Could not produce sample phonemes for ONNX export.')
    return phonemes


def export_onnx(
    *,
    model_path: Path,
    voicepack_path: Path,
    config_path: Path,
    output_path: Path,
    sample_text: str = DEFAULT_SAMPLE_TEXT,
    opset: int = 18,
    check: bool = True,
) -> Path:
    import torch
    from ._kokoro import KModel
    from ._kokoro.model import KModelForONNX

    prepare_transformers_for_kokoro()
    config = load_config(config_path)
    phonemes = sample_phonemes(sample_text)
    input_ids = torch.from_numpy(
        phonemes_to_input_ids(
            phonemes,
            config['vocab'],
            context_length=config['plbert']['max_position_embeddings'],
        )
    )
    voicepack = torch.load(voicepack_path, map_location='cpu', weights_only=True)
    ref_s = torch.from_numpy(select_voice_style(voicepack, len(phonemes)))
    speed = torch.from_numpy(speed_input(1.0))

    model = KModel(
        repo_id='hexgrad/Kokoro-82M',
        config=str(config_path),
        model=str(model_path),
        disable_complex=True,
    ).eval()
    wrapper = KModelForONNX(model).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, ref_s, speed),
            str(output_path),
            input_names=['input_ids', 'ref_s', 'speed'],
            output_names=['waveform', 'duration'],
            opset_version=opset,
            dynamic_axes={
                'input_ids': {1: 'tokens'},
                'waveform': {0: 'samples'},
                'duration': {0: 'tokens'},
            },
        )

    if check:
        import onnx

        onnx.checker.check_model(str(output_path))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Export Kokoro Vietnamese to ONNX')
    parser.add_argument('--repo-id', default=DEFAULT_HF_REPO_ID, help='Hugging Face model repo')
    parser.add_argument('--model', help=f'Local model path or HF filename. Downloads {DEFAULT_MODEL_FILE} when omitted.')
    parser.add_argument('--voicepack', help=f'Local voicepack path or HF filename. Downloads {DEFAULT_VOICEPACK_FILE} when omitted.')
    parser.add_argument('--config', help=f'Local config path or HF filename. Downloads {DEFAULT_CONFIG_FILE} when omitted.')
    parser.add_argument('--output', default=f'outputs/{DEFAULT_ONNX_FILE}', help='Output ONNX path')
    parser.add_argument('--sample-text', default=DEFAULT_SAMPLE_TEXT, help='Text used to trace the ONNX graph')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version')
    parser.add_argument('--no-check', action='store_true', help='Skip onnx.checker validation')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_path = _download_or_resolve(args.repo_id, DEFAULT_MODEL_FILE, args.model)
    voicepack_path = _download_or_resolve(args.repo_id, DEFAULT_VOICEPACK_FILE, args.voicepack)
    config_path = _download_or_resolve(args.repo_id, DEFAULT_CONFIG_FILE, args.config)
    output_path = export_onnx(
        model_path=model_path,
        voicepack_path=voicepack_path,
        config_path=config_path,
        output_path=Path(args.output),
        sample_text=args.sample_text,
        opset=args.opset,
        check=not args.no_check,
    )
    print(output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
