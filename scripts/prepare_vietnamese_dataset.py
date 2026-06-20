#!/usr/bin/env python3
"""Prepare a Vietnamese Kokoro/StyleTTS2 dataset.

The source dataset is expected to use metadata lines of:

    audio/000001.wav|Vietnamese transcript

Outputs are repeatable subset artifacts:

    dataset_vi_30h/audio/*.wav
    dataset_vi_30h/metadata.csv
    dataset_vi_30h/phonemes.csv
    dataset_vi_30h/stats.json
    training/vi_30h/train_list.txt
    training/vi_30h/val_list.txt
    training/vi_30h/OOD_texts.txt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.kokoro_symbols import dicts  # noqa: E402
from vig2p import VI_FIXUPS, VietnameseG2P  # noqa: E402

TEXT_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*|\s+|.", re.UNICODE)
WORD_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*$", re.UNICODE)
VI_MARK_RE = re.compile(r"[À-ỹĐđ]")
NON_VI_S_CLUSTERS = ('sc', 'sh', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw')

OOD_SENTENCES = [
    'Ma mà má mả mã mạ là sáu thanh điệu khác nhau trong tiếng Việt.',
    'Xin chào, hôm nay trời đẹp quá, chúng ta cùng luyện giọng nói tự nhiên.',
    'Tôi đang kiểm tra khả năng đọc đúng dấu hỏi, dấu ngã và dấu nặng.',
    'Một trăm hai mươi ba người đã đặt mua sản phẩm mới vào sáng thứ hai.',
    'Bạn có thể nói chậm hơn một chút được không?',
    'Cô ấy sống ở Hà Nội nhưng thường bay vào Thành phố Hồ Chí Minh làm việc.',
    'Chiếc điện thoại này có giá mười lăm triệu đồng và được bảo hành hai năm.',
    'Nếu trời mưa lớn, chúng ta sẽ dời cuộc họp sang chiều mai.',
    'Những âm cuối như anh, ang, ach, ac, am và ap cần được phát âm rõ.',
    'Dữ liệu sạch giúp mô hình học ngữ điệu và nhịp đọc ổn định hơn.',
    'Tại sao bạn lại chọn hướng nghiên cứu tổng hợp tiếng nói cho tiếng Việt?',
    'Ông ấy nói rằng kết quả thử nghiệm lần này khá khả quan.',
    'Các từ riêng như Tyson Fury, Deontay Wilder và Las Vegas cần được đọc hợp lý.',
    'Không nên bỏ qua bước kiểm tra transcript bằng công cụ nhận dạng giọng nói.',
    'Giọng đọc cần rõ ràng, không bị rè, không bị méo và không quá nhanh.',
    'Hãy giữ nguyên sự nhất quán giữa phoneme lúc train và lúc inference.',
    'Tôi muốn nghe câu này với tốc độ vừa phải và âm sắc tự nhiên.',
    'Sản phẩm mới ra mắt hôm nay đang nhận được rất nhiều phản hồi tích cực.',
    'Các dấu câu như dấu hỏi, dấu chấm than và dấu phẩy ảnh hưởng đến ngắt nghỉ.',
    'Đây là bài kiểm tra cuối cùng trước khi tạo voicepack cho giọng tiếng Việt.',
]


@dataclass(frozen=True)
class MetadataRow:
    source_rel: str
    text: str
    speaker_label: str


@dataclass(frozen=True)
class Entry:
    source_rel: str
    text: str
    output_rel: str
    phonemes: str
    duration_s: float
    speaker_label: str
    speaker_id: int


def fix_vi_phonemes(phonemes: str, source_text: str | None = None) -> str:
    for old, new in VI_FIXUPS:
        phonemes = phonemes.replace(old, new)
    if source_text:
        source_lower = source_text.lower()
        if source_lower.startswith('th'):
            phonemes = phonemes.replace('t', 'θ', 1)
        elif source_lower.startswith('tr'):
            phonemes = phonemes.replace('ʧ', 'ʈʂ', 1)
        elif source_lower.startswith('s') and not source_lower.startswith(NON_VI_S_CLUSTERS):
            phonemes = phonemes.replace('s', 'ʂ', 1)
        elif source_lower.startswith('gi') or re.match(r'^g[iìíỉĩị]', source_lower):
            phonemes = phonemes.replace('z', 'ʝ', 1)
    return phonemes


def unknown_symbols(phonemes: str) -> set[str]:
    return {ch for ch in phonemes if ch not in dicts}


def assert_vocab_compatible(phonemes: str) -> None:
    unknown = sorted(unknown_symbols(phonemes))
    if unknown:
        details = ', '.join(f'{repr(ch)} U+{ord(ch):04X}' for ch in unknown)
        raise ValueError(f'Unknown phoneme symbols: {details}')


def create_phonemizer():
    return VietnameseG2P()


def phonemize_text(text: str, g2p=None) -> str:
    g2p = g2p or create_phonemizer()
    phonemes = g2p.many([text])[0]
    assert_vocab_compatible(phonemes)
    return phonemes


def resolve_row_limit(limit: int | None, target_duration_hours: float | None) -> int | None:
    if limit is not None:
        return limit
    return None


def reached_target_duration(total_duration_s: float, target_duration_hours: float | None) -> bool:
    return target_duration_hours is not None and total_duration_s >= target_duration_hours * 3600


def read_metadata(
    metadata_path: Path,
    limit: int | None,
    default_speaker: str | int = '0',
) -> list[MetadataRow]:
    rows: list[MetadataRow] = []
    with metadata_path.open('r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, 1):
            if limit is not None and len(rows) >= limit:
                break
            line = raw_line.rstrip('\n')
            if not line:
                continue
            parts = [part.strip() for part in line.split('|')]
            if len(parts) < 2:
                raise ValueError(f'{metadata_path}:{line_no}: expected path|text or path|text|speaker')
            source_rel, text = parts[0].strip(), parts[1].strip()
            if source_rel.lower() in {'filename', 'path', 'audio'} and text.lower() == 'text':
                continue
            speaker_label = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else str(default_speaker)
            rows.append(MetadataRow(source_rel, text, speaker_label))
    if limit is not None and len(rows) < limit:
        raise ValueError(f'Only found {len(rows)} metadata rows, requested {limit}')
    return rows


def _is_int_label(label: str) -> bool:
    try:
        return str(int(label)) == label
    except ValueError:
        return False


def build_speaker_registry(rows: list[MetadataRow]) -> dict:
    labels = sorted({row.speaker_label for row in rows})
    if labels and all(_is_int_label(label) and int(label) >= 0 for label in labels):
        ids_by_label = {label: int(label) for label in labels}
    else:
        ids_by_label = {label: speaker_id for speaker_id, label in enumerate(labels)}
    labels_by_id = {str(speaker_id): label for label, speaker_id in ids_by_label.items()}
    counts_by_label = Counter(row.speaker_label for row in rows)
    return {
        'ids_by_label': ids_by_label,
        'labels_by_id': labels_by_id,
        'counts_by_label': dict(sorted(counts_by_label.items())),
        'counts_by_id': {
            str(ids_by_label[label]): count
            for label, count in sorted(counts_by_label.items())
        },
    }


def probe_wav(path: Path) -> tuple[int, int, float]:
    import soundfile as sf

    info = sf.info(str(path))
    return info.samplerate, info.channels, float(info.frames) / float(info.samplerate)


def materialize_audio(source: Path, dest: Path, sample_rate: int, channels: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    if sample_rate == 24000 and channels == 1:
        dest.symlink_to(source.resolve())
        return

    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-i',
            str(source),
            '-ac',
            '1',
            '-ar',
            '24000',
            '-sample_fmt',
            's16',
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_list(path: Path, entries: list[Entry]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for entry in entries:
            f.write(f'{entry.output_rel}|{entry.phonemes}|{entry.speaker_id}\n')


def split_entries(entries: list[Entry], val_ratio: float, seed: int) -> tuple[list[Entry], list[Entry]]:
    rng = random.Random(seed)
    by_speaker: dict[int, list[Entry]] = {}
    for entry in entries:
        by_speaker.setdefault(entry.speaker_id, []).append(entry)

    train_entries: list[Entry] = []
    val_entries: list[Entry] = []
    for speaker_id in sorted(by_speaker):
        speaker_entries = by_speaker[speaker_id][:]
        rng.shuffle(speaker_entries)
        if len(speaker_entries) == 1:
            train_entries.extend(speaker_entries)
            continue
        n_val = max(1, int(len(speaker_entries) * val_ratio))
        n_val = min(n_val, len(speaker_entries) - 1)
        val_entries.extend(speaker_entries[:n_val])
        train_entries.extend(speaker_entries[n_val:])

    if not val_entries and len(train_entries) > 1:
        val_entries.append(train_entries.pop())

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def prepare(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    metadata_path = args.metadata.resolve()
    output_dataset = args.output_dataset.resolve()
    output_training = args.output_training.resolve()

    if args.clean:
        shutil.rmtree(output_dataset, ignore_errors=True)
        shutil.rmtree(output_training, ignore_errors=True)

    output_dataset.mkdir(parents=True, exist_ok=True)
    output_training.mkdir(parents=True, exist_ok=True)

    row_limit = resolve_row_limit(args.limit, args.target_duration_hours)
    rows = read_metadata(metadata_path, row_limit, default_speaker=args.speaker_id)
    speaker_registry = build_speaker_registry(rows)
    ids_by_label = speaker_registry['ids_by_label']
    g2p = create_phonemizer()

    entries: list[Entry] = []
    skipped = Counter()
    tone_counts = Counter()
    unknown_by_symbol = Counter()
    converted_audio = 0
    symlinked_audio = 0
    total_duration = 0.0

    for row in rows:
        source = source_root / row.source_rel
        if not source.exists():
            skipped['missing_audio'] += 1
            continue

        try:
            phonemes = phonemize_text(row.text, g2p)
        except Exception as exc:
            skipped['g2p_or_vocab'] += 1
            for ch in getattr(exc, 'unknown', []):
                unknown_by_symbol[ch] += 1
            continue

        token_len = len(phonemes)
        if token_len > args.max_tokens:
            skipped['too_long'] += 1
            continue
        if token_len < args.min_tokens:
            skipped['too_short'] += 1
            continue

        try:
            sample_rate, channels, duration_s = probe_wav(source)
        except Exception:
            skipped['bad_audio'] += 1
            continue

        if duration_s < args.min_duration:
            skipped['audio_too_short'] += 1
            continue
        if duration_s > args.max_duration:
            skipped['audio_too_long'] += 1
            continue

        output_rel = str(Path('audio') / Path(row.source_rel).name)
        dest = output_dataset / output_rel
        materialize_audio(source, dest, sample_rate, channels)
        if sample_rate == 24000 and channels == 1:
            symlinked_audio += 1
        else:
            converted_audio += 1

        tone_counts.update(ch for ch in phonemes if ch in {'→', '↘', '↗', '↓', 'ʔ'})
        total_duration += duration_s
        entries.append(
            Entry(
                source_rel=row.source_rel,
                text=row.text,
                output_rel=output_rel,
                phonemes=phonemes,
                duration_s=duration_s,
                speaker_label=row.speaker_label,
                speaker_id=ids_by_label[row.speaker_label],
            )
        )
        if reached_target_duration(total_duration, args.target_duration_hours):
            break

    if not entries:
        raise RuntimeError('No valid entries produced')

    train_entries, val_entries = split_entries(entries, args.val_ratio, args.seed)

    write_list(output_training / 'train_list.txt', train_entries)
    write_list(output_training / 'val_list.txt', val_entries)

    ood_phonemes = [phonemize_text(sentence, g2p) for sentence in OOD_SENTENCES]
    with (output_training / 'OOD_texts.txt').open('w', encoding='utf-8') as f:
        f.write('\n'.join(ood_phonemes) + '\n')

    with (output_dataset / 'metadata.csv').open('w', encoding='utf-8') as f:
        f.write('filename|text|speaker|speaker_label\n')
        for entry in entries:
            f.write(f'{entry.output_rel}|{entry.text}|{entry.speaker_id}|{entry.speaker_label}\n')

    with (output_dataset / 'phonemes.csv').open('w', encoding='utf-8') as f:
        f.write('filename|ipa\n')
        for entry in entries:
            f.write(f'{entry.output_rel}|{entry.phonemes}\n')

    stats = {
        'source_root': str(source_root),
        'metadata': str(metadata_path),
        'phonemizer': 'vig2p',
        'requested_rows': row_limit,
        'target_duration_hours': args.target_duration_hours,
        'valid_entries': len(entries),
        'train_entries': len(train_entries),
        'val_entries': len(val_entries),
        'speaker_count': len(speaker_registry['ids_by_label']),
        'speakers': speaker_registry,
        'total_duration_h': round(total_duration / 3600, 4),
        'avg_duration_s': round(total_duration / len(entries), 3),
        'symlinked_audio': symlinked_audio,
        'converted_audio': converted_audio,
        'skipped': dict(skipped),
        'tone_token_counts': dict(sorted(tone_counts.items())),
        'unknown_by_symbol': {
            f'U+{ord(ch):04X} {ch}': count
            for ch, count in sorted(unknown_by_symbol.items())
        },
        'max_phoneme_length': max(len(e.phonemes) for e in entries),
        'min_phoneme_length': min(len(e.phonemes) for e in entries),
    }
    with (output_dataset / 'stats.json').open('w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write('\n')
    for registry_path in [output_dataset / 'speakers.json', output_training / 'speakers.json']:
        with registry_path.open('w', encoding='utf-8') as f:
            json.dump(speaker_registry, f, ensure_ascii=False, indent=2)
            f.write('\n')

    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare Vietnamese Kokoro/StyleTTS2 artifacts')
    parser.add_argument('--source-root', type=Path, required=True, help='Dataset root containing the audio paths referenced by metadata')
    parser.add_argument('--metadata', type=Path, required=True, help='Metadata file with rows like audio/000001.wav|Vietnamese transcript')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--target-duration-hours', type=float, default=None)
    parser.add_argument('--output-dataset', type=Path, default=Path('dataset_vi_30h'))
    parser.add_argument('--output-training', type=Path, default=Path('training/vi_30h'))
    parser.add_argument('--speaker-id', type=int, default=0, help='Default speaker id for 2-column metadata without an explicit speaker field')
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-tokens', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=510)
    parser.add_argument('--min-duration', type=float, default=0.6)
    parser.add_argument('--max-duration', type=float, default=30.0)
    parser.add_argument('--clean', action='store_true')
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == '__main__':
    main()
