#!/usr/bin/env python3
"""Audit Vietnamese G2P output before spending GPU time.

This catches cases where Vietnamese words collapse into the same Kokoro
phoneme string after `vig2p` conversion.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_vietnamese_dataset import unknown_symbols  # noqa: E402


VI_WORD_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*", re.UNICODE)
VI_MARK_RE = re.compile(r"[À-ỹĐđ]")

DEFAULT_MINIMAL_PAIRS = [
    ("tường", "thường"),
    ("tơ", "thơ"),
    ("cách", "kếch"),
    ("mạnh", "mệnh"),
    ("lạnh", "lệnh"),
    ("bạch", "bệch"),
    ("cành", "kềnh"),
    ("ngách", "nghếch"),
    ("đanh", "đênh"),
    ("gành", "ghềnh"),
    ("trước", "chước"),
    ("trúng", "chúng"),
    ("trị", "chị"),
    ("trắc", "chắc"),
    ("số", "xố"),
    ("sử", "xử"),
    ("sếp", "xếp"),
    ("giải", "dải"),
    ("gì", "dì"),
    ("gia", "da"),
]


@dataclass(frozen=True)
class WordPhonemes:
    word: str
    count: int
    raw: str
    fixed: str
    unknown: tuple[str, ...]


def metadata_texts(metadata_paths: list[Path]) -> list[str]:
    texts = []
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "|" if "|" in sample else ","
            reader = csv.reader(handle, delimiter=delimiter)
            for row_index, row in enumerate(reader):
                if not row:
                    continue
                if row_index == 0 and any(cell.lower() in {"text", "transcript", "sentence"} for cell in row):
                    continue
                if len(row) >= 2:
                    texts.append(row[1])
    return texts


def word_counts(texts: list[str], include_ascii: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for match in VI_WORD_RE.finditer(text.lower()):
            word = unicodedata.normalize("NFC", match.group(0).strip("-'"))
            if not word:
                continue
            if include_ascii or VI_MARK_RE.search(word):
                counts[word] += 1
    return counts


def phonemize_words(counts: Counter[str]) -> list[WordPhonemes]:
    from vig2p import VietnameseG2P

    g2p = VietnameseG2P()
    rows = []
    for word, count in counts.items():
        fixed = g2p.many([word])[0]
        rows.append(
            WordPhonemes(
                word=word,
                count=count,
                raw=fixed,
                fixed=fixed,
                unknown=tuple(sorted(unknown_symbols(fixed))),
            )
        )
    return rows


def adapter_collision_groups(rows: list[WordPhonemes]) -> list[tuple[int, str, list[WordPhonemes]]]:
    by_fixed: dict[str, list[WordPhonemes]] = defaultdict(list)
    for row in rows:
        by_fixed[row.fixed].append(row)

    groups = []
    for fixed, fixed_rows in by_fixed.items():
        if len(fixed_rows) > 1 and len({row.raw for row in fixed_rows}) > 1:
            groups.append((sum(row.count for row in fixed_rows), fixed, fixed_rows))
    groups.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    return groups


def same_phoneme_groups(rows: list[WordPhonemes]) -> list[tuple[int, str, list[WordPhonemes]]]:
    by_fixed: dict[str, list[WordPhonemes]] = defaultdict(list)
    for row in rows:
        by_fixed[row.fixed].append(row)

    groups = [
        (sum(row.count for row in fixed_rows), fixed, fixed_rows)
        for fixed, fixed_rows in by_fixed.items()
        if len(fixed_rows) > 1
    ]
    groups.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    return groups


def minimal_pair_failures(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows_by_word = {row.word: row for row in phonemize_words(Counter(word for pair in pairs for word in pair))}
    failures = []
    for left, right in pairs:
        left_row = rows_by_word[left]
        right_row = rows_by_word[right]
        if left_row.fixed == right_row.fixed:
            failures.append(
                {
                    "left": left,
                    "right": right,
                    "phonemes": left_row.fixed,
                    "left_raw": left_row.raw,
                    "right_raw": right_row.raw,
                }
            )
    return failures


def group_pair_count(groups: list[tuple[int, str, list[WordPhonemes]]]) -> int:
    return sum(len(list(itertools.combinations(rows, 2))) for _, _, rows in groups)


def print_groups(title: str, groups: list[tuple[int, str, list[WordPhonemes]]], limit: int) -> None:
    print(title)
    for total, fixed, rows in groups[:limit]:
        print(f"\nfixed: {fixed} total_count: {total} words: {len(rows)}")
        for row in sorted(rows, key=lambda item: (-item.count, item.word))[:20]:
            print(f"  {row.word:<18} count={row.count:<6} raw={row.raw}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Vietnamese frontend phoneme collisions")
    parser.add_argument("--metadata", type=Path, action="append", default=[], help="Metadata file with path|text rows")
    parser.add_argument("--include-ascii", action="store_true", help="Include ASCII-only words in the vocabulary audit")
    parser.add_argument("--top", type=int, default=20, help="Number of groups to print")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    parser.add_argument("--fail-on-adapter-collisions", action="store_true")
    parser.add_argument("--fail-on-minimal-pairs", action="store_true")
    args = parser.parse_args()

    for metadata_path in args.metadata:
        if not metadata_path.exists():
            parser.error(f"metadata not found: {metadata_path}")

    minimal_failures = minimal_pair_failures(DEFAULT_MINIMAL_PAIRS)
    texts = metadata_texts(args.metadata) if args.metadata else []
    counts = word_counts(texts, include_ascii=args.include_ascii)
    rows = phonemize_words(counts) if counts else []
    unknown = sorted({symbol for row in rows for symbol in row.unknown})
    adapter_groups = adapter_collision_groups(rows)
    homophone_groups = same_phoneme_groups(rows)

    summary = {
        "metadata_files": [str(path) for path in args.metadata],
        "texts": len(texts),
        "unique_words": len(counts),
        "tokens": sum(counts.values()),
        "unknown_rows": sum(1 for row in rows if row.unknown),
        "unknown_symbols": unknown,
        "minimal_pair_failures": minimal_failures,
        "adapter_caused_groups": len(adapter_groups),
        "adapter_caused_word_pairs": group_pair_count(adapter_groups),
        "all_same_phoneme_groups": len(homophone_groups),
        "all_same_phoneme_word_pairs": group_pair_count(homophone_groups),
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if adapter_groups:
            print_groups("\nADAPTER-CAUSED COLLISIONS", adapter_groups, args.top)
        if homophone_groups:
            print_groups("\nTOP SAME-PHONEME GROUPS", homophone_groups, min(args.top, 20))

    failed = False
    if args.fail_on_minimal_pairs and minimal_failures:
        failed = True
    if args.fail_on_adapter_collisions and adapter_groups:
        failed = True
    if unknown:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
