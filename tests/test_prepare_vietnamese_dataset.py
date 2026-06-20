import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import scripts.prepare_vietnamese_dataset as prep
from scripts.prepare_vietnamese_dataset import (
    Entry,
    assert_vocab_compatible,
    fix_vi_phonemes,
    phonemize_text,
    reached_target_duration,
    read_metadata,
    resolve_row_limit,
)


class VietnameseDatasetPrepTest(unittest.TestCase):
    def test_fix_vi_phonemes_maps_vig2p_tones_and_phone_fixups(self):
        raw = "sˈin tʃˈaː2w, ɗˈɛ6p t̪ˈiɛɜŋ tˈe-ɲ mˈaː5 mˈaː6 ˈi ' ’ ‘ */& – ˈɛm"

        fixed = fix_vi_phonemes(raw)

        self.assertEqual(
            fixed,
            'sˈin ʧˈaː↘w, dˈɛʔ↓p tˈiɛ↗ŋ tˈæɲ mˈaːʔ↗ mˈaːʔ↓ ˈi       — ˈɛm',
        )

    def test_phonemize_text_uses_vig2p_for_vietnamese_r(self):
        phonemes = phonemize_text('tiếng ve râm ran')

        self.assertIn('ɹˈəm ɹˈaːn', phonemes)
        self.assertNotIn('zˈəm zˈaːn', phonemes)

    def test_phonemize_text_distinguishes_t_and_th_onsets(self):
        tuong = phonemize_text('tường nhà khách')
        thuong = phonemize_text('thường nhà khách')

        self.assertEqual(tuong, 'tˈyə↘ŋ ɲˈaː↘ xˈæ↗c')
        self.assertEqual(thuong, 'θˈyə↘ŋ ɲˈaː↘ xˈæ↗c')
        self.assertNotEqual(tuong, thuong)

    def test_phonemize_text_preserves_unaccented_t_and_th_onsets(self):
        teo = phonemize_text('teo')
        theo = phonemize_text('theo')

        self.assertEqual(teo, 'tˈɛw')
        self.assertEqual(theo, 'θˈɛw')
        self.assertNotEqual(teo, theo)

    def test_phonemize_text_preserves_vig2p_e_dash_vowel(self):
        self.assertEqual(phonemize_text('cách'), 'kˈæ↗c')
        self.assertEqual(phonemize_text('kếch'), 'kˈe↗c')
        self.assertEqual(phonemize_text('mạnh'), 'mˈæʔ↓ɲ')
        self.assertEqual(phonemize_text('mệnh'), 'mˈeʔ↓ɲ')

    def test_phonemize_text_does_not_rewrite_english_t_as_vietnamese_th(self):
        phonemes = phonemize_text("team start heart don't")

        self.assertIn('tˈiːm', phonemes)
        self.assertIn('stˈɑːɹt', phonemes)
        self.assertIn('hˈɑːɹt', phonemes)
        self.assertIn('dˈoʊnt', phonemes)
        self.assertNotIn('θˈiːm', phonemes)
        self.assertNotIn('sθˈɑːɹθ', phonemes)

    def test_phonemize_text_normalizes_mixed_text_punctuation(self):
        phonemes = phonemize_text("Mình cần **budget** UI/UX & don’t panic – thường.")

        self.assertNotIn('*', phonemes)
        self.assertNotIn('/', phonemes)
        self.assertNotIn('&', phonemes)
        self.assertNotIn('’', phonemes)
        self.assertIn('dˈoʊnt', phonemes)
        self.assertIn('—', phonemes)
        self.assertIn('θ', phonemes)

    def test_phonemize_text_preserves_vietnamese_orthographic_onset_contrasts(self):
        pairs = [
            ('trước', 'chước'),
            ('trị', 'chị'),
            ('số', 'xố'),
            ('sử', 'xử'),
            ('sinh', 'xinh'),
            ('sao', 'xao'),
            ('giải', 'dải'),
            ('gì', 'dì'),
            ('gia', 'da'),
        ]

        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertNotEqual(phonemize_text(left), phonemize_text(right))

        self.assertEqual(phonemize_text('trước'), 'ʈʂˈyə↗c')
        self.assertEqual(phonemize_text('chước'), 'ʧˈyə↗c')
        self.assertEqual(phonemize_text('số'), 'ʂˈo↗')
        self.assertEqual(phonemize_text('xố'), 'sˈo↗')
        self.assertEqual(phonemize_text('sinh'), 'ʂˈiɲ')
        self.assertEqual(phonemize_text('xinh'), 'sˈiɲ')
        self.assertEqual(phonemize_text('sao'), 'ʂˈaːw')
        self.assertEqual(phonemize_text('xao'), 'sˈaːw')
        self.assertEqual(phonemize_text('giải'), 'ʝˈaː↓j')
        self.assertEqual(phonemize_text('dải'), 'zˈaː↓j')
        self.assertEqual(phonemize_text('gia'), 'ʝˈaː')
        self.assertEqual(phonemize_text('da'), 'zˈaː')

    def test_assert_vocab_compatible_rejects_unknown_symbols(self):
        with self.assertRaisesRegex(ValueError, 'Unknown phoneme symbols'):
            assert_vocab_compatible('mˈaːɓ')

    def test_fix_vi_phonemes_maps_retroflex_r_from_spelled_abbreviations(self):
        fixed = fix_vi_phonemes('vˌeˈɛʐəː2')

        self.assertEqual(fixed, 'vˌeˈɛʒəː↘')

    def test_read_metadata_accepts_no_limit_for_duration_target_runs(self):
        with TemporaryDirectory() as tmp:
            metadata = Path(tmp) / 'metadata.csv'
            metadata.write_text(
                'filename|text\n'
                'audio/000001.wav|Một\n'
                'audio/000002.wav|Hai\n'
                'audio/000003.wav|Ba\n',
                encoding='utf-8',
            )

            rows = read_metadata(metadata, None)

        self.assertEqual(
            [(row.source_rel, row.text, row.speaker_label) for row in rows],
            [
                ('audio/000001.wav', 'Một', '0'),
                ('audio/000002.wav', 'Hai', '0'),
                ('audio/000003.wav', 'Ba', '0'),
            ],
        )

    def test_read_metadata_accepts_explicit_speaker_column(self):
        with TemporaryDirectory() as tmp:
            metadata = Path(tmp) / 'metadata.csv'
            metadata.write_text(
                'filename|text|speaker\n'
                'audio/a.wav|Xin chào|alice\n'
                'audio/b.wav|Tạm biệt|bob\n',
                encoding='utf-8',
            )

            rows = read_metadata(metadata, None, default_speaker='fallback')

        self.assertEqual(
            [(row.source_rel, row.text, row.speaker_label) for row in rows],
            [
                ('audio/a.wav', 'Xin chào', 'alice'),
                ('audio/b.wav', 'Tạm biệt', 'bob'),
            ],
        )

    def test_prepare_preserves_speaker_ids_and_writes_registry(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / 'source'
            (source_root / 'audio').mkdir(parents=True)
            for name in ['a.wav', 'b.wav', 'c.wav', 'd.wav']:
                (source_root / 'audio' / name).write_bytes(b'wav')
            metadata = root / 'metadata.csv'
            metadata.write_text(
                'filename|text|speaker\n'
                'audio/a.wav|Một|alice\n'
                'audio/b.wav|Hai|bob\n'
                'audio/c.wav|Ba|alice\n'
                'audio/d.wav|Bốn|bob\n',
                encoding='utf-8',
            )

            original_create = prep.create_phonemizer
            original_phonemize = prep.phonemize_text
            original_probe = prep.probe_wav
            original_materialize = prep.materialize_audio
            try:
                prep.create_phonemizer = lambda: object()
                prep.phonemize_text = lambda text, g2p=None: 'mˈaː→'
                prep.probe_wav = lambda path: (24000, 1, 1.0)
                def fake_materialize(source, dest, sample_rate, channels):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(b'wav')

                prep.materialize_audio = fake_materialize

                prep.prepare(
                    Namespace(
                        source_root=source_root,
                        metadata=metadata,
                        limit=None,
                        target_duration_hours=None,
                        output_dataset=root / 'dataset',
                        output_training=root / 'training',
                        speaker_id=0,
                        val_ratio=0.5,
                        seed=1,
                        min_tokens=1,
                        max_tokens=510,
                        min_duration=0.1,
                        max_duration=30.0,
                        clean=True,
                    )
                )
            finally:
                prep.create_phonemizer = original_create
                prep.phonemize_text = original_phonemize
                prep.probe_wav = original_probe
                prep.materialize_audio = original_materialize

            train_lines = (root / 'training/train_list.txt').read_text(encoding='utf-8').splitlines()
            val_lines = (root / 'training/val_list.txt').read_text(encoding='utf-8').splitlines()
            speakers_json = (root / 'training/speakers.json').read_text(encoding='utf-8')

        all_speaker_ids = {line.rsplit('|', 1)[1] for line in train_lines + val_lines}
        self.assertEqual(all_speaker_ids, {'0', '1'})
        self.assertIn('"alice": 0', speakers_json)
        self.assertIn('"bob": 1', speakers_json)

    def test_split_entries_keeps_each_speaker_in_train_and_val_when_possible(self):
        entries = [
            Entry(f'audio/a{i}.wav', 'Một', f'audio/a{i}.wav', 'mˈaː→', 1.0, 'alice', 0)
            for i in range(3)
        ] + [
            Entry(f'audio/b{i}.wav', 'Hai', f'audio/b{i}.wav', 'mˈaː→', 1.0, 'bob', 1)
            for i in range(3)
        ]

        train_entries, val_entries = prep.split_entries(entries, val_ratio=0.5, seed=1)

        self.assertEqual({entry.speaker_id for entry in train_entries}, {0, 1})
        self.assertEqual({entry.speaker_id for entry in val_entries}, {0, 1})

    def test_resolve_row_limit_reads_all_rows_when_target_duration_is_set(self):
        self.assertIsNone(resolve_row_limit(None, 30.0))
        self.assertIsNone(resolve_row_limit(None, None))
        self.assertEqual(resolve_row_limit(12000, 30.0), 12000)

    def test_reached_target_duration_compares_seconds_to_hours(self):
        self.assertFalse(reached_target_duration(107_999.9, 30.0))
        self.assertTrue(reached_target_duration(108_000.0, 30.0))
        self.assertFalse(reached_target_duration(108_000.0, None))


if __name__ == '__main__':
    unittest.main()
