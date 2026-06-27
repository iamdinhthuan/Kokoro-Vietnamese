import unittest
import tomllib
from pathlib import Path

import numpy as np

from kokoro_vietnamese import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILE,
    DEFAULT_NORMALIZE_PEAK,
    DEFAULT_ONNX_FILE,
    DEFAULT_VOICE,
    DEFAULT_VOICEPACK_FILE,
    VOICES,
    list_voices,
    merge_audio_chunks,
    normalize_audio,
    phonemize,
    prepare_transformers_for_kokoro,
    resolve_voicepack_filename,
    split_text,
)
from kokoro_vietnamese.gradio_app import DEMO_EXAMPLES, build_voice_choices


class KokoroVietnameseInferenceTest(unittest.TestCase):
    def test_defaults_point_to_public_artifacts(self):
        self.assertEqual(DEFAULT_HF_REPO_ID, 'contextboxai/Kokoro-Vietnamese')
        self.assertEqual(DEFAULT_MODEL_FILE, 'kokoro_vi.pth')
        self.assertEqual(DEFAULT_ONNX_FILE, 'kokoro_vi.onnx')
        self.assertEqual(DEFAULT_VOICEPACK_FILE, 'kokoro_vi_voicepack.pt')
        self.assertEqual(DEFAULT_CONFIG_FILE, 'config.json')
        self.assertEqual(DEFAULT_VOICE, 'diem_trinh')
        self.assertIsNone(DEFAULT_NORMALIZE_PEAK)

    def test_voice_registry_covers_all_larvoice_speakers_without_pro_suffix(self):
        expected = {
            'diem_trinh',
            'hung_thinh',
            'mai_linh',
            'mai_loan',
            'manh_dung',
            'my_yen',
            'ngoc_huyen',
            'phat_tai',
            'thanh_dat',
            'thuc_trinh',
            'tuan_ngoc',
            'storyvert',
            'duc_an',
            'duc_duy',
        }

        self.assertEqual(set(VOICES), expected)
        self.assertEqual(list_voices(), sorted(expected))
        for name, info in VOICES.items():
            self.assertNotIn('pro', name)
            self.assertNotIn('Pro', info['label'])
            self.assertTrue(info['filename'].startswith('voicepacks/'))
            self.assertTrue(info['filename'].endswith('.pt'))

    def test_resolve_voicepack_filename_accepts_named_voices_and_explicit_paths(self):
        self.assertEqual(resolve_voicepack_filename('mai_linh', None), 'voicepacks/mai_linh.pt')
        self.assertEqual(resolve_voicepack_filename(None, 'local.pt'), 'local.pt')
        with self.assertRaises(ValueError):
            resolve_voicepack_filename('missing_voice', None)

    def test_split_text_keeps_sentence_punctuation(self):
        self.assertEqual(
            split_text('Xin chào. Bạn khỏe không? Tôi ổn! Dòng mới\nvẫn chạy.'),
            ['Xin chào.', 'Bạn khỏe không?', 'Tôi ổn!', 'Dòng mới vẫn chạy.'],
        )

    def test_merge_audio_chunks_crossfades(self):
        first = np.ones(10, dtype=np.float32)
        second = np.zeros(10, dtype=np.float32)

        merged = merge_audio_chunks([first, second], crossfade_samples=4)

        self.assertEqual(len(merged), 16)
        np.testing.assert_allclose(merged[:6], np.ones(6, dtype=np.float32))
        np.testing.assert_allclose(merged[6:10], np.array([0.8, 0.6, 0.4, 0.2], dtype=np.float32))
        np.testing.assert_allclose(merged[10:], np.zeros(6, dtype=np.float32))

    def test_normalize_audio_scales_only_when_peak_exceeds_limit(self):
        loud = np.array([-2.0, 0.5, 1.0], dtype=np.float32)
        normalized = normalize_audio(loud, peak=0.95)

        self.assertEqual(normalized.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(normalized))), 0.95)
        np.testing.assert_allclose(normalized, loud * 0.475)

        quiet = np.array([-0.5, 0.2, 0.7], dtype=np.float32)
        np.testing.assert_allclose(normalize_audio(quiet, peak=0.95), quiet)
        np.testing.assert_allclose(normalize_audio(loud, peak=None), loud)

    def test_vig2p_preserves_vietnamese_t_and_th_contrast(self):
        self.assertEqual(phonemize('Tường nhà khách.'), 'tˈyə↘ŋ ɲˈaː↘ xˈæ↗c.')
        self.assertEqual(phonemize('Thường nhà khách.'), 'θˈyə↘ŋ ɲˈaː↘ xˈæ↗c.')

    def test_gradio_has_multiple_vietnamese_demos_and_all_voices(self):
        self.assertGreaterEqual(len(DEMO_EXAMPLES), 10)
        for text, voice, speed in DEMO_EXAMPLES:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 20)
            self.assertIn(voice, VOICES)
            self.assertGreater(speed, 0)

        choices = build_voice_choices()
        self.assertEqual(len(choices), len(VOICES))
        self.assertIn(('Diễm Trinh', 'diem_trinh'), choices)

    def test_project_pins_transformers_for_kokoro_albert_model(self):
        pyproject = tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))
        dependencies = pyproject['project']['dependencies']
        self.assertIn('transformers>=4.48,<5', dependencies)
        self.assertIn('packaging', dependencies)
        self.assertEqual(pyproject['project']['optional-dependencies']['onnx'], ['onnx', 'onnxruntime'])

    def test_onnx_helpers_prepare_runtime_inputs(self):
        from kokoro_vietnamese.onnx_utils import phonemes_to_input_ids, select_voice_style, speed_input

        np.testing.assert_array_equal(
            phonemes_to_input_ids('ab?', {'a': 11, 'b': 12}, context_length=8),
            np.asarray([[0, 11, 12, 0]], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, 'Phoneme sequence too long'):
            phonemes_to_input_ids('abcd', {'a': 1, 'b': 2, 'c': 3, 'd': 4}, context_length=5)

        voicepack = np.zeros((4, 1, 256), dtype=np.float32)
        voicepack[0, :, :] = 1
        voicepack[2, :, :] = 3
        voicepack[3, :, :] = 4
        np.testing.assert_array_equal(select_voice_style(voicepack, 1), voicepack[0])
        np.testing.assert_array_equal(select_voice_style(voicepack, 3), voicepack[2])
        np.testing.assert_array_equal(select_voice_style(voicepack, 999), voicepack[3])
        np.testing.assert_array_equal(speed_input(1.25), np.asarray(1.25, dtype=np.float32))

    def test_onnx_cli_defaults(self):
        from kokoro_vietnamese.export_onnx import DEFAULT_ONNX_FILE as EXPORT_ONNX_FILE
        from kokoro_vietnamese.export_onnx import build_parser as build_export_parser
        from kokoro_vietnamese.onnx_cli import build_parser as build_onnx_parser

        self.assertEqual(EXPORT_ONNX_FILE, 'kokoro_vi.onnx')
        self.assertEqual(build_export_parser().parse_args([]).output, 'outputs/kokoro_vi.onnx')
        self.assertEqual(build_onnx_parser().parse_args(['--text', 'Xin chào']).device, 'cpu')

    def test_transformers_preflight_imports_albert_model(self):
        prepare_transformers_for_kokoro()
        from transformers import AlbertModel

        self.assertIsNotNone(AlbertModel)

    def test_uses_vendored_kokoro_runtime(self):
        from kokoro_vietnamese._kokoro import KModel

        self.assertEqual(KModel.__module__, 'kokoro_vietnamese._kokoro.model')


if __name__ == '__main__':
    unittest.main()
