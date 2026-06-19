import unittest

import numpy as np

from kokoro_vietnamese import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILE,
    DEFAULT_VOICE,
    DEFAULT_VOICEPACK_FILE,
    VOICES,
    list_voices,
    merge_audio_chunks,
    phonemize,
    resolve_voicepack_filename,
    split_text,
)
from kokoro_vietnamese.gradio_app import DEMO_EXAMPLES, build_voice_choices


class KokoroVietnameseInferenceTest(unittest.TestCase):
    def test_defaults_point_to_public_artifacts(self):
        self.assertEqual(DEFAULT_HF_REPO_ID, 'contextboxai/Kokoro-Vietnamese')
        self.assertEqual(DEFAULT_MODEL_FILE, 'kokoro_vi.pth')
        self.assertEqual(DEFAULT_VOICEPACK_FILE, 'kokoro_vi_voicepack.pt')
        self.assertEqual(DEFAULT_CONFIG_FILE, 'config.json')
        self.assertEqual(DEFAULT_VOICE, 'diem_trinh')

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


if __name__ == '__main__':
    unittest.main()
