import unittest

import numpy as np

from kokoro_vietnamese import (
    DEFAULT_CONFIG_FILE,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILE,
    DEFAULT_VOICEPACK_FILE,
    merge_audio_chunks,
    phonemize,
    split_text,
)


class KokoroVietnameseInferenceTest(unittest.TestCase):
    def test_defaults_point_to_public_artifacts(self):
        self.assertEqual(DEFAULT_HF_REPO_ID, 'contextboxai/Kokoro-Vietnamese')
        self.assertEqual(DEFAULT_MODEL_FILE, 'kokoro_vi.pth')
        self.assertEqual(DEFAULT_VOICEPACK_FILE, 'kokoro_vi_voicepack.pt')
        self.assertEqual(DEFAULT_CONFIG_FILE, 'config.json')

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


if __name__ == '__main__':
    unittest.main()
