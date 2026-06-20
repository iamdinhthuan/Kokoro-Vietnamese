import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KOKORO_ROOT = REPO_ROOT / 'kokoro'
STYLETTS2_ROOT = REPO_ROOT / 'StyleTTS2'
sys.path.insert(0, str(KOKORO_ROOT))
sys.path.insert(0, str(STYLETTS2_ROOT))

from kokoro import KPipeline  # noqa: E402
from kokoro.pipeline import VI_PHONEME_FIXUPS as KOKORO_VI_FIXUPS  # noqa: E402
from kokoro_tb_utils import prepare_test_tokens, TEST_SENTENCES  # noqa: E402
from kokoro_tb_utils import VI_PHONEME_FIXUPS as TB_VI_FIXUPS  # noqa: E402
from kokoro_symbols import TextCleaner  # noqa: E402
from scripts.prepare_vietnamese_dataset import VI_FIXUPS  # noqa: E402


class VietnameseFrontendTest(unittest.TestCase):
    def test_kpipeline_accepts_vietnamese_lang_code_and_uses_vig2p_fixups(self):
        pipeline = KPipeline(lang_code='v', model=False)

        results = list(pipeline('Xin chào, tôi thường đọc số trước. Tiếng ve râm ran.', voice=None))

        self.assertEqual(len(results), 1)
        phonemes = results[0].phonemes
        self.assertIn('ɹˈəm ɹˈaːn', phonemes)
        self.assertIn('↘', phonemes)
        self.assertIn('ʔ↓', phonemes)
        self.assertIn('θ', phonemes)
        self.assertIn('ʂ', phonemes)
        self.assertIn('ʈʂ', phonemes)
        self.assertNotIn('zˈəm zˈaːn', phonemes)
        self.assertNotIn('2', phonemes)
        self.assertNotIn('6', phonemes)
        self.assertNotIn('ɗ', phonemes)
        self.assertNotIn('̪', phonemes)
        self.assertNotIn('tʃ', phonemes)

    def test_kpipeline_preserves_unaccented_vietnamese_onset_contrasts(self):
        pipeline = KPipeline(lang_code='v', model=False)

        result = list(pipeline('teo theo. sinh xinh. sao xao. start style.', voice=None))[0]

        self.assertIn('tˈɛw θˈɛw', result.phonemes)
        self.assertIn('ʂˈiɲ sˈiɲ', result.phonemes)
        self.assertIn('ʂˈaːw sˈaːw', result.phonemes)
        self.assertIn('stˈɑːɹt stˈaɪl', result.phonemes)

    def test_vietnamese_fixups_stay_consistent_across_entrypoints(self):
        self.assertEqual(TB_VI_FIXUPS, VI_FIXUPS)
        self.assertEqual(KOKORO_VI_FIXUPS, VI_FIXUPS)

    def test_tensorboard_preview_uses_vietnamese_test_sentences(self):
        tokens = prepare_test_tokens(TextCleaner())

        self.assertTrue(TEST_SENTENCES[0].startswith('Ma mà má'))
        self.assertGreaterEqual(len(tokens), 10)
        for _, token_ids in tokens:
            self.assertGreater(len(token_ids), 0)
            self.assertLessEqual(len(token_ids), 510)


if __name__ == '__main__':
    unittest.main()
