import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class VietnameseMultispeakerTrainingTest(unittest.TestCase):
    def test_multispeaker_configs_are_opt_in_and_enabled(self):
        stage1 = REPO_ROOT / 'configs/config_vietnamese_30h_multispeaker_stage1.yml'
        stage2 = REPO_ROOT / 'configs/config_vietnamese_30h_multispeaker_stage2.yml'

        self.assertTrue(stage1.exists())
        self.assertTrue(stage2.exists())

        stage1_config = yaml.safe_load(stage1.read_text(encoding='utf-8'))
        stage2_config = yaml.safe_load(stage2.read_text(encoding='utf-8'))

        self.assertTrue(stage1_config['model_params']['multispeaker'])
        self.assertTrue(stage2_config['model_params']['multispeaker'])
        self.assertIn('ms', stage1_config['log_dir'])
        self.assertIn('ms', stage2_config['log_dir'])

    def test_train_second_initializes_multispeaker_ref_when_diffusion_is_disabled(self):
        train_second = (REPO_ROOT / 'StyleTTS2/train_second.py').read_text(encoding='utf-8')

        self.assertIn('ref = None', train_second)
        self.assertIn('ref if ref is not None else None', train_second)


if __name__ == '__main__':
    unittest.main()
