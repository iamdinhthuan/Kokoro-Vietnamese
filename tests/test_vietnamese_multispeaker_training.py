import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class VietnameseMultispeakerTrainingTest(unittest.TestCase):
    def test_training_docs_preprocess_uses_all_metadata_by_default(self):
        training_doc = (REPO_ROOT / 'TRAINING.md').read_text(encoding='utf-8')

        self.assertNotIn('--target-duration-hours 30', training_doc)
        self.assertIn('uses every valid metadata row by default', training_doc)

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

    def test_plbert_disables_optional_transformers_backends_before_albert_import(self):
        plbert_util = (REPO_ROOT / 'StyleTTS2/Utils/PLBERT/util.py').read_text(encoding='utf-8')
        albert_import_index = plbert_util.index('from transformers import AlbertConfig, AlbertModel')

        self.assertLess(plbert_util.index('_torchvision_available'), albert_import_index)
        self.assertLess(plbert_util.index('_librosa_available'), albert_import_index)
        self.assertLess(plbert_util.index('_cv2_available'), albert_import_index)

    def test_ensure_pretrained_model_downloads_missing_base_checkpoint(self):
        from StyleTTS2.pretrained_utils import ensure_pretrained_model

        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'training/kokoro_base.pth'

            def fake_download(url, destination):
                destination.write_bytes(b'checkpoint')

            with patch('StyleTTS2.pretrained_utils._download_file', side_effect=fake_download) as download:
                resolved = ensure_pretrained_model(str(target))

            self.assertEqual(resolved, str(target))
            self.assertEqual(target.read_bytes(), b'checkpoint')
            download.assert_called_once()

    def test_ensure_pretrained_model_keeps_existing_checkpoint(self):
        from StyleTTS2.pretrained_utils import ensure_pretrained_model

        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / 'training/kokoro_base.pth'
            target.parent.mkdir(parents=True)
            target.write_bytes(b'existing')

            with patch('StyleTTS2.pretrained_utils._download_file') as download:
                resolved = ensure_pretrained_model(str(target))

            self.assertEqual(resolved, str(target))
            self.assertEqual(target.read_bytes(), b'existing')
            download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
