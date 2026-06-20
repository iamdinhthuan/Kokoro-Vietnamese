import unittest
from pathlib import Path

import yaml

from scripts.modal_vietnamese_100h_train import make_remote_train_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class ModalVietnamese100hTrainTest(unittest.TestCase):
    def test_make_remote_train_config_points_at_volume_paths_and_batch(self):
        base = yaml.safe_load(
            (REPO_ROOT / 'configs/config_vietnamese_100h_multispeaker_stage1.yml')
            .read_text(encoding='utf-8')
        )

        config = make_remote_train_config(base, batch_size=20, epochs=12)

        self.assertEqual(config['batch_size'], 20)
        self.assertEqual(config['epochs'], 12)
        self.assertEqual(config['epochs_1st'], 12)
        self.assertEqual(config['log_dir'], '/data/work/logs/kokoro-vi-100h-ms')
        self.assertEqual(config['pretrained_model'], '/data/training/kokoro_base.pth')
        self.assertEqual(config['data_params']['root_path'], '/data/source')
        self.assertEqual(config['data_params']['train_data'], '/data/training/vi_100h/train_list.txt')
        self.assertEqual(config['data_params']['val_data'], '/data/training/vi_100h/val_list.txt')
        self.assertEqual(config['data_params']['OOD_data'], '/data/training/vi_100h/OOD_texts.txt')
        self.assertTrue(config['model_params']['multispeaker'])


if __name__ == '__main__':
    unittest.main()
