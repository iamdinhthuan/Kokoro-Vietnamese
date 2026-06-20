import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from scripts.gradio_vietnamese_checkpoint import (
    VietnameseCheckpointPredictor,
    converted_model_path,
    find_best_checkpoint,
    find_latest_checkpoint,
    merge_audio_chunks_with_crossfade,
    needs_refresh,
    speaker_choices,
    stage2_checkpoint_choices,
    split_text_for_prediction,
    voicepack_path,
)


class GradioVietnameseCheckpointTest(unittest.TestCase):
    def test_find_latest_checkpoint_uses_newest_saved_pth(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            old = log_dir / 'epoch_1st_00001.pth'
            new = log_dir / 'epoch_1st_00004.pth'
            old.write_bytes(b'old')
            new.write_bytes(b'new')
            os.utime(old, (time.time() - 20, time.time() - 20))
            os.utime(new, (time.time(), time.time()))

            self.assertEqual(find_latest_checkpoint(log_dir), new)

    def test_find_latest_checkpoint_prefers_latest_stage2_over_newer_stage1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            stage2 = log_dir / 'epoch_2nd_00000.pth'
            stage1 = log_dir / 'first_stage.pth'
            stage2.write_bytes(b'stage2')
            stage1.write_bytes(b'stage1')
            os.utime(stage2, (time.time() - 20, time.time() - 20))
            os.utime(stage1, (time.time(), time.time()))

            self.assertEqual(find_latest_checkpoint(log_dir), stage2)

    def test_find_latest_checkpoint_can_filter_to_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            stage2 = log_dir / 'epoch_2nd_00000.pth'
            stage1 = log_dir / 'first_stage.pth'
            stage2.write_bytes(b'stage2')
            stage1.write_bytes(b'stage1')
            os.utime(stage2, (time.time(), time.time()))
            os.utime(stage1, (time.time() - 20, time.time() - 20))

            self.assertEqual(find_latest_checkpoint(log_dir, stage=1), stage1)

    def test_latest_strategy_uses_latest_stage1_style_encoder_for_stage2(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            stage2 = log_dir / 'epoch_2nd_00000.pth'
            best_stage1 = log_dir / 'epoch_1st_00003.pth'
            final_stage1 = log_dir / 'first_stage.pth'
            for checkpoint in [stage2, best_stage1, final_stage1]:
                checkpoint.write_bytes(b'checkpoint')
            os.utime(best_stage1, (time.time() - 30, time.time() - 30))
            os.utime(stage2, (time.time() - 20, time.time() - 20))
            os.utime(final_stage1, (time.time(), time.time()))

            predictor = VietnameseCheckpointPredictor(
                log_dir=log_dir,
                cache_dir=Path(tmp) / 'cache',
                audio_dir=Path(tmp) / 'audio',
                metadata_path=None,
                speakers_path=None,
                config_path=Path(tmp) / 'config.json',
                device='cpu',
                num_voice_samples=1,
                checkpoint_strategy='latest',
            )

            checkpoint_path, style_encoder_checkpoint, strategy = predictor._select_checkpoint()

            self.assertEqual(strategy, 'latest')
            self.assertEqual(checkpoint_path, stage2)
            self.assertEqual(style_encoder_checkpoint, final_stage1)

    def test_stage2_checkpoint_choices_lists_stage2_checkpoints_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            stage1 = log_dir / 'epoch_1st_00009.pth'
            stage2_old = log_dir / 'epoch_2nd_00000.pth'
            stage2_new = log_dir / 'epoch_2nd_00007.pth'
            for checkpoint in [stage1, stage2_old, stage2_new]:
                checkpoint.write_bytes(b'checkpoint')

            self.assertEqual(
                stage2_checkpoint_choices(log_dir),
                ['epoch_2nd_00007.pth', 'epoch_2nd_00000.pth'],
            )

    def test_select_checkpoint_accepts_explicit_stage2_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            selected_stage2 = log_dir / 'epoch_2nd_00003.pth'
            latest_stage2 = log_dir / 'epoch_2nd_00007.pth'
            final_stage1 = log_dir / 'first_stage.pth'
            for checkpoint in [selected_stage2, latest_stage2, final_stage1]:
                checkpoint.write_bytes(b'checkpoint')
            os.utime(selected_stage2, (time.time() - 40, time.time() - 40))
            os.utime(latest_stage2, (time.time() - 20, time.time() - 20))
            os.utime(final_stage1, (time.time(), time.time()))

            predictor = VietnameseCheckpointPredictor(
                log_dir=log_dir,
                cache_dir=Path(tmp) / 'cache',
                audio_dir=Path(tmp) / 'audio',
                metadata_path=None,
                speakers_path=None,
                config_path=Path(tmp) / 'config.json',
                device='cpu',
                num_voice_samples=1,
                checkpoint_strategy='latest',
            )

            checkpoint_path, style_encoder_checkpoint, strategy = predictor._select_checkpoint(
                selected_checkpoint='epoch_2nd_00003.pth',
            )

            self.assertEqual(strategy, 'selected')
            self.assertEqual(checkpoint_path, selected_stage2)
            self.assertEqual(style_encoder_checkpoint, final_stage1)

    def test_find_best_checkpoint_prefers_best_stage2_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            stage1 = log_dir / 'epoch_1st_00016.pth'
            stage2_bad = log_dir / 'epoch_2nd_00000.pth'
            stage2_best = log_dir / 'epoch_2nd_00008.pth'
            for checkpoint in [stage1, stage2_bad, stage2_best]:
                checkpoint.write_bytes(b'checkpoint')

            losses = {
                stage1: {'val_loss': 0.18},
                stage2_bad: {'val_loss': 0.31},
                stage2_best: {'val_loss': 0.27},
            }

            self.assertEqual(
                find_best_checkpoint(log_dir, metadata_loader=losses.__getitem__),
                stage2_best,
            )

    def test_find_best_checkpoint_uses_best_stage1_when_no_stage2_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            worse = log_dir / 'epoch_1st_00018.pth'
            best = log_dir / 'epoch_1st_00016.pth'
            for checkpoint in [worse, best]:
                checkpoint.write_bytes(b'checkpoint')

            losses = {
                worse: {'val_loss': 0.19},
                best: {'val_loss': 0.18},
            }

            self.assertEqual(
                find_best_checkpoint(log_dir, metadata_loader=losses.__getitem__),
                best,
            )

    def test_find_best_checkpoint_falls_back_to_newest_when_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            old = log_dir / 'epoch_2nd_00000.pth'
            new = log_dir / 'epoch_2nd_00001.pth'
            old.write_bytes(b'old')
            new.write_bytes(b'new')
            os.utime(old, (time.time() - 20, time.time() - 20))
            os.utime(new, (time.time(), time.time()))

            self.assertEqual(
                find_best_checkpoint(log_dir, metadata_loader=lambda _: {}),
                new,
            )

    def test_find_best_checkpoint_skips_unreadable_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            good = log_dir / 'epoch_2nd_00000.pth'
            bad = log_dir / 'epoch_2nd_00001.pth'
            good.write_bytes(b'good')
            bad.write_bytes(b'bad')

            def metadata_loader(path: Path):
                if path == bad:
                    raise RuntimeError('checkpoint still being written')
                return {'val_loss': 0.25}

            self.assertEqual(find_best_checkpoint(log_dir, metadata_loader=metadata_loader), good)

    def test_needs_refresh_when_cache_missing_or_older_than_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'epoch_1st_00004.pth'
            cache = Path(tmp) / 'cache.pth'
            source.write_bytes(b'source')

            self.assertTrue(needs_refresh(source, cache))

            cache.write_bytes(b'cache')
            os.utime(source, (time.time(), time.time()))
            os.utime(cache, (time.time() - 20, time.time() - 20))
            self.assertTrue(needs_refresh(source, cache))

            os.utime(cache, (time.time() + 20, time.time() + 20))
            self.assertFalse(needs_refresh(source, cache))

    def test_converted_model_path_is_checkpoint_stem_in_cache_dir(self):
        checkpoint = Path('StyleTTS2/logs/kokoro-vi-30h/epoch_1st_00004.pth')

        self.assertEqual(
            converted_model_path(checkpoint, Path('gradio_cache')),
            Path('gradio_cache/epoch_1st_00004_kokoro.pth'),
        )

    def test_voicepack_path_includes_style_encoder_checkpoint_when_provided(self):
        checkpoint = Path('StyleTTS2/logs/kokoro-vi-30h/epoch_2nd_00008.pth')
        style_checkpoint = Path('StyleTTS2/logs/kokoro-vi-30h/epoch_1st_00016.pth')

        self.assertEqual(
            voicepack_path(checkpoint, Path('gradio_cache'), style_checkpoint),
            Path('gradio_cache/epoch_2nd_00008_style_epoch_1st_00016_voicepack.pt'),
        )

    def test_voicepack_path_includes_speaker_when_selected(self):
        checkpoint = Path('StyleTTS2/logs/kokoro-vi-30h/epoch_2nd_00008.pth')
        style_checkpoint = Path('StyleTTS2/logs/kokoro-vi-30h/first_stage.pth')

        self.assertEqual(
            voicepack_path(
                checkpoint,
                Path('gradio_cache'),
                style_checkpoint,
                speaker='giọng nữ miền bắc',
            ),
            Path('gradio_cache/epoch_2nd_00008_style_first_stage_speaker_giong_nu_mien_bac_voicepack.pt'),
        )

    def test_speaker_choices_loads_registry_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            speakers = Path(tmp) / 'speakers.json'
            speakers.write_text(
                '{"ids_by_label": {"alice": 0, "bob": 1}, "labels_by_id": {"0": "alice", "1": "bob"}}',
                encoding='utf-8',
            )

            self.assertEqual(speaker_choices(speakers), ['alice', 'bob'])

    def test_split_text_for_prediction_keeps_sentence_punctuation(self):
        self.assertEqual(
            split_text_for_prediction('Xin chào. Bạn khỏe không? Tôi ổn! Dòng mới\nvẫn chạy.'),
            ['Xin chào.', 'Bạn khỏe không?', 'Tôi ổn!', 'Dòng mới vẫn chạy.'],
        )

    def test_split_text_for_prediction_keeps_short_text_as_one_chunk(self):
        self.assertEqual(split_text_for_prediction('Không có dấu chấm'), ['Không có dấu chấm'])

    def test_merge_audio_chunks_with_crossfade_overlaps_adjacent_chunks(self):
        first = np.ones(10, dtype=np.float32)
        second = np.zeros(10, dtype=np.float32)

        merged = merge_audio_chunks_with_crossfade([first, second], crossfade_samples=4)

        self.assertEqual(len(merged), 16)
        np.testing.assert_allclose(merged[:6], np.ones(6, dtype=np.float32))
        np.testing.assert_allclose(merged[6:10], np.array([0.8, 0.6, 0.4, 0.2], dtype=np.float32))
        np.testing.assert_allclose(merged[10:], np.zeros(6, dtype=np.float32))

    def test_merge_audio_chunks_with_crossfade_handles_short_chunks(self):
        first = np.array([1.0, 1.0], dtype=np.float32)
        second = np.array([0.0, 0.0], dtype=np.float32)

        merged = merge_audio_chunks_with_crossfade([first, second], crossfade_samples=8)

        self.assertEqual(len(merged), 2)
        np.testing.assert_allclose(merged, np.array([2 / 3, 1 / 3], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
