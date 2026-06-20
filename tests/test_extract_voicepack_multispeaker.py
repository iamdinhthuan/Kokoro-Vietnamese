import tempfile
import unittest
from pathlib import Path

from scripts.extract_voicepack import select_voicepack_audio_files


class ExtractVoicepackMultispeakerTest(unittest.TestCase):
    def test_select_voicepack_audio_files_filters_by_speaker_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            alice = audio_dir / 'a.wav'
            bob = audio_dir / 'b.wav'
            alice.write_bytes(b'audio')
            bob.write_bytes(b'audio')

            metadata = root / 'metadata.csv'
            metadata.write_text(
                'filename|text|speaker|speaker_label\n'
                'audio/a.wav|Xin chào|0|alice\n'
                'audio/b.wav|Tạm biệt|1|bob\n',
                encoding='utf-8',
            )

            self.assertEqual(
                select_voicepack_audio_files(audio_dir, metadata, speaker='alice'),
                [alice],
            )

    def test_select_voicepack_audio_files_filters_by_speaker_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / 'audio'
            audio_dir.mkdir()
            first = audio_dir / 'first.wav'
            second = audio_dir / 'second.wav'
            first.write_bytes(b'audio')
            second.write_bytes(b'audio')

            metadata = root / 'metadata.csv'
            metadata.write_text(
                'filename|text|speaker\n'
                'audio/first.wav|Một|0\n'
                'audio/second.wav|Hai|1\n',
                encoding='utf-8',
            )

            self.assertEqual(
                select_voicepack_audio_files(audio_dir, metadata, speaker='1'),
                [second],
            )


if __name__ == '__main__':
    unittest.main()
