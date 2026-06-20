# Kokoro-82M symbol mapping (replaces default StyleTTS2 symbols)
# CRITICAL: Kokoro and StyleTTS2 use different index assignments for the same
# 178-token vocabulary. Using the wrong mapping scrambles pre-trained embeddings.
from kokoro_symbols import symbols, dicts, TextCleaner
