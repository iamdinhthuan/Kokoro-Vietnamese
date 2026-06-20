# StyleTTS2 (Kokoro-82M Fine-tuning Fork)

> **Note:** This fork's `main` branch is a patched version of the original [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) repository. It is maintained specifically as a git submodule for the **[kokoro-deutsch](https://github.com/semidark/kokoro-deutsch)** training recipe.

## Why this fork exists

The [Kokoro-82M](https://github.com/hexgrad/kokoro) TTS model is based on the StyleTTS 2 architecture. However, to fine-tune it using its published HuggingFace weights, several modifications to the upstream training code are required:

1. **PyTorch API Migration (Critical):** Migrated `torch.nn.utils.weight_norm` and `spectral_norm` to the modern `torch.nn.utils.parametrizations` API. This is mandatory for compatibility with Kokoro's inference pipeline (`KModel`).
2. **Kokoro Symbols:** Integrated Kokoro's specific 178-token IPA vocabulary (`kokoro_symbols.py`).
3. **Bug Fixes:**
   - Fixed an `unsqueeze` shape mismatch crash at epoch boundaries involving F0 tensors.
   - Fixed checkpoint saving order to prevent data loss if TensorBoard audio generation fails.
   - Fixed missing `.train()` mode re-initializations after checkpoint loading in Stage 2.
   - Removed hardcoded `ipdb` breakpoints that caused silent hangs.
   - Added a monkey-patch for `torch.load` `weights_only=False` for PyTorch 2.6+ compatibility.
   - Filtered long phoneme sequences (> 510 tokens) to prevent PLBERT position embedding overflows.

## Usage

This repository is not meant to be used standalone.

Please see the **[kokoro-deutsch](https://github.com/semidark/kokoro-deutsch)** repository for the full end-to-end training guide, dataset preparation scripts, and voicepack extraction tools.

---
*For the original StyleTTS2 project and documentation, please visit the [upstream repository](https://github.com/yl4579/StyleTTS2).*
