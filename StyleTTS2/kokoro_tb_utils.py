"""Shared TensorBoard inference helpers for Kokoro-faithful audio previews.

Used by both train_first.py (Stage 1) and train_second.py (Stage 2) to generate
TensorBoard audio that matches actual Kokoro inference output — predicted duration,
F0, and energy from a voicepack, not ground-truth reconstruction.
"""

import logging
import random
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from vig2p import VI_FIXUPS as VI_PHONEME_FIXUPS

logger = logging.getLogger(__name__)

# Vietnamese phonetic test sentences for Kokoro-faithful training previews.
TEST_SENTENCES = [
    "Ma mà má mả mã mạ là sáu thanh điệu khác nhau trong tiếng Việt.",
    "Xin chào, hôm nay trời đẹp quá, chúng ta cùng luyện giọng nói tự nhiên.",
    "Tôi đang kiểm tra khả năng đọc đúng dấu hỏi, dấu ngã và dấu nặng.",
    "Một trăm hai mươi ba người đã đặt mua sản phẩm mới vào sáng thứ hai.",
    "Bạn có thể nói chậm hơn một chút được không?",
    "Cô ấy sống ở Hà Nội nhưng thường bay vào Thành phố Hồ Chí Minh làm việc.",
    "Chiếc điện thoại này có giá mười lăm triệu đồng và được bảo hành hai năm.",
    "Nếu trời mưa lớn, chúng ta sẽ dời cuộc họp sang chiều mai.",
    "Những âm cuối như anh, ang, ach, ac, am và ap cần được phát âm rõ.",
    "Dữ liệu sạch giúp mô hình học ngữ điệu và nhịp đọc ổn định hơn.",
    "Tại sao bạn lại chọn hướng nghiên cứu tổng hợp tiếng nói cho tiếng Việt?",
    "Đây là bài kiểm tra cuối cùng trước khi tạo voicepack cho giọng tiếng Việt.",
]


TEXT_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*|\s+|.", re.UNICODE)
WORD_RE = re.compile(r"^[A-Za-zÀ-ỹĐđ]+(?:[-'][A-Za-zÀ-ỹĐđ]+)*$", re.UNICODE)
VI_MARK_RE = re.compile(r"[À-ỹĐđ]")
NON_VI_S_CLUSTERS = ("sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw")


def fix_vi_phonemes(phonemes, source_text=None):
    for old, new in VI_PHONEME_FIXUPS:
        phonemes = phonemes.replace(old, new)
    if source_text:
        source_lower = source_text.lower()
        if source_lower.startswith("th"):
            phonemes = phonemes.replace("t", "θ", 1)
        elif source_lower.startswith("tr"):
            phonemes = phonemes.replace("ʧ", "ʈʂ", 1)
        elif source_lower.startswith("s") and not source_lower.startswith(NON_VI_S_CLUSTERS):
            phonemes = phonemes.replace("s", "ʂ", 1)
        elif source_lower.startswith("gi") or re.match(r"^g[iìíỉĩị]", source_lower):
            phonemes = phonemes.replace("z", "ʝ", 1)
    return phonemes


def phonemize_vi_text(text, g2p):
    return g2p.many([text])[0]


def prepare_test_tokens(text_cleaner):
    """Convert Vietnamese test sentences to token ID lists via vig2p.

    Returns a list of (display_text, token_ids) tuples. Sentences that
    fail G2P or produce sequences longer than 510 tokens are skipped.
    """
    try:
        from vig2p import VietnameseG2P

        g2p = VietnameseG2P()
    except Exception as e:
        logger.warning(f"Could not load vig2p for TensorBoard inference: {e}")
        return []

    result = []
    for text in TEST_SENTENCES:
        try:
            ipa = phonemize_vi_text(text, g2p)
            token_ids = text_cleaner(ipa)
            if not token_ids or len(token_ids) > 510:
                logger.warning(
                    f"Skipping test sentence (token length {len(token_ids)}): {text[:40]}"
                )
                continue
            result.append((text, token_ids))
        except Exception as e:
            logger.warning(f'G2P failed for test sentence "{text[:40]}": {e}')
    return result


def extract_voicepack(model, root_path, device, n_samples=200):
    """Extract a mini voicepack from audio files — mirrors extract_voicepack.py.

    Randomly samples up to n_samples WAV files from root_path, computes
    mel spectrograms with the same params as meldataset.py, runs them through
    model.style_encoder and model.predictor_encoder, and averages the results.

    Returns:
        voicepack: torch.FloatTensor [256] — combined acoustic+prosodic style
        acoustic_norm: float — output norm of style_encoder (health check)
        prosodic_norm: float — output norm of predictor_encoder (health check)
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(device)
    mel_mean, mel_std = -4, 4

    wav_files = list(Path(root_path).rglob("*.wav"))
    if not wav_files:
        logger.warning(f"extract_voicepack: no WAV files found in {root_path}")
        return None, 0.0, 0.0

    random.shuffle(wav_files)
    wav_files = wav_files[:n_samples]

    acoustic_styles = []
    prosodic_styles = []

    with torch.no_grad():
        for wav_path in wav_files:
            try:
                data, file_sr = sf.read(str(wav_path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                waveform = torch.from_numpy(data).unsqueeze(0)
                if file_sr != 24000:
                    waveform = torchaudio.functional.resample(waveform, file_sr, 24000)
                waveform = waveform.to(device)
                mel = mel_transform(waveform)
                mel = (torch.log(1e-5 + mel) - mel_mean) / mel_std
                if mel.shape[-1] < 80:
                    continue
                mel_input = mel.unsqueeze(1)  # [1, 1, 80, T]
                acoustic_styles.append(model.style_encoder(mel_input).cpu())
                prosodic_styles.append(model.predictor_encoder(mel_input).cpu())
            except Exception as e:
                logger.warning(f"extract_voicepack: skipping {wav_path.name}: {e}")

    if not acoustic_styles:
        logger.warning("extract_voicepack: no valid audio files processed")
        return None, 0.0, 0.0

    avg_acoustic = torch.cat(acoustic_styles, dim=0).mean(dim=0)  # [128]
    avg_prosodic = torch.cat(prosodic_styles, dim=0).mean(dim=0)  # [128]
    voicepack = torch.cat([avg_acoustic, avg_prosodic], dim=0)  # [256]

    acoustic_norm = avg_acoustic.norm().item()
    prosodic_norm = avg_prosodic.norm().item()

    return voicepack.to(device), acoustic_norm, prosodic_norm


def run_kokoro_inference(model, test_tokens, voicepack, device, text_cleaner):
    """Run Kokoro-faithful inference for all test sentences.

    Mirrors KModel.forward_with_tokens exactly:
      - Predict duration from the predictor
      - Build alignment from predicted duration
      - Predict F0 and energy
      - Decode audio using voicepack acoustic style

    Returns a list of (display_text, audio_numpy) tuples.
    """
    if voicepack is None or not test_tokens:
        return []

    # voicepack [256]: first 128 = acoustic (decoder), last 128 = prosodic (predictor)
    ref_acoustic = voicepack[:128].unsqueeze(0)  # [1, 128]
    ref_prosodic = voicepack[128:].unsqueeze(0)  # [1, 128]

    results = []
    with torch.no_grad():
        for text, token_ids in test_tokens:
            try:
                # Build input token tensor with BOS/EOS (token 0)
                input_ids = torch.LongTensor([[0, *token_ids, 0]]).to(device)
                input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
                text_mask = torch.gt(
                    torch.arange(input_lengths.max())
                    .unsqueeze(0)
                    .expand(1, -1)
                    .type_as(input_lengths)
                    + 1,
                    input_lengths.unsqueeze(1),
                ).to(device)

                # BERT + encoder
                bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
                d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

                # Predict duration
                s_prosodic = ref_prosodic  # [1, 128]
                d = model.predictor.text_encoder(
                    d_en, s_prosodic, input_lengths, text_mask
                )
                x, _ = model.predictor.lstm(d)
                duration = model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration.squeeze()).clamp(min=1).long()
                if pred_dur.dim() == 0:
                    pred_dur = pred_dur.unsqueeze(0)

                # Build alignment matrix from predicted duration
                n_tokens = input_ids.shape[1]
                total_frames = int(pred_dur.sum().item())
                if total_frames == 0:
                    continue
                pred_aln_trg = torch.zeros(n_tokens, total_frames).to(device)
                c_frame = 0
                for i in range(n_tokens):
                    dur_i = int(pred_dur[i].item())
                    pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
                    c_frame += dur_i
                pred_aln_trg = pred_aln_trg.unsqueeze(0)  # [1, n_tokens, total_frames]

                # Predict F0 and energy
                en = d.transpose(-1, -2) @ pred_aln_trg
                F0_pred, N_pred = model.predictor.F0Ntrain(en, s_prosodic)

                # Text encoder + decode
                t_en = model.text_encoder(input_ids, input_lengths, text_mask)
                asr = t_en @ pred_aln_trg
                audio = model.decoder(asr, F0_pred, N_pred, ref_acoustic)
                results.append((text, audio.squeeze().cpu().numpy()))
            except Exception as e:
                logger.warning(f'Inference failed for "{text[:40]}": {e}')

    return results
