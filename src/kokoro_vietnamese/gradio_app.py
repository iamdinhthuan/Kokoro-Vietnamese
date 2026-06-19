from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass

from .core import DEFAULT_CROSSFADE_MS, DEFAULT_HF_REPO_ID, DEFAULT_VOICE, SAMPLE_RATE, VOICES, KokoroVietnamese


DEMO_EXAMPLES = [
    [
        'Giữa một buổi chiều yên tĩnh, cô ấy kể lại câu chuyện bằng một giọng nói ấm áp và chậm rãi.',
        'diem_trinh',
        1.0,
    ],
    [
        'Sáng nay, thành phố thức dậy trong làn sương mỏng, còn những con đường thì bắt đầu rộn ràng tiếng xe.',
        'mai_linh',
        1.0,
    ],
    [
        'Nếu bạn lắng nghe thật kỹ, bạn sẽ nghe thấy tiếng mưa rơi nhẹ trên mái hiên sau nhà.',
        'ngoc_huyen',
        0.95,
    ],
    [
        'Bản tin hôm nay ghi nhận nhiều tín hiệu tích cực từ thị trường, đặc biệt là nhóm công nghệ và tiêu dùng.',
        'hung_thinh',
        1.03,
    ],
    [
        'Trong căn phòng nhỏ, người dẫn chuyện mỉm cười rồi bắt đầu đọc chương đầu tiên của cuốn sách.',
        'thuc_trinh',
        0.98,
    ],
    [
        'Hành trình qua miền Trung để lại trong tôi ký ức về nắng, gió, biển xanh và những bữa cơm rất đậm đà.',
        'tuan_ngoc',
        1.0,
    ],
    [
        'Chúng ta sẽ kiểm tra từng câu, từng dấu ngắt, để giọng đọc nghe tự nhiên hơn khi ghép thành đoạn dài.',
        'mai_loan',
        1.0,
    ],
    [
        'Ở góc vườn, mấy chậu hoa mới nở làm cả khoảng sân trở nên sáng hơn sau nhiều ngày mưa lạnh.',
        'my_yen',
        0.97,
    ],
    [
        'Đội kỹ thuật đã hoàn thành bản cập nhật mới, giúp hệ thống phản hồi nhanh và ổn định hơn trước.',
        'manh_dung',
        1.02,
    ],
    [
        'Cậu bé nhìn lên bầu trời đêm, chỉ tay vào ngôi sao sáng nhất và hỏi về những chuyến du hành rất xa.',
        'phat_tai',
        0.96,
    ],
    [
        'Một podcast hay không chỉ cần nội dung tốt, mà còn cần nhịp kể đủ cuốn hút để giữ người nghe ở lại.',
        'storyvert',
        1.0,
    ],
    [
        'Khi đọc truyện thiếu nhi, giọng nói nên mềm hơn, vui hơn, nhưng vẫn rõ chữ và giữ đúng nhịp câu.',
        'duc_duy',
        1.0,
    ],
]


def build_voice_choices() -> list[tuple[str, str]]:
    return [(info['label'], name) for name, info in VOICES.items()]


@dataclass
class GradioSynthesizer:
    repo_id: str = DEFAULT_HF_REPO_ID
    device: str = 'cuda'

    def __post_init__(self) -> None:
        self._models: dict[str, KokoroVietnamese] = {}

    def _get_model(self, voice: str) -> KokoroVietnamese:
        if voice not in self._models:
            self._models[voice] = KokoroVietnamese(
                repo_id=self.repo_id,
                voice=voice,
                device=self.device,
            )
        return self._models[voice]

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float,
        crossfade_ms: int,
    ) -> tuple[tuple[int, object] | None, str, str]:
        if not text or not text.strip():
            return None, '', 'Nhập văn bản tiếng Việt trước.'
        try:
            model = self._get_model(voice)
            audio, phonemes = model.synthesize(
                text,
                speed=float(speed),
                crossfade_ms=int(crossfade_ms),
            )
        except Exception as exc:
            return None, traceback.format_exc(limit=6), f'Lỗi: {exc}'
        if len(audio) == 0:
            return None, phonemes, 'Không tạo được audio.'
        label = VOICES.get(voice, {}).get('label', voice)
        return (SAMPLE_RATE, audio), phonemes, f'Voice: {label}'


def build_demo(synthesizer: GradioSynthesizer):
    import gradio as gr

    with gr.Blocks(title='Kokoro Vietnamese') as demo:
        gr.Markdown('# Kokoro Vietnamese')
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label='Text',
                    lines=5,
                    value=DEMO_EXAMPLES[0][0],
                )
                with gr.Row():
                    voice = gr.Dropdown(
                        label='Voice',
                        choices=build_voice_choices(),
                        value=DEFAULT_VOICE,
                    )
                    speed = gr.Slider(0.75, 1.25, value=1.0, step=0.01, label='Speed')
                    crossfade = gr.Slider(
                        0,
                        120,
                        value=DEFAULT_CROSSFADE_MS,
                        step=5,
                        label='Crossfade ms',
                    )
                submit = gr.Button('Generate', variant='primary')
            with gr.Column(scale=1):
                audio = gr.Audio(label='Audio', type='numpy')
                status = gr.Textbox(label='Status', interactive=False)
        phonemes = gr.Textbox(label='Phonemes', lines=4, interactive=False)
        gr.Examples(
            examples=DEMO_EXAMPLES,
            inputs=[text, voice, speed],
        )
        submit.click(
            fn=synthesizer.synthesize,
            inputs=[text, voice, speed, crossfade],
            outputs=[audio, phonemes, status],
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Launch Kokoro Vietnamese Gradio app')
    parser.add_argument('--repo-id', default=DEFAULT_HF_REPO_ID)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--server-name', default='127.0.0.1')
    parser.add_argument('--server-port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    demo = build_demo(GradioSynthesizer(repo_id=args.repo_id, device=args.device))
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
