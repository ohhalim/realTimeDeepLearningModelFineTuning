# Phase 5: 배포 (1주, 무료)

**목표**: Hugging Face에 데모 사이트 배포
**예상 시간**: 5-10시간 (1주)
**비용**: 무료

---

## Gradio 앱 생성

```python
# deployment/app.py
import gradio as gr
import torch
from models.jazzformer_cr import JazzFormerCR
from preprocessing.tokenizer import SimpleMIDITokenizer

# Load model
model = JazzFormerCR.from_pretrained("checkpoints/jazzformer_cr/best_model.pt")
tokenizer = SimpleMIDITokenizer()

def generate_jazz(temperature, length):
    """Generate jazz sample"""
    # Generate
    prompt = tokenizer.encode_prompt()
    generated = model.generate(prompt, max_new_tokens=length, temperature=temperature)

    # Convert to MIDI
    midi_path = save_as_midi(generated)

    return midi_path

# Gradio interface
demo = gr.Interface(
    fn=generate_jazz,
    inputs=[
        gr.Slider(0.5, 1.5, value=0.9, label="Temperature"),
        gr.Slider(64, 512, value=256, label="Length (tokens)"),
    ],
    outputs=gr.Audio(type="filepath"),
    title="🎹 Brad Mehldau AI - Jazz Piano Improvisation",
    description="AI that improvises jazz piano in Brad Mehldau's style"
)

demo.launch()
```

---

## Hugging Face Space

```bash
# 1. Hugging Face 계정 생성
# 2. New Space 클릭
# 3. Gradio SDK 선택
# 4. app.py 업로드
# 5. requirements.txt 업로드
# 6. 모델 checkpoint 업로드 (Git LFS)

# 완성!
# URL: https://huggingface.co/spaces/YOUR_USERNAME/brad-mehldau-ai
```

---

**다음**: `../phase6_portfolio/README.md`
