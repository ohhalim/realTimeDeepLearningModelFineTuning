#!/usr/bin/env python3
"""
QLoRA로 파인튜닝된 모델을 사용한 음악 생성

사용법:
    python scripts/generate_qlora.py --checkpoint models/finetuned/qlora_brad_mehldau --output output/qlora_generated.mid
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pretty_midi


class MIDITokenizer:
    """
    MIDI 토큰 디코더
    """

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.note_on_offset = 0
        self.note_off_offset = 128
        self.time_shift_offset = 256
        self.velocity_offset = 384

        # 특수 토큰
        self.pad_token = vocab_size - 4
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.unk_token = vocab_size - 1

    def decode_to_midi(self, tokens, output_path, tempo=120):
        """
        토큰 시퀀스를 MIDI 파일로 변환
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)

        for token in tokens:
            if token == self.eos_token or token == self.pad_token:
                continue

            # Note On
            if self.note_on_offset <= token < self.note_off_offset:
                pitch = token - self.note_on_offset
                if 0 <= pitch <= 127:
                    # 이미 활성화된 노트가 있으면 종료
                    if pitch in active_notes:
                        start_time, velocity = active_notes[pitch]
                        note = pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=current_time
                        )
                        piano.notes.append(note)
                    active_notes[pitch] = (current_time, 80)  # 기본 velocity

            # Note Off
            elif self.note_off_offset <= token < self.time_shift_offset:
                pitch = token - self.note_off_offset
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
                    del active_notes[pitch]

            # Time Shift
            elif self.time_shift_offset <= token < self.velocity_offset:
                time_shift = (token - self.time_shift_offset) / 100.0  # 10ms 단위
                current_time += time_shift

            # Velocity
            elif self.velocity_offset <= token < self.pad_token:
                velocity = (token - self.velocity_offset) * 2  # 0-63 -> 0-126
                # 마지막 노트의 velocity 업데이트
                if active_notes:
                    last_pitch = list(active_notes.keys())[-1]
                    start_time, _ = active_notes[last_pitch]
                    active_notes[last_pitch] = (start_time, velocity)

        # 남은 활성 노트 종료
        for pitch, (start_time, velocity) in active_notes.items():
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=current_time + 0.5  # 0.5초 추가
            )
            piano.notes.append(note)

        midi.instruments.append(piano)

        # MIDI 파일 저장
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        midi.write(output_path)
        print(f"✓ MIDI 파일 생성: {output_path}")
        print(f"  - 총 노트 수: {len(piano.notes)}")
        print(f"  - 길이: {midi.get_end_time():.2f}초")


def load_qlora_model(checkpoint_path, base_model="gpt2"):
    """
    QLoRA 파인튜닝된 모델 로드
    """
    print("=" * 60)
    print("모델 로딩 중...")
    print("=" * 60)

    # Base model 로드
    print(f"Base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # LoRA weights 적용
    print(f"LoRA checkpoint: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)

    # Evaluation mode
    model.eval()

    print("✓ 모델 로딩 완료")
    print("=" * 60)

    return model


def generate_music(
    model,
    tokenizer,
    primer_tokens=None,
    max_length=1024,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
):
    """
    음악 생성
    """
    print("\n음악 생성 중...")

    # Primer 설정
    if primer_tokens is None:
        # BOS 토큰으로 시작
        input_ids = torch.tensor([[tokenizer.bos_token]]).to(model.device)
    else:
        input_ids = torch.tensor([primer_tokens]).to(model.device)

    # 생성
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token,
            eos_token_id=tokenizer.eos_token,
        )

    generated_tokens = output[0].cpu().numpy().tolist()
    print(f"✓ 생성 완료: {len(generated_tokens)} 토큰")

    return generated_tokens


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA 파인튜닝 모델로 음악 생성"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="파인튜닝된 모델 체크포인트 경로"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt2",
        help="Base model 이름 (기본값: gpt2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/qlora_generated.mid",
        help="출력 MIDI 파일 경로"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="생성할 샘플 수"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="최대 시퀀스 길이"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature (0.5=안전, 1.0=균형, 1.5=창의적)"
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=120,
        help="MIDI 템포 (BPM)"
    )

    args = parser.parse_args()

    # GPU 확인
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CPU 모드")

    # 토크나이저
    tokenizer = MIDITokenizer(vocab_size=512)

    # 모델 로드
    model = load_qlora_model(args.checkpoint, args.base_model)

    # 여러 샘플 생성
    for i in range(args.num_samples):
        print(f"\n{'=' * 60}")
        print(f"샘플 {i + 1}/{args.num_samples} 생성")
        print(f"{'=' * 60}")

        # 음악 생성
        generated_tokens = generate_music(
            model,
            tokenizer,
            max_length=args.max_length,
            temperature=args.temperature,
        )

        # MIDI로 변환
        if args.num_samples > 1:
            output_path = args.output.replace('.mid', f'_{i + 1}.mid')
        else:
            output_path = args.output

        tokenizer.decode_to_midi(
            generated_tokens,
            output_path,
            tempo=args.tempo
        )

    print(f"\n{'=' * 60}")
    print("생성 완료!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
