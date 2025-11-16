#!/usr/bin/env python3
"""
QLoRA를 사용한 음악 생성 모델 파인튜닝

Brad Mehldau 스타일의 재즈 피아노를 생성하기 위해
GPT-2 기반 모델을 QLoRA로 파인튜닝합니다.

사용법:
    python scripts/train_qlora.py --config configs/qlora_config.yaml
"""

import os
import argparse
import yaml
import torch
from datetime import datetime
from pathlib import Path

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from transformers import BitsAndBytesConfig
from datasets import load_dataset, Dataset

import pretty_midi
import numpy as np
from tqdm import tqdm


class MIDITokenizer:
    """
    MIDI를 토큰으로 변환하는 간단한 토크나이저
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

    def encode_midi(self, midi_path, max_length=1024):
        """
        MIDI 파일을 토큰 시퀀스로 변환
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            tokens = [self.bos_token]

            # 피아노 트랙만 추출
            piano_notes = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    piano_notes.extend(instrument.notes)

            # 시간 순으로 정렬
            piano_notes.sort(key=lambda x: x.start)

            current_time = 0
            for note in piano_notes:
                # 시간 간격 토큰
                time_diff = int((note.start - current_time) * 100)  # 10ms 단위
                time_diff = min(time_diff, 127)  # 최대값 제한
                if time_diff > 0:
                    tokens.append(self.time_shift_offset + time_diff)

                # 노트 온 토큰
                tokens.append(self.note_on_offset + note.pitch)

                # 벨로시티 토큰
                velocity = note.velocity // 2  # 0-127 -> 0-63
                tokens.append(self.velocity_offset + velocity)

                # 노트 오프 토큰 (duration)
                duration = int((note.end - note.start) * 100)
                duration = min(duration, 127)
                tokens.append(self.note_off_offset + note.pitch)

                current_time = note.start

                # 최대 길이 제한
                if len(tokens) >= max_length - 1:
                    break

            tokens.append(self.eos_token)

            # 패딩
            if len(tokens) < max_length:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]

            return tokens

        except Exception as e:
            print(f"Error encoding {midi_path}: {e}")
            # 빈 시퀀스 반환
            return [self.bos_token] + [self.pad_token] * (max_length - 1)


def prepare_dataset(midi_dir, tokenizer, max_length=1024, train_split=0.8):
    """
    MIDI 파일들을 데이터셋으로 변환
    """
    print("=" * 60)
    print("데이터셋 준비 중...")
    print("=" * 60)

    midi_files = []
    for ext in ['*.mid', '*.midi']:
        midi_files.extend(Path(midi_dir).rglob(ext))

    print(f"찾은 MIDI 파일: {len(midi_files)}개")

    if len(midi_files) == 0:
        raise ValueError(f"MIDI 파일을 찾을 수 없습니다: {midi_dir}")

    # MIDI 파일을 토큰으로 변환
    all_sequences = []
    for midi_file in tqdm(midi_files, desc="MIDI 인코딩"):
        tokens = tokenizer.encode_midi(str(midi_file), max_length)
        all_sequences.append({"input_ids": tokens})

    # 데이터 증강: 각 시퀀스를 여러 번 사용
    augmented_sequences = []
    for seq in all_sequences:
        # 원본
        augmented_sequences.append(seq)
        # 약간의 변형 (추후 transpose 등 추가 가능)
        augmented_sequences.append(seq)

    print(f"증강된 시퀀스: {len(augmented_sequences)}개")

    # Train/Val 분할
    split_idx = int(len(augmented_sequences) * train_split)
    train_data = augmented_sequences[:split_idx]
    val_data = augmented_sequences[split_idx:]

    print(f"학습 데이터: {len(train_data)}개")
    print(f"검증 데이터: {len(val_data)}개")

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def setup_qlora_model(model_name, qlora_config):
    """
    QLoRA 설정으로 모델 로드
    """
    print("=" * 60)
    print("QLoRA 모델 설정 중...")
    print("=" * 60)

    # 4-bit 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 모델 로드
    print(f"모델 로드: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # QLoRA를 위한 모델 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    lora_config = LoraConfig(
        r=qlora_config.get("lora_r", 16),
        lora_alpha=qlora_config.get("lora_alpha", 32),
        target_modules=qlora_config.get("target_modules", ["c_attn", "c_proj"]),
        lora_dropout=qlora_config.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # LoRA 적용
    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 출력
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params

    print(f"전체 파라미터: {all_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")
    print(f"학습 비율: {trainable_percent:.2f}%")
    print("=" * 60)

    return model


def train(config_path):
    """
    QLoRA 파인튜닝 실행
    """
    # 설정 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("QLoRA 파인튜닝 시작")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"설정 파일: {config_path}")
    print()

    # 설정 추출
    model_name = config.get("model_name", "gpt2")
    midi_dir = config.get("midi_dir", "data/raw_midi")
    output_dir = config.get("output_dir", "models/finetuned/qlora")
    max_length = config.get("max_length", 1024)

    # MIDI 토크나이저
    tokenizer = MIDITokenizer(vocab_size=512)

    # 데이터셋 준비
    train_dataset, val_dataset = prepare_dataset(
        midi_dir, tokenizer, max_length
    )

    # QLoRA 모델 설정
    model = setup_qlora_model(model_name, config.get("qlora", {}))

    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("epochs", 50),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        fp16=True,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        eval_steps=config.get("eval_steps", 100),
        evaluation_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        warmup_steps=config.get("warmup_steps", 50),
        optim="paged_adamw_8bit",  # QLoRA 최적화
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 학습 시작
    print("\n학습 시작...")
    trainer.train()

    # 모델 저장
    print("\n모델 저장 중...")
    model.save_pretrained(output_dir)

    print("=" * 60)
    print("학습 완료!")
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델 저장 위치: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA를 사용한 Brad Mehldau 스타일 음악 생성 모델 파인튜닝"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qlora_config.yaml",
        help="설정 파일 경로"
    )

    args = parser.parse_args()

    # GPU 확인
    if torch.cuda.is_available():
        print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  GPU를 사용할 수 없습니다. CPU로 학습합니다 (매우 느림)")

    print()

    # 학습 실행
    train(args.config)


if __name__ == "__main__":
    main()
