#!/usr/bin/env python3
"""
PyTorch Music Transformer 음악 생성 스크립트

학습된 모델로 MIDI 파일 생성

사용법:
    python scripts/generate_pytorch_transformer.py \
        --checkpoint models/finetuned/pytorch_transformer/best_model.pt \
        --output output/generated.mid \
        --num_samples 5
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from models.music_transformer import MusicTransformer
from models.midi_dataset import MIDITokenizer


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained model from checkpoint
    """
    print("=" * 60)
    print("모델 로딩...")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MusicTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['data']['max_length'],
        dropout=0.0  # No dropout for inference
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"✓ Loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    print()

    return model, config


def create_primer(tokenizer, primer_type='random', length=32):
    """
    Create primer sequence

    Args:
        tokenizer: MIDI tokenizer
        primer_type: 'random', 'scale', or 'chord'
        length: Primer length

    Returns:
        primer: List of tokens
    """
    if primer_type == 'random':
        # Random notes from C major scale
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major
        primer = [tokenizer.bos_token]

        for _ in range(length):
            # Note on
            pitch = scale[torch.randint(0, len(scale), (1,)).item()]
            primer.append(tokenizer.note_on_offset + pitch)

            # Velocity
            velocity = torch.randint(40, 80, (1,)).item()
            primer.append(tokenizer.velocity_offset + velocity)

            # Time shift (8th note)
            primer.append(tokenizer.time_shift_offset + 12)

            # Note off
            primer.append(tokenizer.note_off_offset + pitch)

    elif primer_type == 'scale':
        # C major scale ascending
        scale = [60, 62, 64, 65, 67, 69, 71, 72]
        primer = [tokenizer.bos_token]

        for pitch in scale:
            primer.extend([
                tokenizer.note_on_offset + pitch,
                tokenizer.velocity_offset + 40,
                tokenizer.time_shift_offset + 12,
                tokenizer.note_off_offset + pitch
            ])

    elif primer_type == 'chord':
        # C major chord (C-E-G)
        chord = [60, 64, 67]
        primer = [tokenizer.bos_token]

        # Play chord
        for pitch in chord:
            primer.extend([
                tokenizer.note_on_offset + pitch,
                tokenizer.velocity_offset + 50
            ])

        # Hold
        primer.append(tokenizer.time_shift_offset + 50)

        # Release
        for pitch in chord:
            primer.append(tokenizer.note_off_offset + pitch)

    else:
        raise ValueError(f"Unknown primer type: {primer_type}")

    return primer


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Music Transformer 음악 생성"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="모델 체크포인트 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/pytorch_generated.mid",
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
        help="최대 생성 길이"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="샘플링 temperature (0.5=보수적, 1.5=창의적)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k 샘플링"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus 샘플링"
    )
    parser.add_argument(
        "--primer_type",
        type=str,
        default="random",
        choices=['random', 'scale', 'chord'],
        help="Primer 타입"
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=120,
        help="MIDI 템포 (BPM)"
    )

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Tokenizer
    tokenizer = MIDITokenizer(vocab_size=config['model']['vocab_size'])

    # Generate samples
    for i in range(args.num_samples):
        print("=" * 60)
        print(f"샘플 {i + 1}/{args.num_samples} 생성")
        print("=" * 60)

        # Create primer
        primer = create_primer(tokenizer, args.primer_type)
        print(f"Primer length: {len(primer)} tokens")
        print(f"Primer type: {args.primer_type}")

        # Generate
        print("생성 중...")
        generated = model.generate(
            primer=primer,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )

        print(f"✓ 생성 완료: {len(generated)} 토큰")

        # Decode to MIDI
        if args.num_samples > 1:
            output_path = args.output.replace('.mid', f'_{i + 1}.mid')
        else:
            output_path = args.output

        tokenizer.decode(generated.tolist(), output_path, args.tempo)

        print(f"✓ MIDI 저장: {output_path}")
        print()

    print("=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
