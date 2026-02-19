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


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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
            velocity = torch.randint(20, 50, (1,)).item()  # token bin (0-63)
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


def create_conditioned_primer(
    tokenizer: MIDITokenizer,
    conditioning_midi: str,
    role: str = "lead",
    tempo_bpm: float = 120.0,
    max_conditioning_tokens: int = 256,
):
    """
    Build role-conditioned primer:
    [BOS, ROLE, CHORD_UNKNOWN, TEMPO_*, COND_SEP, conditioning events..., COND_SEP]
    """
    if not os.path.exists(conditioning_midi):
        raise FileNotFoundError(f"conditioning_midi not found: {conditioning_midi}")

    role_token = tokenizer.role_to_token(role)
    chord_token = tokenizer.chord_to_token(None)
    tempo_token = tokenizer.tempo_to_token(tempo_bpm)
    sep_token = tokenizer.control_token("COND_SEP")

    conditioning_tokens = tokenizer.encode_compact(
        conditioning_midi,
        max_tokens=max_conditioning_tokens,
        include_bos=False,
        include_eos=False,
    )

    primer = [
        tokenizer.bos_token,
        role_token,
        chord_token,
        tempo_token,
        sep_token,
    ]
    primer.extend(conditioning_tokens)
    primer.append(sep_token)
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
        "--inference_config",
        type=str,
        default=None,
        help="추론 전용 YAML 설정 파일 (예: configs/inference/realtime_stage_a.yaml)"
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
        default=None,
        help="최대 생성 길이"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="샘플링 temperature (0.5=보수적, 1.5=창의적)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k 샘플링"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus 샘플링"
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=None,
        help="생성 시 참조할 최근 컨텍스트 토큰 수"
    )
    parser.add_argument(
        "--primer_type",
        type=str,
        default="random",
        choices=['random', 'scale', 'chord'],
        help="Primer 타입"
    )
    parser.add_argument(
        "--conditioning_midi",
        type=str,
        default=None,
        help="Role-conditioned primer로 사용할 MIDI 파일"
    )
    parser.add_argument(
        "--role",
        type=str,
        default="lead",
        choices=["lead", "accompaniment", "call_response"],
        help="role-conditioned 생성에서 사용할 role"
    )
    parser.add_argument(
        "--conditioning_tempo",
        type=float,
        default=None,
        help="role-conditioned primer tempo metadata"
    )
    parser.add_argument(
        "--max_conditioning_tokens",
        type=int,
        default=None,
        help="conditioning MIDI에서 사용할 최대 토큰 수"
    )
    parser.add_argument(
        "--include_primer_in_output",
        action="store_true",
        help="출력 MIDI에 primer 구간도 포함"
    )
    parser.add_argument(
        "--tempo",
        type=int,
        default=None,
        help="MIDI 템포 (BPM)"
    )

    args = parser.parse_args()

    infer_cfg = {}
    if args.inference_config:
        infer_cfg = load_yaml(args.inference_config)
        print(f"Inference config loaded: {args.inference_config}")

    generation_cfg = infer_cfg.get("generation", {})

    max_length = args.max_length if args.max_length is not None else int(generation_cfg.get("max_length", 1024))
    temperature = args.temperature if args.temperature is not None else float(generation_cfg.get("temperature", 1.0))
    top_k = args.top_k if args.top_k is not None else int(generation_cfg.get("top_k", 40))
    top_p = args.top_p if args.top_p is not None else float(generation_cfg.get("top_p", 0.9))
    context_window = args.context_window if args.context_window is not None else int(generation_cfg.get("context_window", 256))
    max_conditioning_tokens = (
        args.max_conditioning_tokens
        if args.max_conditioning_tokens is not None
        else int(generation_cfg.get("max_conditioning_tokens", 256))
    )
    conditioning_tempo = (
        args.conditioning_tempo
        if args.conditioning_tempo is not None
        else float(generation_cfg.get("conditioning_tempo", 120.0))
    )
    midi_tempo = args.tempo if args.tempo is not None else int(generation_cfg.get("midi_tempo", 120))

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
        if args.conditioning_midi:
            primer = create_conditioned_primer(
                tokenizer=tokenizer,
                conditioning_midi=args.conditioning_midi,
                role=args.role,
                tempo_bpm=conditioning_tempo,
                max_conditioning_tokens=max_conditioning_tokens,
            )
            print(f"Primer mode: role-conditioned ({args.role})")
            print(f"Conditioning MIDI: {args.conditioning_midi}")
        else:
            primer = create_primer(tokenizer, args.primer_type)
            print(f"Primer mode: {args.primer_type}")

        if len(primer) >= max_length:
            print(
                f"Warning: primer 길이({len(primer)})가 max_length({max_length}) 이상입니다. "
                "primer를 잘라서 사용합니다."
            )
            primer = primer[: max(8, max_length // 2)]

        print(f"Primer length: {len(primer)} tokens")

        # Generate
        print("생성 중...")
        generated = model.generate(
            primer=primer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            stop_at_eos=True,
            context_window=context_window,
            eos_token_id=tokenizer.eos_token,
        )

        print(f"✓ 생성 완료: {len(generated)} 토큰")

        # Decode to MIDI
        if args.num_samples > 1:
            output_path = args.output.replace('.mid', f'_{i + 1}.mid')
        else:
            output_path = args.output

        generated_list = generated.tolist()
        if args.include_primer_in_output:
            decode_tokens = generated_list
        else:
            continuation_tokens = generated_list[len(primer):]
            decode_tokens = [tokenizer.bos_token] + continuation_tokens
            if not continuation_tokens:
                decode_tokens = generated_list

        tokenizer.decode(decode_tokens, output_path, midi_tempo)

        print(f"✓ MIDI 저장: {output_path}")
        print()

    print("=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
