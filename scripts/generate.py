#!/usr/bin/env python3
"""
Generate music using trained JazzFormer-RT model

Usage:
    python scripts/generate.py --checkpoint path/to/model.pt --output_dir output/
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.jazzformer_rt import JazzFormerRT
from src.data.dataset import MIDITokenizer


def load_model(checkpoint_path: str, device: str = 'cuda') -> JazzFormerRT:
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})

    # Create model
    if config and 'architecture' in config:
        arch_config = config['architecture']
        model = JazzFormerRT(
            vocab_size=config['io']['vocab_size'],
            d_model=arch_config['d_model'],
            n_heads=arch_config['n_heads'],
            n_layers=arch_config['n_layers'],
            d_ff=arch_config['d_ff'],
            max_seq_len=config['io']['max_sequence_length'],
            dropout=arch_config['dropout'],
            num_artists=arch_config['style_embedding']['num_artists'],
            style_embedding_dim=arch_config['style_embedding']['embedding_dim'],
            window_size=arch_config['streaming']['window_size']
        )
    else:
        # Default configuration
        model = JazzFormerRT()

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully!")
    return model


def generate_music(
    model: JazzFormerRT,
    num_outputs: int = 5,
    output_dir: str = 'output',
    temperature: float = 1.0,
    max_length: int = 512,
    artist_id: int = 0,
    device: str = 'cuda'
):
    """Generate music samples"""

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = MIDITokenizer()

    print(f"\nGenerating {num_outputs} samples...")
    print(f"Temperature: {temperature}")
    print(f"Max length: {max_length}")
    print(f"Artist ID: {artist_id}")
    print(f"Output directory: {output_dir}\n")

    for i in range(num_outputs):
        # Create random prompt (start token)
        prompt = torch.randint(0, model.vocab_size, (1, 32), device=device)

        # Generate
        with torch.no_grad():
            generated = model.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                artist_id=artist_id
            )

        # Convert to MIDI
        output_path = os.path.join(output_dir, f'generated_{i+1}.mid')
        tokenizer.decode_tokens(generated[0].cpu(), output_path)

        print(f"✓ Generated sample {i+1}/{num_outputs}: {output_path}")

    print(f"\n✓ All samples generated successfully!")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with JazzFormer-RT"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/generated',
        help='Output directory for generated MIDI files'
    )
    parser.add_argument(
        '--num_outputs',
        type=int,
        default=5,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (0.5-1.5)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--artist_id',
        type=int,
        default=0,
        help='Artist ID for style (0=Brad Mehldau)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Generate music
    generate_music(
        model=model,
        num_outputs=args.num_outputs,
        output_dir=args.output_dir,
        temperature=args.temperature,
        max_length=args.max_length,
        artist_id=args.artist_id,
        device=args.device
    )


if __name__ == '__main__':
    main()
