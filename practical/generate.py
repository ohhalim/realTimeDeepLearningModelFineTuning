"""
Simple Generation Script

Generate Brad Mehldau-style jazz piano samples from trained model.
"""

import argparse
import torch
from pathlib import Path
import json

from simple_model import SimpleGPT2, ModelConfig
from simple_tokenizer import SimpleMIDITokenizer


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model config
    model_config_dict = checkpoint.get('model_config', {})
    config = ModelConfig(
        vocab_size=model_config_dict.get('vocab_size', 220),
        n_layer=model_config_dict.get('n_layer', 6),
        n_head=model_config_dict.get('n_head', 8),
        n_embd=model_config_dict.get('n_embd', 512),
    )

    # Create model
    model = SimpleGPT2(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Val loss: {checkpoint.get('val_loss', 0):.4f}")
    print(f"  Perplexity: {checkpoint.get('perplexity', 0):.2f}")

    return model, config


def generate_sample(model, tokenizer, primer_length=32, generate_length=256,
                   temperature=0.8, top_k=40, device='cpu'):
    """
    Generate a single MIDI sample.

    Args:
        model: Trained model
        tokenizer: MIDI tokenizer
        primer_length: Length of random primer
        generate_length: How many tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling

    Returns:
        tokens: Generated token sequence
    """
    # Create random primer (C major scale pattern)
    primer_tokens = []
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C major

    for i in range(primer_length // 2):
        pitch = scale[i % len(scale)]
        primer_tokens.append(tokenizer.pitch_to_token(pitch))
        primer_tokens.append(tokenizer.time_to_token(250))  # Quarter note

    # Convert to tensor
    idx = torch.tensor([primer_tokens], dtype=torch.long, device=device)

    # Generate
    print(f"Generating {generate_length} tokens...")
    with torch.no_grad():
        generated = model.generate(
            idx,
            max_new_tokens=generate_length,
            temperature=temperature,
            top_k=top_k
        )

    tokens = generated[0].tolist()

    return tokens


def tokens_to_midi_file(tokens, tokenizer, output_path, tempo_bpm=120):
    """
    Convert tokens to MIDI file.

    Note: This is a simplified version. For proper MIDI export,
    you would use pretty_midi or mido library.
    """
    events = tokenizer.decode(tokens)
    notes = tokenizer.events_to_midi_like(events)

    # Save as simple text format (pseudo-MIDI)
    with open(output_path, 'w') as f:
        f.write(f"# Brad Mehldau Style Jazz Piano\n")
        f.write(f"# Tempo: {tempo_bpm} BPM\n")
        f.write(f"# Total notes: {len(notes)}\n")
        f.write(f"#\n")
        f.write(f"# Format: time_ms, pitch, velocity\n")
        f.write(f"#\n")

        for time_ms, pitch, velocity in notes:
            f.write(f"{time_ms:6d}, {pitch:3d}, {velocity:3d}\n")

    print(f"  ✓ Saved to {output_path}")
    print(f"    {len(notes)} notes, {max(t for t, _, _ in notes)/1000:.1f} seconds")


def evaluate_sample_quality(tokens, tokenizer):
    """
    Simple quality evaluation of generated sample.

    Returns:
        dict with quality metrics
    """
    events = tokenizer.decode(tokens)
    notes = tokenizer.events_to_midi_like(events)

    # Extract pitches and times
    pitches = [pitch for _, pitch, _ in notes]
    times = [time for time, _, _ in notes]
    velocities = [vel for _, _, vel in notes]

    # Compute basic stats
    metrics = {}

    # 1. Note count
    metrics['num_notes'] = len(notes)

    # 2. Pitch range
    if pitches:
        metrics['pitch_min'] = min(pitches)
        metrics['pitch_max'] = max(pitches)
        metrics['pitch_range'] = max(pitches) - min(pitches)

    # 3. Duration
    if times:
        metrics['duration_ms'] = max(times)
        metrics['duration_sec'] = max(times) / 1000.0

    # 4. Average velocity
    if velocities:
        metrics['avg_velocity'] = sum(velocities) / len(velocities)

    # 5. Pitch diversity (unique pitches used)
    if pitches:
        metrics['unique_pitches'] = len(set(pitches))
        metrics['pitch_diversity'] = len(set(pitches)) / 88  # Piano has 88 keys

    # 6. Rhythm regularity (std of inter-note times)
    if len(times) > 1:
        inter_note_times = [times[i+1] - times[i] for i in range(len(times)-1)]
        avg_int = sum(inter_note_times) / len(inter_note_times)
        variance = sum((t - avg_int)**2 for t in inter_note_times) / len(inter_note_times)
        metrics['rhythm_std'] = variance ** 0.5

    return metrics


def print_quality_report(metrics):
    """Print quality evaluation report."""
    print("\n" + "="*60)
    print("Quality Evaluation")
    print("="*60)

    print(f"\nBasic Stats:")
    print(f"  Notes generated:    {metrics.get('num_notes', 0):>6}")
    print(f"  Duration:           {metrics.get('duration_sec', 0):>6.1f} seconds")

    print(f"\nPitch Stats:")
    print(f"  Range:              {metrics.get('pitch_min', 0):>3} - {metrics.get('pitch_max', 0):>3} ({metrics.get('pitch_range', 0):>2} semitones)")
    print(f"  Unique pitches:     {metrics.get('unique_pitches', 0):>3} / 88")
    print(f"  Diversity:          {metrics.get('pitch_diversity', 0)*100:>5.1f}%")

    print(f"\nDynamics:")
    print(f"  Average velocity:   {metrics.get('avg_velocity', 0):>6.1f} / 127")

    print(f"\nRhythm:")
    print(f"  Timing regularity:  {metrics.get('rhythm_std', 0):>6.1f} ms std dev")

    # Quality assessment
    print(f"\nQuality Assessment:")

    # Check if it looks reasonable
    checks = []

    if metrics.get('num_notes', 0) > 10:
        checks.append("✓ Sufficient notes generated")
    else:
        checks.append("⚠ Too few notes")

    if 20 <= metrics.get('pitch_range', 0) <= 60:
        checks.append("✓ Reasonable pitch range")
    elif metrics.get('pitch_range', 0) < 20:
        checks.append("⚠ Pitch range too narrow")
    else:
        checks.append("⚠ Pitch range too wide")

    if metrics.get('pitch_diversity', 0) > 0.1:
        checks.append("✓ Good pitch diversity")
    else:
        checks.append("⚠ Low pitch diversity")

    if 30 <= metrics.get('avg_velocity', 0) <= 100:
        checks.append("✓ Reasonable dynamics")
    else:
        checks.append("⚠ Dynamics out of range")

    for check in checks:
        print(f"  {check}")

    print("="*60)


def main(args):
    print("="*70)
    print("Sample Generation - Brad Mehldau Style Jazz Piano")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\n✓ Using device: {device}")

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Create tokenizer
    tokenizer = SimpleMIDITokenizer()

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    print(f"\n{'='*70}")
    print(f"Generating {args.num_samples} samples...")
    print(f"{'='*70}")

    all_metrics = []

    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}:")

        # Generate
        tokens = generate_sample(
            model, tokenizer,
            primer_length=args.primer_length,
            generate_length=args.generate_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )

        # Save
        output_path = output_dir / f"sample_{i+1:03d}.txt"
        tokens_to_midi_file(tokens, tokenizer, output_path, tempo_bpm=args.tempo)

        # Evaluate
        metrics = evaluate_sample_quality(tokens, tokenizer)
        all_metrics.append(metrics)

        # Print brief stats
        print(f"  {metrics['num_notes']} notes, "
              f"{metrics['duration_sec']:.1f}s, "
              f"pitch range: {metrics['pitch_range']} semitones")

    # Overall quality report
    print(f"\n{'='*70}")
    print("Overall Quality Report")
    print(f"{'='*70}")

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m.get(key, 0) for m in all_metrics]
        avg_metrics[key] = sum(values) / len(values)

    print_quality_report(avg_metrics)

    print(f"\n✓ All samples saved to: {output_dir}")
    print(f"\nTo play samples:")
    print(f"  (Currently saved as text. For MIDI, install 'pretty_midi':)")
    print(f"  pip install pretty_midi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Brad Mehldau-style jazz piano samples"
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to generate (default: 10)')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                       help='Output directory (default: ./generated_samples)')

    parser.add_argument('--primer_length', type=int, default=32,
                       help='Primer length in tokens (default: 32)')
    parser.add_argument('--generate_length', type=int, default=256,
                       help='Generation length in tokens (default: 256)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (default: 0.8)')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k sampling (default: 40)')
    parser.add_argument('--tempo', type=int, default=120,
                       help='Tempo in BPM (default: 120)')

    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')

    args = parser.parse_args()
    main(args)
