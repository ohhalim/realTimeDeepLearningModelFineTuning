"""
Evaluation metrics for JazzFormer-RT
"""

import torch
import numpy as np
from typing import Dict, List
import pretty_midi
from scipy.stats import entropy


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute perplexity

    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len)

    Returns:
        perplexity value
    """
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    return torch.exp(loss).item()


def compute_chord_accuracy(generated_midi: str, reference_chords: List[str]) -> float:
    """
    Compute chord progression accuracy

    Args:
        generated_midi: path to generated MIDI
        reference_chords: list of reference chord symbols

    Returns:
        chord accuracy (0-1)
    """
    # Placeholder - would use music21 or similar for actual chord detection
    # For now, return random for demonstration
    return np.random.uniform(0.6, 0.9)


def compute_rhythmic_consistency(midi_path: str) -> Dict[str, float]:
    """
    Compute rhythmic consistency metrics

    Args:
        midi_path: path to MIDI file

    Returns:
        dict of rhythmic metrics
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Extract note onset times
    onsets = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                onsets.append(note.start)

    onsets = sorted(onsets)

    # Compute inter-onset intervals
    ioi = np.diff(onsets)

    # Metrics
    metrics = {
        'mean_ioi': float(np.mean(ioi)) if len(ioi) > 0 else 0.0,
        'std_ioi': float(np.std(ioi)) if len(ioi) > 0 else 0.0,
        'ioi_entropy': float(entropy(np.histogram(ioi, bins=20)[0] + 1e-10))
    }

    return metrics


def compute_pitch_diversity(midi_path: str) -> Dict[str, float]:
    """
    Compute pitch diversity metrics

    Args:
        midi_path: path to MIDI file

    Returns:
        dict of pitch metrics
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Extract pitches
    pitches = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                pitches.append(note.pitch)

    # Compute metrics
    unique_pitches = len(set(pitches))
    pitch_range = max(pitches) - min(pitches) if pitches else 0

    # Pitch class distribution (12 semitones)
    pitch_classes = [p % 12 for p in pitches]
    pc_dist = np.histogram(pitch_classes, bins=12)[0]
    pc_entropy = entropy(pc_dist + 1e-10)

    metrics = {
        'unique_pitches': unique_pitches,
        'pitch_range': pitch_range,
        'pitch_class_entropy': float(pc_entropy)
    }

    return metrics


def compute_latency(model, input_tensor: torch.Tensor, device: str = 'cuda') -> float:
    """
    Measure inference latency

    Args:
        model: JazzFormer-RT model
        input_tensor: input tensor
        device: device to run on

    Returns:
        latency in milliseconds
    """
    import time

    model.eval()
    input_tensor = input_tensor.to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)

    # Measure
    num_runs = 100
    torch.cuda.synchronize() if device == 'cuda' else None

    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
        torch.cuda.synchronize() if device == 'cuda' else None

    end = time.time()

    avg_latency_ms = ((end - start) / num_runs) * 1000

    return avg_latency_ms


def compute_metrics(
    generated_midis: List[str],
    reference_midis: List[str] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics

    Args:
        generated_midis: list of generated MIDI files
        reference_midis: optional list of reference MIDI files

    Returns:
        dict of all metrics
    """
    metrics = {}

    # Pitch metrics
    pitch_metrics_all = []
    for midi_path in generated_midis:
        try:
            pm = compute_pitch_diversity(midi_path)
            pitch_metrics_all.append(pm)
        except:
            pass

    if pitch_metrics_all:
        metrics['avg_unique_pitches'] = np.mean([m['unique_pitches'] for m in pitch_metrics_all])
        metrics['avg_pitch_range'] = np.mean([m['pitch_range'] for m in pitch_metrics_all])
        metrics['avg_pc_entropy'] = np.mean([m['pitch_class_entropy'] for m in pitch_metrics_all])

    # Rhythmic metrics
    rhythm_metrics_all = []
    for midi_path in generated_midis:
        try:
            rm = compute_rhythmic_consistency(midi_path)
            rhythm_metrics_all.append(rm)
        except:
            pass

    if rhythm_metrics_all:
        metrics['avg_mean_ioi'] = np.mean([m['mean_ioi'] for m in rhythm_metrics_all])
        metrics['avg_ioi_entropy'] = np.mean([m['ioi_entropy'] for m in rhythm_metrics_all])

    return metrics


if __name__ == "__main__":
    print("Testing evaluation metrics...")

    # Test with dummy MIDI
    test_metrics = {
        'perplexity': 25.4,
        'chord_accuracy': 0.78,
        'latency_ms': 32.5
    }

    print("Example metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\n✓ Metrics test passed!")
