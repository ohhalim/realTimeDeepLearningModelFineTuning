"""
Objective Evaluation Metrics for Music Generation

Implements comprehensive metrics for evaluating generated MIDI:
1. Perplexity (language modeling metric)
2. Harmonic Accuracy (chord recognition accuracy)
3. Rhythm Consistency (tempo stability, syncopation)
4. Voicing Similarity (pitch range, interval distribution)
5. Style Classification Accuracy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import pretty_midi


class PerplexityMetric:
    """
    Perplexity: exp(cross_entropy)

    Lower is better. Measures how well the model predicts the next token.
    """

    def __init__(self):
        self.total_loss = 0.0
        self.num_tokens = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        # Reshape
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        # Compute cross-entropy
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')

        self.total_loss += loss.item()
        self.num_tokens += targets_flat.numel()

    def compute(self) -> float:
        """Compute perplexity."""
        if self.num_tokens == 0:
            return float('inf')

        avg_loss = self.total_loss / self.num_tokens
        perplexity = np.exp(avg_loss)

        return perplexity

    def reset(self):
        self.total_loss = 0.0
        self.num_tokens = 0


class HarmonicAccuracyMetric:
    """
    Harmonic Accuracy: Chord recognition accuracy.

    Compares predicted chord labels with ground truth.
    Uses a simple pitch class histogram-based chord recognizer.
    """

    # Major and minor chord templates (pitch class sets)
    CHORD_TEMPLATES = {
        'C_major': {0, 4, 7},
        'C_minor': {0, 3, 7},
        'D_major': {2, 6, 9},
        'D_minor': {2, 5, 9},
        'E_major': {4, 8, 11},
        'E_minor': {4, 7, 11},
        'F_major': {5, 9, 0},
        'F_minor': {5, 8, 0},
        'G_major': {7, 11, 2},
        'G_minor': {7, 10, 2},
        'A_major': {9, 1, 4},
        'A_minor': {9, 0, 4},
        'B_major': {11, 3, 6},
        'B_minor': {11, 2, 6},
    }

    def __init__(self):
        self.correct = 0
        self.total = 0

    def extract_chord(self, tokens: torch.Tensor) -> str:
        """
        Extract chord from MIDI tokens.

        Args:
            tokens: [seq_len] - MIDI tokens

        Returns:
            Chord label (e.g., "C_major")
        """
        # Get note tokens (0-127)
        note_mask = tokens < 128
        notes = tokens[note_mask]

        if len(notes) == 0:
            return "unknown"

        # Pitch classes
        pitch_classes = set((notes % 12).tolist())

        # Find best matching chord template
        best_match = "unknown"
        best_score = 0

        for chord_name, template in self.CHORD_TEMPLATES.items():
            # Jaccard similarity
            intersection = len(pitch_classes & template)
            union = len(pitch_classes | template)

            if union > 0:
                score = intersection / union

                if score > best_score:
                    best_score = score
                    best_match = chord_name

        return best_match if best_score > 0.5 else "unknown"

    def update(self, predicted_tokens: torch.Tensor, target_tokens: torch.Tensor):
        """
        Args:
            predicted_tokens: [batch, seq_len]
            target_tokens: [batch, seq_len]
        """
        batch_size = predicted_tokens.shape[0]

        for i in range(batch_size):
            pred_chord = self.extract_chord(predicted_tokens[i])
            tgt_chord = self.extract_chord(target_tokens[i])

            if pred_chord != "unknown" and tgt_chord != "unknown":
                if pred_chord == tgt_chord:
                    self.correct += 1
                self.total += 1

    def compute(self) -> float:
        """Compute harmonic accuracy."""
        if self.total == 0:
            return 0.0

        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0


class RhythmConsistencyMetric:
    """
    Rhythm Consistency Metrics:
    1. Tempo Stability (variance of inter-onset intervals)
    2. Syncopation Index (frequency of off-beat notes)
    """

    def __init__(self):
        self.all_iois = []  # Inter-onset intervals
        self.all_syncopations = []

    def extract_onsets(self, tokens: torch.Tensor) -> List[int]:
        """
        Extract note onset positions from MIDI tokens.

        Returns:
            List of absolute positions (in time shift units)
        """
        position = 0
        onsets = []

        for token in tokens:
            token = token.item()

            if token < 128:  # Note ON
                onsets.append(position)
            elif 256 <= token < 384:  # Time shift
                time_shift = token - 256
                position += time_shift

        return onsets

    def compute_ioi(self, onsets: List[int]) -> List[int]:
        """Compute inter-onset intervals."""
        if len(onsets) < 2:
            return []

        iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
        return iois

    def compute_syncopation(self, onsets: List[int]) -> float:
        """
        Compute syncopation index.

        Syncopation = fraction of notes on off-beats.
        Assumes 16 time units per beat.
        """
        if len(onsets) == 0:
            return 0.0

        # Beats are at multiples of 16
        on_beat = sum(1 for onset in onsets if onset % 16 == 0)
        off_beat = len(onsets) - on_beat

        return off_beat / len(onsets)

    def update(self, tokens: torch.Tensor):
        """
        Args:
            tokens: [batch, seq_len]
        """
        batch_size = tokens.shape[0]

        for i in range(batch_size):
            onsets = self.extract_onsets(tokens[i])

            # IOI
            iois = self.compute_ioi(onsets)
            self.all_iois.extend(iois)

            # Syncopation
            syncopation = self.compute_syncopation(onsets)
            self.all_syncopations.append(syncopation)

    def compute(self) -> Dict[str, float]:
        """Compute rhythm consistency metrics."""
        # Tempo stability (lower variance = more stable)
        if len(self.all_iois) > 0:
            ioi_mean = np.mean(self.all_iois)
            ioi_std = np.std(self.all_iois)
            tempo_stability = 1.0 / (1.0 + ioi_std / (ioi_mean + 1e-8))
        else:
            tempo_stability = 0.0

        # Syncopation index
        if len(self.all_syncopations) > 0:
            syncopation_index = np.mean(self.all_syncopations)
        else:
            syncopation_index = 0.0

        return {
            'tempo_stability': tempo_stability,
            'syncopation_index': syncopation_index,
        }

    def reset(self):
        self.all_iois = []
        self.all_syncopations = []


class VoicingSimilarityMetric:
    """
    Voicing Similarity Metrics:
    1. Pitch Range Distribution (register usage)
    2. Interval Distribution (melodic intervals)
    """

    def __init__(self):
        self.pitch_ranges = []
        self.interval_histograms = []

    def extract_pitch_range(self, tokens: torch.Tensor) -> Tuple[int, int]:
        """
        Extract pitch range (lowest and highest notes).

        Returns:
            (min_pitch, max_pitch)
        """
        # Get note tokens (0-127)
        note_mask = tokens < 128
        notes = tokens[note_mask]

        if len(notes) == 0:
            return (0, 0)

        min_pitch = notes.min().item()
        max_pitch = notes.max().item()

        return (min_pitch, max_pitch)

    def extract_intervals(self, tokens: torch.Tensor) -> np.ndarray:
        """
        Extract melodic intervals.

        Returns:
            Histogram of intervals (24 bins: -12 to +12 semitones)
        """
        # Get note tokens in order
        note_mask = tokens < 128
        notes = tokens[note_mask].tolist()

        if len(notes) < 2:
            return np.zeros(25)  # -12 to +12

        # Compute intervals
        intervals = [notes[i+1] - notes[i] for i in range(len(notes)-1)]

        # Clip to [-12, 12]
        intervals = [max(-12, min(12, interval)) for interval in intervals]

        # Create histogram
        histogram = np.zeros(25)
        for interval in intervals:
            histogram[interval + 12] += 1

        # Normalize
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()

        return histogram

    def update(self, predicted_tokens: torch.Tensor, target_tokens: torch.Tensor):
        """
        Args:
            predicted_tokens: [batch, seq_len]
            target_tokens: [batch, seq_len]
        """
        batch_size = predicted_tokens.shape[0]

        for i in range(batch_size):
            # Pitch range
            pred_range = self.extract_pitch_range(predicted_tokens[i])
            self.pitch_ranges.append(pred_range)

            # Intervals
            pred_intervals = self.extract_intervals(predicted_tokens[i])
            self.interval_histograms.append(pred_intervals)

    def compute(self, target_pitch_ranges: List[Tuple[int, int]], target_interval_histograms: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute voicing similarity.

        Args:
            target_pitch_ranges: Reference pitch ranges
            target_interval_histograms: Reference interval histograms

        Returns:
            Dictionary of metrics
        """
        # Pitch range similarity (IoU)
        if len(self.pitch_ranges) > 0 and len(target_pitch_ranges) > 0:
            pred_min = np.mean([r[0] for r in self.pitch_ranges])
            pred_max = np.mean([r[1] for r in self.pitch_ranges])
            tgt_min = np.mean([r[0] for r in target_pitch_ranges])
            tgt_max = np.mean([r[1] for r in target_pitch_ranges])

            # Intersection over union of pitch ranges
            intersection = max(0, min(pred_max, tgt_max) - max(pred_min, tgt_min))
            union = max(pred_max, tgt_max) - min(pred_min, tgt_min)

            pitch_range_similarity = intersection / (union + 1e-8)
        else:
            pitch_range_similarity = 0.0

        # Interval distribution similarity (cosine similarity)
        if len(self.interval_histograms) > 0 and len(target_interval_histograms) > 0:
            pred_avg = np.mean(self.interval_histograms, axis=0)
            tgt_avg = np.mean(target_interval_histograms, axis=0)

            # Cosine similarity
            dot_product = np.dot(pred_avg, tgt_avg)
            pred_norm = np.linalg.norm(pred_avg)
            tgt_norm = np.linalg.norm(tgt_avg)

            interval_similarity = dot_product / (pred_norm * tgt_norm + 1e-8)
        else:
            interval_similarity = 0.0

        return {
            'pitch_range_similarity': pitch_range_similarity,
            'interval_similarity': interval_similarity,
        }

    def reset(self):
        self.pitch_ranges = []
        self.interval_histograms = []


class MetricComputer:
    """
    Computes all objective metrics.
    """

    def __init__(self):
        self.perplexity = PerplexityMetric()
        self.harmonic_accuracy = HarmonicAccuracyMetric()
        self.rhythm_consistency = RhythmConsistencyMetric()
        self.voicing_similarity = VoicingSimilarityMetric()

    def update(
        self,
        logits: torch.Tensor,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ):
        """
        Update all metrics.

        Args:
            logits: [batch, seq_len, vocab_size]
            predicted_tokens: [batch, seq_len]
            target_tokens: [batch, seq_len]
        """
        self.perplexity.update(logits, target_tokens)
        self.harmonic_accuracy.update(predicted_tokens, target_tokens)
        self.rhythm_consistency.update(predicted_tokens)
        self.voicing_similarity.update(predicted_tokens, target_tokens)

    def compute(self, target_pitch_ranges=None, target_interval_histograms=None) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}

        # Perplexity
        results['perplexity'] = self.perplexity.compute()

        # Harmonic accuracy
        results['harmonic_accuracy'] = self.harmonic_accuracy.compute()

        # Rhythm consistency
        rhythm_metrics = self.rhythm_consistency.compute()
        results.update(rhythm_metrics)

        # Voicing similarity (if reference data provided)
        if target_pitch_ranges is not None and target_interval_histograms is not None:
            voicing_metrics = self.voicing_similarity.compute(
                target_pitch_ranges, target_interval_histograms
            )
            results.update(voicing_metrics)

        return results

    def reset(self):
        self.perplexity.reset()
        self.harmonic_accuracy.reset()
        self.rhythm_consistency.reset()
        self.voicing_similarity.reset()


if __name__ == "__main__":
    print("Testing Objective Metrics...\n")

    # Create dummy data
    batch_size = 4
    seq_len = 128
    vocab_size = 512

    logits = torch.randn(batch_size, seq_len, vocab_size)
    predicted_tokens = torch.argmax(logits, dim=-1)
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create metric computer
    computer = MetricComputer()

    # Update
    computer.update(logits, predicted_tokens, target_tokens)

    # Compute
    results = computer.compute()

    print("Objective Metrics:")
    print("=" * 50)
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\n✓ All metrics computed successfully!")
