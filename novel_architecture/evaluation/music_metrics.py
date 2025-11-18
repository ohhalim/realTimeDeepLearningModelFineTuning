"""
Music Evaluation Metrics

Improvement: Add objective musical quality metrics
- Harmonic coherence (chord recognition accuracy)
- Rhythmic consistency (tempo stability, syncopation)
- Pitch diversity and range
- Voicing quality

These metrics allow quantitative evaluation of generated music.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math


def tokens_to_notes_simple(tokens: torch.Tensor) -> List[Tuple[int, float, float, int]]:
    """Convert tokens to (pitch, onset, duration, velocity) tuples

    Simplified version for evaluation
    """
    NOTE_ON_OFFSET = 0
    TIME_SHIFT_OFFSET = 88
    VELOCITY_OFFSET = 188

    notes = []
    current_time = 0.0
    i = 0

    while i < len(tokens):
        token = tokens[i].item()

        if TIME_SHIFT_OFFSET <= token < VELOCITY_OFFSET:
            # Time shift
            time_delta = (token - TIME_SHIFT_OFFSET) / 100.0
            current_time += time_delta
            i += 1

            if i >= len(tokens):
                break

            # Pitch
            pitch_token = tokens[i].item()
            if NOTE_ON_OFFSET <= pitch_token < TIME_SHIFT_OFFSET:
                pitch = pitch_token - NOTE_ON_OFFSET
            else:
                i += 1
                continue
            i += 1

            if i >= len(tokens):
                break

            # Velocity
            vel_token = tokens[i].item()
            if VELOCITY_OFFSET <= vel_token < 220:
                velocity = (vel_token - VELOCITY_OFFSET) * 4
            else:
                velocity = 64
            i += 1

            notes.append((pitch, current_time, 0.5, velocity))
        else:
            i += 1

    return notes


class HarmonicCoherenceMetric:
    """Measure harmonic coherence via chord recognition

    Good music should have coherent chord progressions.
    We measure:
    1. Chord clarity (how well notes fit into chords)
    2. Voice leading smoothness
    3. Harmonic rhythm regularity
    """
    def __init__(self):
        # Major and minor triads (12 keys each)
        self.chord_templates = self._build_chord_templates()

    def _build_chord_templates(self) -> Dict[str, set]:
        """Build chord templates (pitch class sets)"""
        templates = {}

        # Major triads
        for root in range(12):
            pitches = {root, (root + 4) % 12, (root + 7) % 12}
            templates[f"{root}_major"] = pitches

        # Minor triads
        for root in range(12):
            pitches = {root, (root + 3) % 12, (root + 7) % 12}
            templates[f"{root}_minor"] = pitches

        # Dominant 7th
        for root in range(12):
            pitches = {root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12}
            templates[f"{root}_dom7"] = pitches

        return templates

    def recognize_chord(self, pitches: List[int], time_window: float = 0.5) -> Optional[str]:
        """Recognize chord from simultaneous pitches

        Args:
            pitches: List of MIDI pitches sounding together
            time_window: Time window to consider simultaneous

        Returns:
            Best matching chord name or None
        """
        if len(pitches) < 2:
            return None

        # Convert to pitch classes
        pitch_classes = set(p % 12 for p in pitches)

        # Find best matching chord
        best_match = None
        best_score = 0

        for chord_name, template in self.chord_templates.items():
            # Jaccard similarity
            intersection = len(pitch_classes & template)
            union = len(pitch_classes | template)
            score = intersection / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_match = chord_name

        # Require at least 50% match
        if best_score >= 0.5:
            return best_match
        return None

    def compute_coherence(self, notes: List[Tuple[int, float, float, int]]) -> float:
        """Compute harmonic coherence score [0, 1]

        Higher score = more coherent harmony
        """
        if len(notes) < 4:
            return 0.0

        # Group notes into time windows (0.5 sec)
        window_size = 0.5
        max_time = max(n[1] + n[2] for n in notes)
        num_windows = int(max_time / window_size) + 1

        chord_clarity_scores = []

        for w in range(num_windows):
            window_start = w * window_size
            window_end = (w + 1) * window_size

            # Get pitches in this window
            pitches = [
                n[0] for n in notes
                if n[1] < window_end and (n[1] + n[2]) > window_start
            ]

            if len(pitches) >= 2:
                chord = self.recognize_chord(pitches)
                if chord is not None:
                    chord_clarity_scores.append(1.0)
                else:
                    chord_clarity_scores.append(0.0)

        if not chord_clarity_scores:
            return 0.0

        return sum(chord_clarity_scores) / len(chord_clarity_scores)


class RhythmicConsistencyMetric:
    """Measure rhythmic consistency

    Good music should have:
    1. Stable tempo
    2. Regular metric structure
    3. Appropriate syncopation
    """
    def compute_tempo_stability(self, notes: List[Tuple[int, float, float, int]]) -> float:
        """Measure tempo stability via inter-onset intervals

        Returns:
            Stability score [0, 1], where 1 = perfectly stable
        """
        if len(notes) < 4:
            return 0.0

        # Get onset times
        onsets = sorted([n[1] for n in notes])

        # Compute inter-onset intervals (IOI)
        iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]

        if not iois:
            return 0.0

        # Measure coefficient of variation (lower = more stable)
        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)

        if mean_ioi == 0:
            return 0.0

        cv = std_ioi / mean_ioi

        # Convert to [0, 1] score (CV of 0 = perfect, CV of 1 = very unstable)
        stability = 1.0 / (1.0 + cv)

        return stability

    def compute_metric_strength(self, notes: List[Tuple[int, float, float, int]]) -> float:
        """Measure metric strength (how well aligned to beat grid)

        Returns:
            Metric strength [0, 1]
        """
        if len(notes) < 8:
            return 0.0

        # Estimate tempo from inter-onset intervals
        onsets = sorted([n[1] for n in notes])
        iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
        median_ioi = np.median(iois)

        if median_ioi == 0:
            return 0.0

        # Assume beat = median IOI
        beat_period = median_ioi

        # Measure how many onsets fall near beat grid
        on_beat_count = 0

        for onset in onsets:
            # Find nearest beat
            beat_number = round(onset / beat_period)
            expected_time = beat_number * beat_period

            # Tolerance: within 10% of beat period
            tolerance = beat_period * 0.1

            if abs(onset - expected_time) < tolerance:
                on_beat_count += 1

        metric_strength = on_beat_count / len(onsets)
        return metric_strength


class PitchDiversityMetric:
    """Measure pitch diversity and range"""

    def compute_pitch_range(self, notes: List[Tuple[int, float, float, int]]) -> int:
        """Compute pitch range in semitones"""
        if not notes:
            return 0

        pitches = [n[0] for n in notes]
        return max(pitches) - min(pitches)

    def compute_pitch_diversity(self, notes: List[Tuple[int, float, float, int]]) -> float:
        """Compute pitch class diversity [0, 1]

        Uses Shannon entropy of pitch class distribution
        """
        if not notes:
            return 0.0

        # Get pitch classes
        pitch_classes = [n[0] % 12 for n in notes]

        # Count occurrences
        counts = Counter(pitch_classes)
        total = len(pitch_classes)

        # Compute Shannon entropy
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        # Normalize by max entropy (log2(12) for 12 pitch classes)
        max_entropy = math.log2(12)
        diversity = entropy / max_entropy

        return diversity


class VoicingQualityMetric:
    """Measure voicing quality (for piano music)"""

    def compute_voicing_quality(self, notes: List[Tuple[int, float, float, int]]) -> float:
        """Compute voicing quality based on:
        - Interval distribution (prefer consonant intervals)
        - Voice spacing (not too wide in low register)
        - Doubling appropriateness

        Returns:
            Quality score [0, 1]
        """
        if len(notes) < 4:
            return 0.0

        # Get simultaneous note groups (chords)
        chords = self._extract_chords(notes)

        if not chords:
            return 0.0

        quality_scores = []

        for chord_pitches in chords:
            if len(chord_pitches) < 2:
                continue

            sorted_pitches = sorted(chord_pitches)

            # Check intervals
            intervals = [sorted_pitches[i+1] - sorted_pitches[i]
                        for i in range(len(sorted_pitches)-1)]

            # Prefer consonant intervals (3, 4, 5, 7 semitones)
            consonant_intervals = {3, 4, 5, 7, 8, 9}
            consonance_score = sum(1 for i in intervals if i in consonant_intervals) / len(intervals)

            # Check spacing (shouldn't have >12 semitones in bass)
            if sorted_pitches[0] < 60:  # Below middle C
                wide_spacing_penalty = sum(1 for i in intervals[:2] if i > 12) / min(len(intervals), 2)
                spacing_score = 1.0 - wide_spacing_penalty
            else:
                spacing_score = 1.0

            # Combine
            chord_quality = (consonance_score + spacing_score) / 2
            quality_scores.append(chord_quality)

        return np.mean(quality_scores)

    def _extract_chords(self, notes: List[Tuple[int, float, float, int]], window: float = 0.1) -> List[List[int]]:
        """Extract simultaneous notes (chords)"""
        if not notes:
            return []

        # Sort by onset
        sorted_notes = sorted(notes, key=lambda n: n[1])

        chords = []
        current_chord = [sorted_notes[0][0]]
        current_onset = sorted_notes[0][1]

        for i in range(1, len(sorted_notes)):
            pitch, onset, _, _ = sorted_notes[i]

            if abs(onset - current_onset) < window:
                # Same chord
                current_chord.append(pitch)
            else:
                # New chord
                if len(current_chord) >= 2:
                    chords.append(current_chord)
                current_chord = [pitch]
                current_onset = onset

        # Add last chord
        if len(current_chord) >= 2:
            chords.append(current_chord)

        return chords


def evaluate_music_quality(tokens: torch.Tensor) -> Dict[str, float]:
    """Evaluate musical quality of generated sequence

    Args:
        tokens: Generated token sequence [seq_len]

    Returns:
        Dictionary of metric scores
    """
    # Convert tokens to notes
    notes = tokens_to_notes_simple(tokens)

    if len(notes) < 4:
        return {
            'num_notes': len(notes),
            'harmonic_coherence': 0.0,
            'tempo_stability': 0.0,
            'metric_strength': 0.0,
            'pitch_range': 0,
            'pitch_diversity': 0.0,
            'voicing_quality': 0.0,
        }

    # Compute metrics
    harmonic = HarmonicCoherenceMetric()
    rhythmic = RhythmicConsistencyMetric()
    pitch = PitchDiversityMetric()
    voicing = VoicingQualityMetric()

    metrics = {
        'num_notes': len(notes),
        'harmonic_coherence': harmonic.compute_coherence(notes),
        'tempo_stability': rhythmic.compute_tempo_stability(notes),
        'metric_strength': rhythmic.compute_metric_strength(notes),
        'pitch_range': pitch.compute_pitch_range(notes),
        'pitch_diversity': pitch.compute_pitch_diversity(notes),
        'voicing_quality': voicing.compute_voicing_quality(notes),
    }

    # Overall quality (weighted average)
    metrics['overall_quality'] = (
        0.3 * metrics['harmonic_coherence'] +
        0.2 * metrics['tempo_stability'] +
        0.1 * metrics['metric_strength'] +
        0.2 * metrics['pitch_diversity'] +
        0.2 * metrics['voicing_quality']
    )

    return metrics


def test_music_metrics():
    """Test music evaluation metrics"""
    print("Testing Music Evaluation Metrics...")

    # Create synthetic note sequence (C major scale)
    notes_good = [
        (60, 0.0, 0.5, 80),   # C
        (62, 0.5, 0.5, 75),   # D
        (64, 1.0, 0.5, 70),   # E
        (65, 1.5, 0.5, 75),   # F
        (67, 2.0, 0.5, 80),   # G
        (69, 2.5, 0.5, 75),   # A
        (71, 3.0, 0.5, 70),   # B
        (72, 3.5, 0.5, 85),   # C
    ]

    # Bad sequence (random pitches, irregular timing)
    notes_bad = [
        (40, 0.0, 0.5, 80),
        (73, 0.7, 0.5, 90),
        (55, 1.3, 0.5, 60),
        (88, 2.1, 0.5, 100),
    ]

    print("\n1. Testing Harmonic Coherence...")
    harmonic = HarmonicCoherenceMetric()
    coherence_good = harmonic.compute_coherence(notes_good)
    coherence_bad = harmonic.compute_coherence(notes_bad)
    print(f"  Good sequence: {coherence_good:.3f}")
    print(f"  Bad sequence: {coherence_bad:.3f}")
    assert coherence_good > coherence_bad, "Good should have higher coherence!"

    print("\n2. Testing Rhythmic Consistency...")
    rhythmic = RhythmicConsistencyMetric()
    tempo_good = rhythmic.compute_tempo_stability(notes_good)
    tempo_bad = rhythmic.compute_tempo_stability(notes_bad)
    print(f"  Tempo stability (good): {tempo_good:.3f}")
    print(f"  Tempo stability (bad): {tempo_bad:.3f}")
    assert tempo_good > tempo_bad, "Good should be more stable!"

    metric_good = rhythmic.compute_metric_strength(notes_good)
    metric_bad = rhythmic.compute_metric_strength(notes_bad)
    print(f"  Metric strength (good): {metric_good:.3f}")
    print(f"  Metric strength (bad): {metric_bad:.3f}")

    print("\n3. Testing Pitch Diversity...")
    pitch_metric = PitchDiversityMetric()
    diversity_good = pitch_metric.compute_pitch_diversity(notes_good)
    range_good = pitch_metric.compute_pitch_range(notes_good)
    print(f"  Pitch diversity: {diversity_good:.3f}")
    print(f"  Pitch range: {range_good} semitones")

    print("\n4. Testing Voicing Quality...")
    voicing_metric = VoicingQualityMetric()

    # Good voicing (C major triad in root position)
    chord_good = [(60, 0.0, 1.0, 80), (64, 0.0, 1.0, 75), (67, 0.0, 1.0, 70)]
    quality_good = voicing_metric.compute_voicing_quality(chord_good)
    print(f"  Good voicing (C major triad): {quality_good:.3f}")

    # Bad voicing (wide intervals in bass)
    chord_bad = [(40, 0.0, 1.0, 80), (60, 0.0, 1.0, 75), (80, 0.0, 1.0, 70)]
    quality_bad = voicing_metric.compute_voicing_quality(chord_bad)
    print(f"  Bad voicing (wide spacing): {quality_bad:.3f}")

    print("\n5. Testing overall evaluation...")
    # Create dummy tokens (simplified)
    tokens_good = torch.tensor([88, 60, 188,  # C
                                 93, 62, 190,  # D (after 0.5s)
                                 93, 64, 188,  # E
                                 93, 65, 190])  # F

    metrics = evaluate_music_quality(tokens_good)
    print(f"\n  Evaluation results:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"    {name}: {value:.3f}")
        else:
            print(f"    {name}: {value}")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_music_metrics()
