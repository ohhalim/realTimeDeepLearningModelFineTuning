"""
Corruption Functions for JazzFormer-CR
Six corruption types for data augmentation and flexible generation
"""

import random
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class MIDINote:
    """MIDI note representation"""
    pitch: int        # 0-127
    onset: float      # seconds (absolute)
    duration: float   # seconds
    velocity: int     # 0-127


class Corruption:
    """Base class for corruption functions"""
    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty  # 1=easy, 2=medium, 3=hard, 4=very hard

    def __call__(self, notes: List[MIDINote]) -> List[MIDINote]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(difficulty={self.difficulty})"


class NoCorrupt(Corruption):
    """Identity function - no corruption (baseline)"""
    def __init__(self):
        super().__init__(difficulty=1)

    def __call__(self, notes: List[MIDINote]) -> List[MIDINote]:
        return notes.copy()


class TimeCrop(Corruption):
    """Remove temporal segments - for continuation/infilling tasks

    Modes:
        - 'start': Remove beginning (for continuation)
        - 'end': Remove ending (for backward generation)
        - 'middle': Remove middle section (for infilling)
        - 'random': Randomly choose mode
    """
    def __init__(self, crop_ratio: float = 0.3, mode: str = 'random'):
        super().__init__(difficulty=3)
        self.crop_ratio = crop_ratio
        self.mode = mode

    def __call__(self, notes: List[MIDINote]) -> List[MIDINote]:
        if not notes:
            return notes

        total_duration = max(n.onset + n.duration for n in notes)
        crop_duration = total_duration * self.crop_ratio

        # Choose mode
        if self.mode == 'random':
            mode = random.choice(['start', 'end', 'middle'])
        else:
            mode = self.mode

        # Apply cropping
        if mode == 'start':
            # Remove beginning → model learns continuation
            return [n for n in notes if n.onset >= crop_duration]

        elif mode == 'end':
            # Remove ending → model learns backward generation
            crop_start = total_duration - crop_duration
            return [n for n in notes if n.onset < crop_start]

        else:  # middle
            # Remove middle → model learns infilling
            crop_start = (total_duration - crop_duration) / 2
            crop_end = crop_start + crop_duration
            return [n for n in notes if n.onset < crop_start or n.onset > crop_end]


class NoteCrop(Corruption):
    """Remove random notes - for harmonization/arrangement tasks"""
    def __init__(self, dropout_rate: float = 0.4):
        super().__init__(difficulty=3)
        self.dropout_rate = dropout_rate

    def __call__(self, notes: List[MIDINote]) -> List[MIDINote]:
        return [n for n in notes if random.random() > self.dropout_rate]


class GenreChange(Corruption):
    """Add genre conditioning tokens - for style transfer (HARDEST)

    This is the most important corruption for jazz piano generation.
    Model learns to translate classical piano → jazz style.
    """
    def __init__(self, source: str = 'classical', target: str = 'jazz'):
        super().__init__(difficulty=4)  # Hardest task
        self.source = source
        self.target = target

        # Special token IDs (added to vocabulary)
        self.genre_tokens = {
            'classical': 220,
            'jazz': 221,
            'blues': 222,
            'pop': 223,
        }

    def __call__(self, notes: List[MIDINote]) -> Tuple[List[MIDINote], int, int]:
        """Returns (notes, source_token, target_token)"""
        return (
            notes.copy(),
            self.genre_tokens[self.source],
            self.genre_tokens[self.target]
        )


class PitchDropout(Corruption):
    """Zero out note pitches - keep rhythm, remove melody

    Model learns to generate melody from rhythm templates.
    Useful for rhythm-driven jazz improvisation.
    """
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__(difficulty=2)
        self.dropout_rate = dropout_rate
        self.unknown_pitch = 128  # Special token for "unknown pitch"

    def __call__(self, notes: List[MIDINote]) -> List[MIDINote]:
        corrupted = []
        for n in notes:
            if random.random() < self.dropout_rate:
                # Replace pitch with unknown token
                corrupted.append(MIDINote(
                    self.unknown_pitch, n.onset, n.duration, n.velocity
                ))
            else:
                corrupted.append(n)
        return corrupted


class VelocityDropout(Corruption):
    """Zero out dynamics - keep notes, remove expression

    Model learns to add expressive dynamics to mechanical performance.
    Important for jazz articulation (accents, ghost notes, etc.)
    """
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__(difficulty=2)
        self.dropout_rate = dropout_rate
        self.default_velocity = 64  # mf (mezzo-forte)

    def __call__(self, notes: List[MIDINote]) -> List[MIDINote]:
        corrupted = []
        for n in notes:
            if random.random() < self.dropout_rate:
                # Set velocity to default
                corrupted.append(MIDINote(
                    n.pitch, n.onset, n.duration, self.default_velocity
                ))
            else:
                corrupted.append(n)
        return corrupted


# Factory function
def get_all_corruptions() -> List[Corruption]:
    """Get all 6 corruption functions"""
    return [
        NoCorrupt(),
        TimeCrop(crop_ratio=0.3, mode='random'),
        NoteCrop(dropout_rate=0.4),
        GenreChange(source='classical', target='jazz'),
        PitchDropout(dropout_rate=0.5),
        VelocityDropout(dropout_rate=0.3),
    ]


def get_corruption_by_name(name: str) -> Corruption:
    """Get corruption by class name"""
    corruption_map = {c.__class__.__name__: c for c in get_all_corruptions()}
    return corruption_map.get(name)


# Tokenization helpers
def notes_to_tokens(notes: List[MIDINote], max_length: int = 512) -> torch.Tensor:
    """Convert MIDI notes to token sequence (simplified Aria-style)

    Token format:
        - 0-87: NOTE_ON (88 pitches)
        - 88-187: TIME_SHIFT (100 bins, 0-1000ms in 10ms increments)
        - 188-219: VELOCITY (32 bins)
        - 220-223: GENRE tokens
        - 224: PAD
        - 225: UNKNOWN_PITCH (for PitchDropout)
    """
    NOTE_ON_OFFSET = 0
    TIME_SHIFT_OFFSET = 88
    VELOCITY_OFFSET = 188
    PAD_TOKEN = 224

    tokens = []
    prev_onset = 0.0

    for note in sorted(notes, key=lambda n: n.onset):
        # Time shift since previous note (in 10ms increments)
        time_delta = note.onset - prev_onset
        time_bins = min(int(time_delta * 100), 99)  # 0-990ms
        tokens.append(TIME_SHIFT_OFFSET + time_bins)

        # Pitch (or unknown if PitchDropout)
        if note.pitch == 128:  # Unknown pitch marker
            tokens.append(225)
        else:
            tokens.append(NOTE_ON_OFFSET + note.pitch)

        # Velocity (quantize to 32 bins)
        vel_bin = note.velocity // 4
        tokens.append(VELOCITY_OFFSET + vel_bin)

        prev_onset = note.onset

    # Pad or truncate to max_length
    if len(tokens) < max_length:
        tokens += [PAD_TOKEN] * (max_length - len(tokens))
    else:
        tokens = tokens[:max_length]

    return torch.tensor(tokens, dtype=torch.long)


def tokens_to_notes(tokens: torch.Tensor) -> List[MIDINote]:
    """Convert token sequence back to MIDI notes"""
    NOTE_ON_OFFSET = 0
    TIME_SHIFT_OFFSET = 88
    VELOCITY_OFFSET = 188
    PAD_TOKEN = 224
    UNKNOWN_PITCH = 225

    notes = []
    current_time = 0.0
    i = 0

    while i < len(tokens):
        token = tokens[i].item()

        if token == PAD_TOKEN:
            break

        # Expect: TIME_SHIFT → NOTE_ON → VELOCITY
        if TIME_SHIFT_OFFSET <= token < VELOCITY_OFFSET:
            # Time shift
            time_delta = (token - TIME_SHIFT_OFFSET) / 100.0  # 10ms increments
            current_time += time_delta
            i += 1

            if i >= len(tokens):
                break

            # Note pitch
            pitch_token = tokens[i].item()
            if pitch_token == UNKNOWN_PITCH:
                pitch = 60  # Default to middle C if unknown
            elif NOTE_ON_OFFSET <= pitch_token < TIME_SHIFT_OFFSET:
                pitch = pitch_token - NOTE_ON_OFFSET
            else:
                i += 1
                continue
            i += 1

            if i >= len(tokens):
                break

            # Velocity
            vel_token = tokens[i].item()
            if VELOCITY_OFFSET <= vel_token < PAD_TOKEN:
                velocity = (vel_token - VELOCITY_OFFSET) * 4
            else:
                velocity = 64
            i += 1

            # Create note (default duration = 0.5s)
            note = MIDINote(pitch, current_time, 0.5, velocity)
            notes.append(note)
        else:
            i += 1

    return notes


if __name__ == '__main__':
    """Test corruption functions"""
    # Create sample notes
    notes = [
        MIDINote(60, 0.0, 0.5, 80),   # C4
        MIDINote(64, 0.5, 0.5, 70),   # E4
        MIDINote(67, 1.0, 0.5, 75),   # G4
        MIDINote(72, 1.5, 0.5, 85),   # C5
    ]

    print("Original notes:", len(notes))

    # Test each corruption
    for corruption in get_all_corruptions():
        result = corruption(notes)

        if isinstance(corruption, GenreChange):
            corrupted_notes, src_token, tgt_token = result
            print(f"{corruption}: {len(corrupted_notes)} notes, "
                  f"source={src_token}, target={tgt_token}")
        else:
            print(f"{corruption}: {len(result)} notes")

    # Test tokenization
    print("\nTesting tokenization...")
    tokens = notes_to_tokens(notes, max_length=32)
    print(f"Tokens: {tokens.shape}")
    print(f"First 10 tokens: {tokens[:10].tolist()}")

    # Round-trip
    reconstructed = tokens_to_notes(tokens)
    print(f"Reconstructed: {len(reconstructed)} notes")
    for i, note in enumerate(reconstructed[:4]):
        print(f"  Note {i}: pitch={note.pitch}, onset={note.onset:.2f}, vel={note.velocity}")
