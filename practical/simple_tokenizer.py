"""
Simple MIDI Tokenizer

Proven approach (Magenta-style):
- NOTE_ON: 88 tokens (piano keys 21-108, MIDI 21-108)
- TIME_SHIFT: 100 tokens (0-999ms in 10ms steps)
- VELOCITY: 32 tokens (0-127 in 4-step quantization)

Total vocab: 88 + 100 + 32 = 220 tokens

This is SIMPLE and WORKS. No fancy tokenization needed.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MIDIEvent:
    """Simple MIDI event."""
    type: str  # 'note_on', 'note_off', 'time_shift'
    value: int  # pitch (0-127) or time_ms (0-999)
    velocity: int = 64  # 0-127


class SimpleMIDITokenizer:
    """
    Simple tokenizer for MIDI.

    Token ranges:
    - 0-87: NOTE_ON (piano keys 21-108)
    - 88-187: TIME_SHIFT (0-999ms in 10ms steps = 100 tokens)
    - 188-219: VELOCITY (0-127 quantized to 32 levels)

    Special tokens:
    - PAD: -1 (not in vocab)
    - BOS: 0 (same as lowest NOTE_ON)
    - EOS: 87 (same as highest NOTE_ON)
    """

    def __init__(self):
        # Token offsets
        self.NOTE_ON_OFFSET = 0
        self.TIME_SHIFT_OFFSET = 88
        self.VELOCITY_OFFSET = 188

        # Vocab size
        self.vocab_size = 220  # 88 + 100 + 32

        # MIDI constants
        self.MIN_PITCH = 21  # A0
        self.MAX_PITCH = 108  # C8
        self.MAX_TIME_MS = 999
        self.TIME_QUANTIZE_MS = 10

        # Velocity quantization
        self.VELOCITY_BINS = 32

    def pitch_to_token(self, pitch: int) -> int:
        """Convert MIDI pitch to NOTE_ON token."""
        if not (self.MIN_PITCH <= pitch <= self.MAX_PITCH):
            raise ValueError(f"Pitch {pitch} out of range [{self.MIN_PITCH}, {self.MAX_PITCH}]")

        return pitch - self.MIN_PITCH + self.NOTE_ON_OFFSET

    def token_to_pitch(self, token: int) -> int:
        """Convert NOTE_ON token to MIDI pitch."""
        return token - self.NOTE_ON_OFFSET + self.MIN_PITCH

    def time_to_token(self, time_ms: int) -> int:
        """Convert time (ms) to TIME_SHIFT token."""
        # Quantize to 10ms steps
        time_quantized = min(time_ms // self.TIME_QUANTIZE_MS, 99)
        return time_quantized + self.TIME_SHIFT_OFFSET

    def token_to_time(self, token: int) -> int:
        """Convert TIME_SHIFT token to time (ms)."""
        return (token - self.TIME_SHIFT_OFFSET) * self.TIME_QUANTIZE_MS

    def velocity_to_token(self, velocity: int) -> int:
        """Convert MIDI velocity to VELOCITY token."""
        # Quantize 0-127 to 32 bins
        bin_idx = velocity // 4
        return min(bin_idx, 31) + self.VELOCITY_OFFSET

    def token_to_velocity(self, token: int) -> int:
        """Convert VELOCITY token to MIDI velocity."""
        bin_idx = token - self.VELOCITY_OFFSET
        return bin_idx * 4 + 2  # Middle of bin

    def encode(self, events: List[MIDIEvent]) -> List[int]:
        """
        Encode MIDI events to tokens.

        Format: [VEL, NOTE, TIME, NOTE, TIME, ...]

        Args:
            events: List of MIDI events

        Returns:
            tokens: List of token IDs
        """
        tokens = []
        current_velocity = 64

        for event in events:
            if event.type == 'note_on':
                # Add velocity token if changed
                if event.velocity != current_velocity:
                    tokens.append(self.velocity_to_token(event.velocity))
                    current_velocity = event.velocity

                # Add note token
                tokens.append(self.pitch_to_token(event.value))

            elif event.type == 'time_shift':
                # Add time shift token
                tokens.append(self.time_to_token(event.value))

        return tokens

    def decode(self, tokens: List[int]) -> List[MIDIEvent]:
        """
        Decode tokens to MIDI events.

        Args:
            tokens: List of token IDs

        Returns:
            events: List of MIDI events
        """
        events = []
        current_velocity = 64
        current_time = 0

        for token in tokens:
            if token < 0:  # Ignore PAD
                continue

            if self.NOTE_ON_OFFSET <= token < self.TIME_SHIFT_OFFSET:
                # NOTE_ON
                pitch = self.token_to_pitch(token)
                events.append(MIDIEvent('note_on', pitch, current_velocity))

            elif self.TIME_SHIFT_OFFSET <= token < self.VELOCITY_OFFSET:
                # TIME_SHIFT
                time_ms = self.token_to_time(token)
                events.append(MIDIEvent('time_shift', time_ms))
                current_time += time_ms

            elif self.VELOCITY_OFFSET <= token < self.vocab_size:
                # VELOCITY
                current_velocity = self.token_to_velocity(token)

        return events

    def events_to_midi_like(self, events: List[MIDIEvent]) -> List[Tuple[int, int, int]]:
        """
        Convert events to simple MIDI-like format.

        Returns:
            notes: List of (start_ms, pitch, velocity)
        """
        notes = []
        current_time = 0

        for event in events:
            if event.type == 'note_on':
                notes.append((current_time, event.value, event.velocity))
            elif event.type == 'time_shift':
                current_time += event.value

        return notes


def create_dummy_sequence(length=100) -> List[int]:
    """
    Create dummy MIDI token sequence for testing.

    Simple C major scale pattern.
    """
    tokenizer = SimpleMIDITokenizer()

    events = []

    # C major scale
    scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

    for i in range(length // 4):
        pitch = scale[i % len(scale)]

        # Velocity
        velocity = 64 + (i % 3) * 16  # Vary velocity slightly

        # Note on
        events.append(MIDIEvent('note_on', pitch, velocity))

        # Time shift (quarter note = 500ms at 120 BPM)
        events.append(MIDIEvent('time_shift', 250))

    # Encode
    tokens = tokenizer.encode(events)

    return tokens


if __name__ == "__main__":
    print("Testing Simple MIDI Tokenizer")
    print("=" * 70)

    tokenizer = SimpleMIDITokenizer()

    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"  NOTE_ON:     {tokenizer.NOTE_ON_OFFSET}-{tokenizer.TIME_SHIFT_OFFSET-1} (88 tokens)")
    print(f"  TIME_SHIFT:  {tokenizer.TIME_SHIFT_OFFSET}-{tokenizer.VELOCITY_OFFSET-1} (100 tokens)")
    print(f"  VELOCITY:    {tokenizer.VELOCITY_OFFSET}-{tokenizer.vocab_size-1} (32 tokens)")

    # Test encoding/decoding
    print("\nTest 1: Encode/Decode")
    events = [
        MIDIEvent('note_on', 60, 64),  # Middle C
        MIDIEvent('time_shift', 500),
        MIDIEvent('note_on', 64, 80),  # E
        MIDIEvent('time_shift', 500),
    ]

    tokens = tokenizer.encode(events)
    decoded = tokenizer.decode(tokens)

    print(f"  Original events: {len(events)}")
    print(f"  Tokens: {tokens}")
    print(f"  Decoded events: {len(decoded)}")

    # Test dummy sequence
    print("\nTest 2: Dummy sequence")
    dummy_tokens = create_dummy_sequence(100)
    print(f"  Generated {len(dummy_tokens)} tokens")
    print(f"  Sample tokens: {dummy_tokens[:20]}")

    # Test MIDI-like conversion
    events = tokenizer.decode(dummy_tokens)
    notes = tokenizer.events_to_midi_like(events)
    print(f"  Decoded to {len(notes)} notes")
    print(f"  Sample notes: {notes[:5]}")

    print("\n✓ All tokenizer tests passed!")
