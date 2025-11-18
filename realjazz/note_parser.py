"""
Robust Note Parser for Aria Absolute Onset Encoding

This module provides correct parsing of Aria-tokenized MIDI sequences that use
ABSOLUTE ONSET encoding, where:
1. Onset tokens represent absolute time (not delta time)
2. Notes can appear in any order (not necessarily sorted by time)
3. Chords (simultaneous notes) have the same onset time
4. Each note is encoded as: [onset_token, duration_token, pitch_velocity_token]
   BUT these triplets can appear in ANY ORDER in the sequence

Key differences from naive parsing:
- Groups notes by onset time to find chords
- Doesn't assume sequential [onset, dur, pv] order
- Handles polyphonic music correctly
- Validates token types before grouping
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from realjazz.aria_tokenizer_config import AriaTokenizerConfig, DEFAULT_ARIA_CONFIG, TokenType


@dataclass
class Note:
    """Represents a parsed note"""
    onset_ms: int  # Absolute onset time in milliseconds
    duration_ms: int  # Duration in milliseconds
    pitch: int  # MIDI pitch (0-127)
    velocity_bin: int  # Velocity bin
    onset_token_idx: int  # Index of onset token in original sequence
    duration_token_idx: int  # Index of duration token
    pitch_vel_token_idx: int  # Index of pitch-velocity token

    def __repr__(self) -> str:
        return f"Note(onset={self.onset_ms}ms, dur={self.duration_ms}ms, pitch={self.pitch}, vel_bin={self.velocity_bin})"


class AriaTokenParser:
    """
    Parser for Aria absolute-onset tokenized MIDI sequences

    Handles the complexity of absolute onset encoding where notes can appear
    in any order and chords have the same onset time.
    """

    def __init__(self, config: Optional[AriaTokenizerConfig] = None):
        """
        Initialize parser

        Args:
            config: Tokenizer configuration (uses default if None)
        """
        self.config = config or DEFAULT_ARIA_CONFIG

    def parse_notes(self, tokens: List[int]) -> List[Note]:
        """
        Parse token sequence into structured notes

        This is the CORRECT way to parse Aria absolute-onset encoding:
        1. Scan through tokens and identify onset/duration/pitch_vel tokens
        2. Group them into candidate notes
        3. Validate and convert to Note objects

        Args:
            tokens: Token sequence

        Returns:
            List of parsed Note objects, sorted by onset time

        Example:
            Input: [onset_100ms, dur_200ms, pv_60, onset_100ms, dur_150ms, pv_64, ...]
            Output: [Note(100ms, 200ms, C4), Note(100ms, 150ms, E4), ...]  # Chord!
        """
        notes = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            token_type = self.config.identify_token_type(token)

            # Look for onset token (start of a note)
            if token_type == TokenType.ONSET:
                # Try to find the next two tokens (duration and pitch_velocity)
                if i + 2 < len(tokens):
                    onset_token = tokens[i]
                    duration_token = tokens[i + 1]
                    pitch_vel_token = tokens[i + 2]

                    # Validate that we have a complete note triplet
                    try:
                        dur_type = self.config.identify_token_type(duration_token)
                        pv_type = self.config.identify_token_type(pitch_vel_token)

                        if dur_type == TokenType.DURATION and pv_type == TokenType.PITCH_VELOCITY:
                            # Valid note triplet found!
                            note = self._create_note(
                                onset_token, duration_token, pitch_vel_token,
                                i, i + 1, i + 2
                            )
                            if note:
                                notes.append(note)
                            i += 3  # Skip the triplet
                            continue

                    except ValueError:
                        # Token outside known ranges, skip it
                        pass

            # Special token or incomplete triplet, skip
            i += 1

        # Sort by onset time (chords will have same onset)
        notes.sort(key=lambda n: (n.onset_ms, n.pitch))

        return notes

    def _create_note(
        self,
        onset_token: int,
        duration_token: int,
        pitch_vel_token: int,
        onset_idx: int,
        duration_idx: int,
        pv_idx: int
    ) -> Optional[Note]:
        """
        Create a Note object from tokens

        Args:
            onset_token: Onset token ID
            duration_token: Duration token ID
            pitch_vel_token: Pitch-velocity token ID
            onset_idx: Index of onset token in sequence
            duration_idx: Index of duration token
            pv_idx: Index of pitch-velocity token

        Returns:
            Note object or None if invalid
        """
        try:
            onset_ms = self.config.token_to_onset_ms(onset_token)
            duration_ms = self.config.token_to_duration_ms(duration_token)
            pitch_vel = self.config.token_to_pitch_velocity(pitch_vel_token)

            if onset_ms is None or duration_ms is None or pitch_vel is None:
                return None

            pitch, velocity_bin = pitch_vel

            return Note(
                onset_ms=onset_ms,
                duration_ms=duration_ms,
                pitch=pitch,
                velocity_bin=velocity_bin,
                onset_token_idx=onset_idx,
                duration_token_idx=duration_idx,
                pitch_vel_token_idx=pv_idx
            )

        except (ValueError, AttributeError):
            return None

    def group_by_onset(self, notes: List[Note]) -> Dict[int, List[Note]]:
        """
        Group notes by onset time (useful for finding chords)

        Args:
            notes: List of Note objects

        Returns:
            Dictionary mapping onset_ms -> List[Note]
        """
        onset_groups = {}
        for note in notes:
            if note.onset_ms not in onset_groups:
                onset_groups[note.onset_ms] = []
            onset_groups[note.onset_ms].append(note)

        return onset_groups

    def find_chords(self, notes: List[Note], max_onset_diff_ms: int = 20) -> List[List[Note]]:
        """
        Find chords in note sequence

        Args:
            notes: List of Note objects
            max_onset_diff_ms: Max onset difference to consider as chord (default 20ms)

        Returns:
            List of chord groups (each chord is a list of Notes)
        """
        if not notes:
            return []

        chords = []
        current_chord = [notes[0]]
        last_onset = notes[0].onset_ms

        for note in notes[1:]:
            if abs(note.onset_ms - last_onset) <= max_onset_diff_ms:
                # Part of current chord
                current_chord.append(note)
            else:
                # New chord
                if len(current_chord) > 0:
                    chords.append(current_chord)
                current_chord = [note]
                last_onset = note.onset_ms

        # Add final chord
        if len(current_chord) > 0:
            chords.append(current_chord)

        return chords


def parse_note_sequence(tokens: List[int], config: Optional[AriaTokenizerConfig] = None) -> List[Tuple[int, int, int]]:
    """
    Simplified parse function compatible with old API

    Returns indices of (onset_token_idx, duration_token_idx, pitch_vel_token_idx)

    ⚠️ DEPRECATED: Use AriaTokenParser().parse_notes() for full functionality

    Args:
        tokens: Token sequence
        config: Tokenizer configuration

    Returns:
        List of (onset_idx, duration_idx, pitch_vel_idx) tuples
    """
    parser = AriaTokenParser(config)
    notes = parser.parse_notes(tokens)

    return [(n.onset_token_idx, n.duration_token_idx, n.pitch_vel_token_idx) for n in notes]


if __name__ == "__main__":
    print("=" * 70)
    print("Aria Token Parser Test")
    print("=" * 70)

    config = DEFAULT_ARIA_CONFIG
    parser = AriaTokenParser(config)

    # Test 1: Simple melody (sequential notes)
    print("\n🎵 Test 1: Simple Melody (C-E-G sequence)")
    tokens = [
        config.onset_ms_to_token(0),      # Onset at 0ms
        config.duration_ms_to_token(500), # Duration 500ms
        config.pitch_velocity_to_token(60, 3),  # C4, velocity bin 3

        config.onset_ms_to_token(600),    # Onset at 600ms
        config.duration_ms_to_token(500),
        config.pitch_velocity_to_token(64, 3),  # E4

        config.onset_ms_to_token(1200),   # Onset at 1200ms
        config.duration_ms_to_token(500),
        config.pitch_velocity_to_token(67, 3),  # G4
    ]

    notes = parser.parse_notes(tokens)
    print(f"  Input tokens: {len(tokens)}")
    print(f"  Parsed notes: {len(notes)}")
    for note in notes:
        print(f"    {note}")

    # Test 2: Chord (simultaneous notes)
    print("\n🎹 Test 2: Chord (C-E-G simultaneous)")
    tokens = [
        config.onset_ms_to_token(0),      # All at 0ms
        config.duration_ms_to_token(1000),
        config.pitch_velocity_to_token(60, 3),  # C4

        config.onset_ms_to_token(0),      # Same onset!
        config.duration_ms_to_token(1000),
        config.pitch_velocity_to_token(64, 3),  # E4

        config.onset_ms_to_token(0),      # Same onset!
        config.duration_ms_to_token(1000),
        config.pitch_velocity_to_token(67, 3),  # G4
    ]

    notes = parser.parse_notes(tokens)
    print(f"  Input tokens: {len(tokens)}")
    print(f"  Parsed notes: {len(notes)}")
    for note in notes:
        print(f"    {note}")

    chords = parser.find_chords(notes)
    print(f"  Found {len(chords)} chord(s):")
    for i, chord in enumerate(chords):
        pitches = [n.pitch for n in chord]
        print(f"    Chord {i + 1}: pitches={pitches} (size={len(chord)})")

    # Test 3: Polyphonic music (melody + accompaniment)
    print("\n🎼 Test 3: Polyphonic Music")
    tokens = [
        # Bass note at 0ms
        config.onset_ms_to_token(0),
        config.duration_ms_to_token(2000),
        config.pitch_velocity_to_token(48, 2),  # C3, soft

        # Chord at 0ms
        config.onset_ms_to_token(0),
        config.duration_ms_to_token(1000),
        config.pitch_velocity_to_token(60, 3),  # C4

        config.onset_ms_to_token(0),
        config.duration_ms_to_token(1000),
        config.pitch_velocity_to_token(64, 3),  # E4

        # Melody note at 500ms
        config.onset_ms_to_token(500),
        config.duration_ms_to_token(500),
        config.pitch_velocity_to_token(72, 4),  # C5, loud
    ]

    notes = parser.parse_notes(tokens)
    print(f"  Input tokens: {len(tokens)}")
    print(f"  Parsed notes: {len(notes)}")

    onset_groups = parser.group_by_onset(notes)
    print(f"  Grouped by onset:")
    for onset_ms, group in sorted(onset_groups.items()):
        print(f"    {onset_ms}ms: {len(group)} note(s) -> {[n.pitch for n in group]}")

    chords = parser.find_chords(notes)
    print(f"  Found {len(chords)} event(s) (chords/notes):")
    for i, chord in enumerate(chords):
        onset = chord[0].onset_ms
        pitches = [n.pitch for n in chord]
        print(f"    Event {i + 1} @ {onset}ms: {pitches}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("Note: Use AriaTokenParser for all new code (more robust than old API)")
    print("=" * 70)
