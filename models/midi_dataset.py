#!/usr/bin/env python3
"""
MIDI Dataset for PyTorch Music Transformer

MIDI 파일을 토큰 시퀀스로 변환하고 PyTorch Dataset으로 제공
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset


class MIDITokenizer:
    """
    MIDI를 토큰 시퀀스로 변환하는 토크나이저

    Event types:
    - NOTE_ON: 0-127
    - NOTE_OFF: 128-255
    - TIME_SHIFT: 256-383 (0-127 * 10ms)
    - VELOCITY: 384-447 (0-63)
    - Special tokens: 448-511
    """

    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size

        # Token ranges
        self.note_on_offset = 0
        self.note_off_offset = 128
        self.time_shift_offset = 256
        self.velocity_offset = 384

        # Special tokens
        self.pad_token = 448
        self.bos_token = 449
        self.eos_token = 450
        self.mask_token = 451

    def encode(self, midi_path: str, max_length: int = 2048) -> List[int]:
        """
        MIDI 파일을 토큰 시퀀스로 변환

        Args:
            midi_path: MIDI 파일 경로
            max_length: 최대 시퀀스 길이

        Returns:
            tokens: 토큰 리스트
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
            tokens = [self.bos_token]

            # 모든 노트 수집
            all_notes = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    all_notes.extend(instrument.notes)

            # 시간순 정렬
            all_notes.sort(key=lambda x: x.start)

            current_time = 0.0

            for note in all_notes:
                # Time shift
                time_diff = int((note.start - current_time) * 100)  # 10ms 단위
                while time_diff > 0:
                    shift = min(time_diff, 127)
                    tokens.append(self.time_shift_offset + shift)
                    time_diff -= shift

                # Note on
                tokens.append(self.note_on_offset + note.pitch)

                # Velocity
                velocity = min(note.velocity // 2, 63)  # 0-127 -> 0-63
                tokens.append(self.velocity_offset + velocity)

                # Note duration
                duration = int((note.end - note.start) * 100)
                while duration > 0:
                    shift = min(duration, 127)
                    tokens.append(self.time_shift_offset + shift)
                    duration -= shift

                # Note off
                tokens.append(self.note_off_offset + note.pitch)

                current_time = note.end

                # 최대 길이 체크
                if len(tokens) >= max_length - 1:
                    break

            tokens.append(self.eos_token)

            # 패딩 또는 자르기
            if len(tokens) < max_length:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]

            return tokens

        except Exception as e:
            print(f"Error encoding {midi_path}: {e}")
            # 빈 시퀀스 반환
            return [self.bos_token] + [self.pad_token] * (max_length - 1)

    def decode(self, tokens: List[int], output_path: str, tempo: int = 120):
        """
        토큰 시퀀스를 MIDI 파일로 변환

        Args:
            tokens: 토큰 리스트
            output_path: 출력 MIDI 파일 경로
            tempo: BPM
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)

        for token in tokens:
            if token == self.eos_token or token == self.pad_token:
                continue

            # Note on
            if self.note_on_offset <= token < self.note_off_offset:
                pitch = token - self.note_on_offset
                if pitch in active_notes:
                    # Close previous note
                    start_time, velocity = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
                active_notes[pitch] = (current_time, 80)  # Default velocity

            # Note off
            elif self.note_off_offset <= token < self.time_shift_offset:
                pitch = token - self.note_off_offset
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)
                    del active_notes[pitch]

            # Time shift
            elif self.time_shift_offset <= token < self.velocity_offset:
                time_shift = (token - self.time_shift_offset) / 100.0
                current_time += time_shift

            # Velocity
            elif self.velocity_offset <= token < self.pad_token:
                velocity = (token - self.velocity_offset) * 2
                # Update last note's velocity
                if active_notes:
                    last_pitch = list(active_notes.keys())[-1]
                    start_time, _ = active_notes[last_pitch]
                    active_notes[last_pitch] = (start_time, velocity)

        # Close remaining notes
        for pitch, (start_time, velocity) in active_notes.items():
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=current_time + 0.5
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        midi.write(output_path)


class MIDIDataset(Dataset):
    """
    PyTorch Dataset for MIDI files
    """

    def __init__(
        self,
        midi_dir: str,
        max_length: int = 2048,
        augment: bool = True
    ):
        """
        Args:
            midi_dir: MIDI 파일 디렉토리
            max_length: 최대 시퀀스 길이
            augment: 데이터 증강 여부
        """
        self.midi_dir = midi_dir
        self.max_length = max_length
        self.augment = augment

        # MIDI 파일 수집
        self.midi_files = []
        for ext in ['*.mid', '*.midi']:
            self.midi_files.extend(Path(midi_dir).rglob(ext))

        if len(self.midi_files) == 0:
            raise ValueError(f"No MIDI files found in {midi_dir}")

        print(f"Found {len(self.midi_files)} MIDI files")

        # Tokenizer
        self.tokenizer = MIDITokenizer()

        # Pre-encode all files (optional for speed)
        self.encoded_files = []
        print("Encoding MIDI files...")
        for midi_file in self.midi_files:
            tokens = self.tokenizer.encode(str(midi_file), max_length)
            self.encoded_files.append(tokens)

    def __len__(self):
        return len(self.encoded_files)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids: (max_length,) - Input sequence
            labels: (max_length,) - Target sequence (shifted by 1)
        """
        tokens = self.encoded_files[idx].copy()

        # Data augmentation: transpose
        if self.augment and random.random() > 0.5:
            tokens = self._transpose(tokens, random.randint(-5, 5))

        # Convert to tensors
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }

    def _transpose(self, tokens: List[int], semitones: int) -> List[int]:
        """
        Transpose note tokens by semitones

        Args:
            tokens: Token sequence
            semitones: Number of semitones to transpose

        Returns:
            transposed_tokens: Transposed token sequence
        """
        transposed = []
        for token in tokens:
            # Transpose note on/off
            if self.tokenizer.note_on_offset <= token < self.tokenizer.note_off_offset:
                new_pitch = token - self.tokenizer.note_on_offset + semitones
                if 0 <= new_pitch <= 127:
                    transposed.append(self.tokenizer.note_on_offset + new_pitch)
                else:
                    transposed.append(token)  # Keep original if out of range

            elif self.tokenizer.note_off_offset <= token < self.tokenizer.time_shift_offset:
                new_pitch = token - self.tokenizer.note_off_offset + semitones
                if 0 <= new_pitch <= 127:
                    transposed.append(self.tokenizer.note_off_offset + new_pitch)
                else:
                    transposed.append(token)

            else:
                transposed.append(token)

        return transposed


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal attention mask

    Args:
        seq_len: Sequence length

    Returns:
        mask: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = MIDITokenizer()

    # Test encoding/decoding
    print("Testing MIDI tokenization...")

    # Create test dataset
    try:
        dataset = MIDIDataset("data/raw_midi", max_length=512)
        print(f"Dataset size: {len(dataset)}")

        # Get sample
        sample = dataset[0]
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Label shape: {sample['labels'].shape}")

    except ValueError as e:
        print(f"Note: {e}")
        print("Create sample MIDI files first using generate_sample_midi.py")
