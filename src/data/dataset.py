"""
Dataset classes for JazzFormer-RT training
"""

import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi
from typing import List, Tuple, Optional


class MIDITokenizer:
    """
    Tokenizer for converting MIDI to discrete tokens
    Vocabulary: [NOTE_ON(0-127), NOTE_OFF(128-255), VELOCITY(256-287), TIME_SHIFT(288-387)]
    """

    def __init__(self, vocab_size: int = 388):
        self.vocab_size = vocab_size
        self.note_on_offset = 0
        self.note_off_offset = 128
        self.velocity_offset = 256  # 32 velocity bins
        self.time_offset = 288      # 100 time bins (10ms each)

    def encode_midi(self, midi_path: str, max_length: int = 2048) -> torch.Tensor:
        """
        Convert MIDI file to token sequence

        Args:
            midi_path: path to MIDI file
            max_length: maximum sequence length

        Returns:
            tokens: (seq_len,) tensor of token indices
        """
        midi = pretty_midi.PrettyMIDI(midi_path)

        events = []

        # Extract all note events from piano tracks
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    # Note on event
                    events.append({
                        'time': note.start,
                        'type': 'note_on',
                        'pitch': note.pitch,
                        'velocity': note.velocity
                    })
                    # Note off event
                    events.append({
                        'time': note.end,
                        'type': 'note_off',
                        'pitch': note.pitch,
                        'velocity': 0
                    })

        # Sort events by time
        events.sort(key=lambda x: x['time'])

        # Convert to tokens
        tokens = []
        prev_time = 0

        for event in events:
            # Add time shift token
            time_delta = int((event['time'] - prev_time) * 100)  # 10ms resolution
            time_delta = min(time_delta, 99)  # Cap at 990ms
            if time_delta > 0:
                tokens.append(self.time_offset + time_delta)

            # Add note event token
            if event['type'] == 'note_on':
                tokens.append(self.note_on_offset + event['pitch'])
                # Add velocity token
                velocity_bin = min(event['velocity'] // 4, 31)  # 32 bins
                tokens.append(self.velocity_offset + velocity_bin)
            else:
                tokens.append(self.note_off_offset + event['pitch'])

            prev_time = event['time']

            # Stop if max length reached
            if len(tokens) >= max_length:
                break

        # Pad or truncate
        if len(tokens) < max_length:
            tokens.extend([0] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]

        return torch.tensor(tokens, dtype=torch.long)

    def decode_tokens(self, tokens: torch.Tensor, output_path: str):
        """
        Convert token sequence back to MIDI file

        Args:
            tokens: (seq_len,) tensor of token indices
            output_path: path to save MIDI file
        """
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        active_notes = {}  # pitch -> (start_time, velocity)

        for token in tokens:
            token = token.item()

            if token >= self.time_offset:
                # Time shift
                time_delta = (token - self.time_offset) / 100.0
                current_time += time_delta

            elif token >= self.velocity_offset:
                # Velocity token (accompanies note_on)
                pass  # Handled with note_on

            elif token >= self.note_off_offset:
                # Note off
                pitch = token - self.note_off_offset
                if pitch in active_notes:
                    start_time, velocity = active_notes.pop(pitch)
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    piano.notes.append(note)

            else:
                # Note on
                pitch = token - self.note_on_offset
                velocity = 80  # Default velocity
                active_notes[pitch] = (current_time, velocity)

        midi.instruments.append(piano)
        midi.write(output_path)


class JazzMIDIDataset(Dataset):
    """
    Dataset for loading MIDI files for JazzFormer-RT training
    """

    def __init__(
        self,
        data_dir: str,
        max_length: int = 2048,
        artist_name: Optional[str] = None,
        split: str = 'train'
    ):
        """
        Args:
            data_dir: directory containing MIDI files
            max_length: maximum sequence length
            artist_name: optional artist name for style conditioning
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.max_length = max_length
        self.artist_name = artist_name
        self.split = split

        # Find all MIDI files
        self.midi_files = glob.glob(os.path.join(data_dir, '*.mid')) + \
                         glob.glob(os.path.join(data_dir, '*.midi'))

        print(f"Found {len(self.midi_files)} MIDI files in {data_dir}")

        # Tokenizer
        self.tokenizer = MIDITokenizer()

        # Artist mapping (for style embeddings)
        self.artist_to_id = {
            'brad_mehldau': 0,
            'bill_evans': 1,
            'keith_jarrett': 2,
            'herbie_hancock': 3,
            # Add more artists as needed
        }

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            tokens: (max_length,) - input tokens
            target: (max_length,) - target tokens (shifted by 1)
            artist_id: int - artist identifier
        """
        midi_path = self.midi_files[idx]

        # Encode MIDI to tokens
        tokens = self.tokenizer.encode_midi(midi_path, self.max_length)

        # Create target (next token prediction)
        target = torch.cat([tokens[1:], torch.tensor([0])])

        # Get artist ID
        artist_id = self.artist_to_id.get(self.artist_name, 0)

        return tokens, target, artist_id


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    max_length: int = 2048,
    artist_name: Optional[str] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders

    Args:
        train_dir: directory with training MIDI files
        val_dir: directory with validation MIDI files
        batch_size: batch size
        max_length: maximum sequence length
        artist_name: artist for style conditioning
        num_workers: number of dataloader workers

    Returns:
        train_loader, val_loader
    """
    train_dataset = JazzMIDIDataset(
        train_dir,
        max_length=max_length,
        artist_name=artist_name,
        split='train'
    )

    val_dataset = JazzMIDIDataset(
        val_dir,
        max_length=max_length,
        artist_name=artist_name,
        split='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test tokenizer
    print("Testing MIDI Tokenizer...")
    tokenizer = MIDITokenizer()

    # Create a simple test dataset
    dataset = JazzMIDIDataset(
        data_dir="data/raw_midi",
        max_length=512,
        artist_name="brad_mehldau"
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        tokens, target, artist_id = dataset[0]
        print(f"Tokens shape: {tokens.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Artist ID: {artist_id}")
        print(f"First 10 tokens: {tokens[:10]}")

    print("\n✓ Dataset test passed!")
