"""
Simple data loading for MIDI files
~100 lines
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class SimpleMIDIDataset(Dataset):
    """
    Ultra-simple MIDI dataset.
    Just converts MIDI notes to sequences.
    """
    
    def __init__(self, midi_files: list, seq_len: int = 128):
        """
        Args:
            midi_files: List of MIDI file paths (or just note sequences)
            seq_len: Sequence length
        """
        self.seq_len = seq_len
        self.sequences = []
        
        # For now, generate synthetic data
        # In real version, would load actual MIDI
        print(f"Generating {len(midi_files)} synthetic sequences...")
        for i in range(len(midi_files)):
            # Synthetic jazz-like sequence
            # Uses common jazz notes (C major scale + chromatic passing tones)
            seq = self._generate_jazz_sequence(seq_len + 1)
            self.sequences.append(seq)
    
    def _generate_jazz_sequence(self, length: int) -> torch.Tensor:
        """Generate synthetic jazz-like MIDI sequence"""
        # Jazz commonly uses: C D E F G A B + chromatic alterations
        # MIDI: C4=60, so use range 48-84 (C3 to C6)
        jazz_notes = [48, 50, 52, 53, 55, 57, 59,  # C major scale
                      49, 51, 54, 56, 58, 61, 63]  # Chromatic notes
        
        seq = []
        for _ in range(length):
            # 70% from scale, 30% chromatic
            if np.random.rand() < 0.7:
                note = np.random.choice(jazz_notes[:7])
            else:
                note = np.random.choice(jazz_notes[7:])
            
            # Add some octave variation
            note += np.random.choice([0, 12, -12]) * (np.random.rand() < 0.3)
            note = np.clip(note, 0, 127)
            seq.append(int(note))
        
        return torch.tensor(seq, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            input: (seq_len,) - input sequence
            target: (seq_len,) - target sequence (shifted by 1)
        """
        seq = self.sequences[idx]
        
        input_seq = seq[:-1]
        target_seq = seq[1:]
        
        return input_seq, target_seq


def create_dataloaders(
    num_train: int = 10,
    num_val: int = 2,
    batch_size: int = 4,
    seq_len: int = 128
):
    """
    Create train and val dataloaders
    
    Args:
        num_train: Number of training sequences
        num_val: Number of validation sequences
        batch_size: Batch size
        seq_len: Sequence length
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = SimpleMIDIDataset(
        midi_files=[f"train_{i}" for i in range(num_train)],
        seq_len=seq_len
    )
    
    val_dataset = SimpleMIDIDataset(
        midi_files=[f"val_{i}" for i in range(num_val)],
        seq_len=seq_len
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader


# Test
if __name__ == "__main__":
    print("Testing data loading...")
    
    train_loader, val_loader = create_dataloaders(
        num_train=10,
        num_val=2,
        batch_size=4,
        seq_len=128
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Get one batch
    for inputs, targets in train_loader:
        print(f"✓ Batch shape: {inputs.shape}")
        print(f"✓ Sample notes: {inputs[0, :10].tolist()}")
        break
    
    print("\n✓ Data loading works!")
