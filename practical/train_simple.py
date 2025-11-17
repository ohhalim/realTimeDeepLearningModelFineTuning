"""
Simple Training Script - Works Out of Box

No complex setup. Just run:
  python train_simple.py

Will use dummy data if no real MIDI available.
Proven training loop. No fancy tricks.
"""

import os
import argparse
from pathlib import Path
import json
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import our simple model and tokenizer
from simple_model import create_model
from simple_tokenizer import SimpleMIDITokenizer, create_dummy_sequence


def set_seed(seed=42):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleMIDIDataset(Dataset):
    """
    Simple dataset for MIDI tokens.

    If no data_dir provided, generates dummy data.
    """

    def __init__(self, seq_len=512, num_sequences=1000, data_dir=None):
        self.seq_len = seq_len
        self.tokenizer = SimpleMIDITokenizer()

        # Load or generate data
        if data_dir and Path(data_dir).exists():
            # TODO: Load real MIDI files
            # For now, use dummy data
            print(f"  ⚠️  Real MIDI loading not implemented yet")
            print(f"  Using dummy data for demonstration")
            self.sequences = self._generate_dummy_data(num_sequences)
        else:
            print(f"  Using dummy data ({num_sequences} sequences)")
            self.sequences = self._generate_dummy_data(num_sequences)

    def _generate_dummy_data(self, num_sequences):
        """Generate dummy MIDI sequences."""
        sequences = []

        for _ in range(num_sequences):
            # Create sequence slightly longer than seq_len
            tokens = create_dummy_sequence(self.seq_len + 100)

            # Random crop to seq_len
            if len(tokens) > self.seq_len:
                start = random.randint(0, len(tokens) - self.seq_len)
                tokens = tokens[start:start + self.seq_len]

            # Pad if too short
            while len(tokens) < self.seq_len:
                tokens.append(-1)  # PAD token

            sequences.append(tokens)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]

        # Convert to tensor
        x = torch.tensor(tokens[:-1], dtype=torch.long)  # Input
        y = torch.tensor(tokens[1:], dtype=torch.long)   # Target (shifted by 1)

        return x, y


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(progress_bar):
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()

    total_loss = 0
    num_batches = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits, loss = model(x, y)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)

    return avg_loss, perplexity


def main(args):
    print("=" * 70)
    print("Simple Training Script - Brad Mehldau Jazz Piano")
    print("=" * 70)

    # Set seed
    set_seed(args.seed)
    print(f"\n✓ Random seed set to {args.seed}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")

    # Save config
    config_dict = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Config saved")

    # Create datasets
    print(f"\n{'='*70}")
    print("Creating datasets...")
    print(f"{'='*70}")

    train_dataset = SimpleMIDIDataset(
        seq_len=args.seq_len,
        num_sequences=args.num_train,
        data_dir=args.data_dir
    )

    val_dataset = SimpleMIDIDataset(
        seq_len=args.seq_len,
        num_sequences=args.num_val,
        data_dir=args.data_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"✓ Train dataset: {len(train_dataset)} sequences")
    print(f"✓ Val dataset:   {len(val_dataset)} sequences")

    # Create model
    print(f"\n{'='*70}")
    print(f"Creating model: {args.model_size}")
    print(f"{'='*70}")

    model, model_config = create_model(args.model_size, use_lora=args.use_lora)
    model = model.to(device)

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel parameters:")
    print(f"  Total:     {params['total']:>10,}")
    print(f"  Trainable: {params['trainable']:>10,}")
    if args.use_lora:
        print(f"  LoRA:      {params['lora']:>10,} ({params['lora']/params['total']*100:.2f}%)")

    # Freeze base model if using LoRA
    if args.use_lora and args.freeze_base:
        model.freeze_base_model()
        params_after = model.count_parameters()
        print(f"\n  After freezing:")
        print(f"  Trainable: {params_after['trainable']:>10,} ({params_after['trainable']/params['total']*100:.2f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Learning rate scheduler (cosine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.1
    )

    # Training loop
    print(f"\n{'='*70}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*70}")

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, perplexity = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print results
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  LR:         {current_lr:.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'perplexity': perplexity,
                'config': config_dict,
                'model_config': {
                    'vocab_size': model_config.vocab_size,
                    'n_layer': model_config.n_layer,
                    'n_head': model_config.n_head,
                    'n_embd': model_config.n_embd,
                },
            }

            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pt')

    print("\n" + "=" * 70)
    print("✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir / 'best_model.pt'}")
    print("=" * 70)

    print("\nNext steps:")
    print(f"  1. Generate samples:")
    print(f"     python generate.py --checkpoint {output_dir / 'best_model.pt'}")
    print(f"\n  2. Train with real MIDI data:")
    print(f"     python train_simple.py --data_dir ./midi_files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple training script for Brad Mehldau jazz piano"
    )

    # Model
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='Model size (default: small)')
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='Use LoRA adaptation (default: True)')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                       help='Freeze base model when using LoRA (default: True)')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                       help='Learning rate (default: 5e-4)')
    parser.add_argument('--seq_len', type=int, default=512,
                       help='Sequence length (default: 512)')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with MIDI files (default: None = use dummy data)')
    parser.add_argument('--num_train', type=int, default=1000,
                       help='Number of training sequences (default: 1000)')
    parser.add_argument('--num_val', type=int, default=100,
                       help='Number of validation sequences (default: 100)')

    # Misc
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory (default: ./outputs)')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    main(args)
