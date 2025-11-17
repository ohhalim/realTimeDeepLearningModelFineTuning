#!/usr/bin/env python3
"""
Simple training script for JazzFormer-RT

Usage:
    python scripts/train.py --data_dir data/brad_mehldau
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.jazzformer_rt import JazzFormerRT
from src.data.dataset import create_dataloaders


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for tokens, targets, artist_ids in pbar:
        tokens = tokens.to(device)
        targets = targets.to(device)
        artist_ids = artist_ids.to(device)

        # Forward
        logits = model(tokens, artist_ids)

        # Loss
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    for tokens, targets, artist_ids in tqdm(val_loader, desc="Validation"):
        tokens = tokens.to(device)
        targets = targets.to(device)
        artist_ids = artist_ids.to(device)

        logits = model(tokens, artist_ids)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train JazzFormer-RT")
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--output_dir', type=str, default='models/finetuned/brad_mehldau')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = None
        print(f"Warning: Config file not found at {args.config}, using defaults")

    # Create model
    print("Creating JazzFormer-RT model...")
    if config and 'architecture' in config:
        arch = config['architecture']
        model = JazzFormerRT(
            vocab_size=config['io']['vocab_size'],
            d_model=arch['d_model'],
            n_heads=arch['n_heads'],
            n_layers=arch['n_layers'],
            d_ff=arch['d_ff'],
            max_seq_len=config['io']['max_sequence_length'],
            dropout=arch['dropout'],
            num_artists=arch['style_embedding']['num_artists'],
            style_embedding_dim=arch['style_embedding']['embedding_dim'],
            window_size=arch['streaming']['window_size']
        )
    else:
        model = JazzFormerRT()

    model = model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dataloaders
    print(f"Loading data from {args.data_dir}...")
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        print(f"Please create the directory and add MIDI files")
        return

    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory not found: {val_dir}")
        print(f"Using training data for validation")
        val_dir = train_dir

    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        max_length=2048,
        artist_name='brad_mehldau',
        num_workers=2
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch)
        val_loss = validate(model, val_loader, criterion, args.device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }

        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(args.output_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ New best model! Saved to {best_path}")

    print(f"\n{'='*60}")
    print("✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
