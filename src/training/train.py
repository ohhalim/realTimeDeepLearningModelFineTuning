"""
Training script for JazzFormer-RT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.jazzformer_rt import create_jazzformer_rt
from src.data.dataset import create_dataloaders
from src.evaluation.metrics import compute_metrics


class Trainer:
    """
    Trainer for JazzFormer-RT model
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['pretrain']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['pretrain']['max_steps']
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Wandb
        if config['experiment']['wandb']['enabled']:
            wandb.init(
                project=config['experiment']['wandb']['project'],
                config=config
            )

    def train_epoch(self) -> float:
        """
        Train for one epoch

        Returns:
            average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (tokens, targets, artist_ids) in enumerate(pbar):
            # Move to device
            tokens = tokens.to(self.device)
            targets = targets.to(self.device)
            artist_ids = artist_ids.to(self.device)

            # Forward pass
            logits = self.model(tokens, artist_ids)

            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['pretrain']['gradient_clip']
            )

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'lr': self.optimizer.param_groups[0]['lr']})

            # Log to wandb
            if self.config['experiment']['wandb']['enabled']:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model

        Returns:
            validation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for tokens, targets, artist_ids in pbar:
            # Move to device
            tokens = tokens.to(self.device)
            targets = targets.to(self.device)
            artist_ids = artist_ids.to(self.device)

            # Forward pass
            logits = self.model(tokens, artist_ids)

            # Compute loss
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'val_loss': loss.item()})

        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {
            'val/loss': avg_loss,
            'val/perplexity': perplexity
        }

        return metrics

    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            path: checkpoint save path
            is_best: whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }

        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

    def train(self, num_epochs: int, checkpoint_dir: str):
        """
        Main training loop

        Args:
            num_epochs: number of epochs to train
            checkpoint_dir: directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_metrics = self.validate()
            print(f"Val Loss: {val_metrics['val/loss']:.4f}")
            print(f"Val Perplexity: {val_metrics['val/perplexity']:.2f}")

            # Log to wandb
            if self.config['experiment']['wandb']['enabled']:
                wandb.log({
                    'epoch': epoch + 1,
                    **val_metrics
                })

            # Save checkpoint
            is_best = val_metrics['val/loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val/loss']

            checkpoint_path = os.path.join(
                checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            self.save_checkpoint(checkpoint_path, is_best)

            print(f"Checkpoint saved: {checkpoint_path}")
            if is_best:
                print("✓ New best model!")


def main():
    """
    Main training function
    """
    # Load configuration
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    print("Creating JazzFormer-RT model...")
    model = create_jazzformer_rt(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dir='data/brad_mehldau/train',
        val_dir='data/brad_mehldau/val',
        batch_size=config['training']['pretrain']['batch_size'],
        max_length=config['io']['max_sequence_length'],
        artist_name='brad_mehldau',
        num_workers=config['hardware']['num_workers']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    num_epochs = config['training']['pretrain']['max_steps'] // len(train_loader)
    print(f"\nTraining for {num_epochs} epochs...")

    trainer.train(
        num_epochs=num_epochs,
        checkpoint_dir='models/finetuned/brad_mehldau'
    )

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
