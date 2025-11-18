"""
Training Loop for Multi-Task JazzFormer

Implements ImprovNet-style corruption-refinement training strategy with:
- Multi-task learning (5 tasks)
- Random corruption augmentation
- Genre conditioning
- Mixed precision training (optional)
- Checkpointing and logging
- Reproducibility

Usage:
    from realjazz.training import TrainingConfig, train_multitask_model

    config = TrainingConfig(
        batch_size=32,
        learning_rate=1e-4,
        n_epochs=100,
        corruption_prob=0.5,
        seed=42
    )

    model = MultiTaskJazzFormer(...)
    train_multitask_model(model, train_dataloader, val_dataloader, config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
from tqdm import tqdm
import json

from .multitask_model import MultiTaskJazzFormer, TaskType, GenreType
from .corruption import CorruptionType, apply_random_corruption
from .reproducibility import set_global_seed, ReproducibleContext
import random


@dataclass
class TrainingConfig:
    """Configuration for training"""

    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    n_epochs: int = 100
    warmup_steps: int = 1000
    grad_clip_norm: float = 1.0

    # Multi-task learning
    task_weights: Dict[int, float] = field(default_factory=lambda: {
        TaskType.JAMMING: 1.0,
        TaskType.CONTINUATION: 1.0,
        TaskType.INFILLING: 1.0,
        TaskType.HARMONIZATION: 0.5,  # Lower weight (harder task)
        TaskType.CROSS_GENRE: 0.5,    # Lower weight (harder task)
    })

    # Corruption-refinement
    corruption_prob: float = 0.5  # Probability of applying corruption
    corruption_types: List[int] = field(default_factory=lambda: list(range(9)))  # All 9 types

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 1000
    keep_n_checkpoints: int = 5

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000

    # Mixed precision (requires CUDA with tensor cores)
    use_amp: bool = False

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Validate configuration"""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.corruption_prob <= 1, "corruption_prob must be in [0, 1]"
        assert all(0 <= t < 9 for t in self.corruption_types), "Invalid corruption types"

        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


class MultiTaskTrainer:
    """
    Trainer for Multi-Task JazzFormer with corruption-refinement
    """

    def __init__(
        self,
        model: MultiTaskJazzFormer,
        config: TrainingConfig
    ):
        """
        Initialize trainer

        Args:
            model: Multi-task model to train
            config: Training configuration
        """
        self.model = model.to(config.device)
        self.config = config

        # Set seed for reproducibility
        set_global_seed(config.seed, config.deterministic)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler (cosine annealing with warmup)
        self.scheduler = None  # Set in train() after knowing total steps

        # Mixed precision scaler (for AMP)
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Metrics
        self.train_losses = []
        self.val_losses = []

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step

        Args:
            batch: Dictionary with keys 'tokens', 'event_types', 'genre_id'

        Returns:
            Dictionary of losses
        """
        self.model.train()
        device = self.config.device

        # Extract batch
        tokens = batch['tokens'].to(device)  # (B, L)
        event_types = batch['event_types'].to(device)  # (B, L)
        genre_id = batch.get('genre_id', torch.zeros(tokens.size(0), dtype=torch.long)).to(device)  # (B,)

        batch_size = tokens.size(0)

        # Multi-task training: random task selection per sample
        task_ids = torch.randint(0, 5, (batch_size,))  # Random task for each sample
        total_loss = 0.0
        task_losses = {i: 0.0 for i in range(5)}

        # Process each sample (could be optimized with batched processing)
        for i in range(batch_size):
            sample_tokens = tokens[i:i+1]  # (1, L)
            sample_event_types = event_types[i:i+1]
            sample_genre = genre_id[i].item() if genre_id.dim() > 0 else genre_id.item()
            task_id = task_ids[i].item()

            # Apply corruption with probability
            if random.random() < self.config.corruption_prob:
                # Convert to list for corruption
                token_list = sample_tokens[0].cpu().tolist()

                # Apply random corruption
                corruption_type = random.choice(self.config.corruption_types)
                corrupted_list, corr_id = apply_random_corruption(
                    token_list,
                    corruption_type=CorruptionType(corruption_type)
                )

                # Convert back to tensor
                corrupted_tokens = torch.tensor([corrupted_list], device=device)
            else:
                corrupted_tokens = sample_tokens
                corr_id = None

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                logits, _ = self.model.forward_multitask(
                    corrupted_tokens[:, :-1],  # Input (excluding last token)
                    sample_event_types[:, :-1],
                    task_id=task_id,
                    genre_id=sample_genre,
                    corruption_id=corr_id
                )

                # Compute loss (predict original tokens)
                targets = sample_tokens[:, 1:]  # Shifted targets
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100  # Padding token if any
                )

                # Weight by task importance
                weighted_loss = loss * self.config.task_weights[task_id]

            total_loss += weighted_loss
            task_losses[task_id] += loss.item()

        # Average loss
        total_loss = total_loss / batch_size

        # Backward pass
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Return metrics
        return {
            'loss': total_loss.item(),
            **{f'task_{i}_loss': task_losses[i] for i in range(5)}
        }

    @torch.no_grad()
    def eval_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single evaluation step

        Args:
            batch: Dictionary with keys 'tokens', 'event_types', 'genre_id'

        Returns:
            Dictionary of losses
        """
        self.model.eval()
        device = self.config.device

        tokens = batch['tokens'].to(device)
        event_types = batch['event_types'].to(device)
        genre_id = batch.get('genre_id', torch.zeros(tokens.size(0), dtype=torch.long)).to(device)

        batch_size = tokens.size(0)
        total_loss = 0.0

        # Evaluate on all tasks
        for task_id in range(5):
            for i in range(batch_size):
                sample_tokens = tokens[i:i+1]
                sample_event_types = event_types[i:i+1]
                sample_genre = genre_id[i].item() if genre_id.dim() > 0 else genre_id.item()

                logits, _ = self.model.forward_multitask(
                    sample_tokens[:, :-1],
                    sample_event_types[:, :-1],
                    task_id=task_id,
                    genre_id=sample_genre
                )

                targets = sample_tokens[:, 1:]
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100
                )

                total_loss += loss.item()

        avg_loss = total_loss / (batch_size * 5)  # Average over all tasks

        return {'val_loss': avg_loss}

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        n_epochs: Optional[int] = None
    ):
        """
        Full training loop

        Args:
            train_dataloader: Training data
            val_dataloader: Validation data (optional)
            n_epochs: Number of epochs (overrides config if provided)
        """
        n_epochs = n_epochs or self.config.n_epochs
        total_steps = len(train_dataloader) * n_epochs

        # Initialize learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_steps / total_steps
        )

        print("=" * 70)
        print("Starting Multi-Task Training")
        print("=" * 70)
        print(f"  Device: {self.config.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Total steps: {total_steps}")
        print(f"  Corruption prob: {self.config.corruption_prob}")
        print(f"  Seed: {self.config.seed}")
        print("=" * 70)

        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()

            # Training
            train_loss = 0.0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                metrics = self.train_step(batch)
                train_loss += metrics['loss']

                self.global_step += 1

                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    avg_loss = train_loss / self.config.log_every_n_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                    train_loss = 0.0

                # Validation
                if val_dataloader and self.global_step % self.config.eval_every_n_steps == 0:
                    val_metrics = self.evaluate(val_dataloader)
                    val_loss = val_metrics['val_loss']

                    print(f"\n  [Step {self.global_step}] Val loss: {val_loss:.4f}")

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(is_best=True)

                # Checkpointing
                if self.global_step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint()

            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s")

        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print("=" * 70)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation set

        Args:
            dataloader: Validation data

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc="Evaluating"):
            metrics = self.eval_step(batch)
            total_loss += metrics['val_loss']
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0

        return {'val_loss': avg_loss}

    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_path = Path(self.config.checkpoint_dir)

        # Checkpoint data
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }

        # Save regular checkpoint
        filename = f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path / filename)

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_path / "best_model.pt")
            print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")

        # Clean up old checkpoints (keep only N most recent)
        checkpoints = sorted(checkpoint_path.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > self.config.keep_n_checkpoints:
            for old_ckpt in checkpoints[:-self.config.keep_n_checkpoints]:
                old_ckpt.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"✓ Loaded checkpoint from step {self.global_step}")


def train_multitask_model(
    model: MultiTaskJazzFormer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None
) -> MultiTaskJazzFormer:
    """
    Convenience function for training

    Args:
        model: Model to train
        train_dataloader: Training data
        val_dataloader: Validation data
        config: Training configuration (uses defaults if None)

    Returns:
        Trained model
    """
    config = config or TrainingConfig()
    trainer = MultiTaskTrainer(model, config)
    trainer.train(train_dataloader, val_dataloader)

    return model


if __name__ == "__main__":
    print("=" * 70)
    print("Training Module Test")
    print("=" * 70)

    print("\n⚠️  NOTE: This is a structure test only")
    print("  Real training requires actual dataset (MIDI files)")
    print("  See scripts/train.py for full training pipeline")

    # Create dummy model
    from .multitask_model import MultiTaskJazzFormer

    print("\n[1] Creating model...")
    model = MultiTaskJazzFormer(
        vocab_size=128,
        d_model=256,
        n_heads=4,
        n_layers=2  # Small for testing
    )
    print(f"  ✓ Model created ({sum(p.numel() for p in model.parameters()):,} params)")

    # Create dummy dataset
    print("\n[2] Creating dummy dataset...")

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'tokens': torch.randint(0, 128, (32,)),
                'event_types': torch.randint(0, 3, (32,)),
                'genre_id': torch.tensor(random.randint(0, 1))
            }

    train_dataset = DummyDataset()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    print(f"  ✓ Dataset created ({len(train_dataset)} samples)")

    # Create training config
    print("\n[3] Creating training config...")
    config = TrainingConfig(
        batch_size=8,
        n_epochs=2,
        log_every_n_steps=10,
        eval_every_n_steps=50,
        save_every_n_steps=50,
        seed=42
    )
    print("  ✓ Config created")

    # Test training (just 2 epochs on dummy data)
    print("\n[4] Testing training loop (2 epochs)...")
    try:
        train_multitask_model(model, train_dataloader, config=config)
        print("\n✅ Training loop test passed!")
    except Exception as e:
        print(f"\n❌ Training loop test failed: {e}")
        raise

    print("\n" + "=" * 70)
