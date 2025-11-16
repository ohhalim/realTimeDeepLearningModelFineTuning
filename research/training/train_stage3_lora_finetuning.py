"""
Stage 3: Hierarchical LoRA Fine-tuning for Brad Mehldau Style

Training objective: Fine-tune only LoRA parameters on Brad Mehldau MIDI data
with multi-task objectives (next token + harmony + voicing + rhythm).

Expected results:
- 99.6% parameter reduction vs full fine-tuning
- 15-20% perplexity improvement
- >90% style classification accuracy
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

# Add research directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))

from music_transformer import MusicTransformerWithLoRA
from style_encoder import StyleContrastiveModel


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for hierarchical LoRA fine-tuning.

    L_total = L_next_token + λ_H * L_harmony + λ_V * L_voicing + λ_R * L_rhythm

    Components:
    - Next token prediction (standard cross-entropy)
    - Harmony loss (chord label prediction)
    - Voicing loss (pitch class distribution matching)
    - Rhythm loss (inter-onset interval distribution)
    """

    def __init__(
        self,
        lambda_harmony: float = 0.1,
        lambda_voicing: float = 0.1,
        lambda_rhythm: float = 0.1,
        vocab_size: int = 512,
    ):
        super().__init__()

        self.lambda_harmony = lambda_harmony
        self.lambda_voicing = lambda_voicing
        self.lambda_rhythm = lambda_rhythm
        self.vocab_size = vocab_size

        # Cross-entropy for next token prediction
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def extract_harmony_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract chord-related tokens from MIDI sequence.

        Simplified: Just use note_on tokens (0-127)
        In practice, use a proper chord recognizer.
        """
        # Get note tokens (0-127)
        note_mask = tokens < 128
        notes = tokens[note_mask]

        # Group into pitch classes (mod 12)
        pitch_classes = notes % 12

        # Return one-hot encoding of pitch classes
        if len(pitch_classes) == 0:
            return torch.zeros(12, device=tokens.device)

        pitch_class_hist = torch.zeros(12, device=tokens.device)
        for pc in pitch_classes:
            pitch_class_hist[pc] += 1

        return pitch_class_hist / (pitch_class_hist.sum() + 1e-8)

    def harmony_loss(
        self,
        generated_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Harmony loss: Match pitch class distribution.

        Args:
            generated_tokens: [batch, seq_len]
            target_tokens: [batch, seq_len]
        """
        batch_size = generated_tokens.shape[0]
        loss = 0.0

        for i in range(batch_size):
            gen_harmony = self.extract_harmony_features(generated_tokens[i])
            tgt_harmony = self.extract_harmony_features(target_tokens[i])

            # KL divergence
            loss += F.kl_div(
                gen_harmony.log(),
                tgt_harmony,
                reduction='sum',
            )

        return loss / batch_size

    def voicing_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Voicing loss: Match note distribution within predicted chords.

        Simplified version: Use cross-entropy on note tokens only.
        """
        # Get logits for note tokens (0-127)
        note_logits = logits[:, :, :128]  # [batch, seq_len, 128]
        note_targets = targets.clone()

        # Mask non-note tokens
        note_targets[note_targets >= 128] = -100

        # Cross-entropy
        loss = F.cross_entropy(
            note_logits.reshape(-1, 128),
            note_targets.reshape(-1),
            ignore_index=-100,
        )

        return loss

    def rhythm_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Rhythm loss: Match time shift distribution.

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
        """
        # Get logits for time shift tokens (256-383)
        time_shift_logits = logits[:, :, 256:384]  # [batch, seq_len, 128]
        time_shift_targets = targets.clone()

        # Mask non-time-shift tokens
        time_shift_mask = (targets >= 256) & (targets < 384)
        time_shift_targets[~time_shift_mask] = -100
        time_shift_targets[time_shift_mask] -= 256  # Re-index to 0-127

        # Cross-entropy
        loss = F.cross_entropy(
            time_shift_logits.reshape(-1, 128),
            time_shift_targets.reshape(-1),
            ignore_index=-100,
        )

        return loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]

        Returns:
            dict with keys: total_loss, next_token_loss, harmony_loss, voicing_loss, rhythm_loss
        """
        # Next token prediction loss
        next_token_loss = self.ce_loss(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

        # Get predicted tokens for auxiliary losses
        predicted_tokens = torch.argmax(logits, dim=-1)  # [batch, seq_len]

        # Harmony loss
        h_loss = self.harmony_loss(predicted_tokens, targets)

        # Voicing loss
        v_loss = self.voicing_loss(logits, targets)

        # Rhythm loss
        r_loss = self.rhythm_loss(logits, targets)

        # Total loss
        total_loss = (
            next_token_loss
            + self.lambda_harmony * h_loss
            + self.lambda_voicing * v_loss
            + self.lambda_rhythm * r_loss
        )

        return {
            'total_loss': total_loss,
            'next_token_loss': next_token_loss,
            'harmony_loss': h_loss,
            'voicing_loss': v_loss,
            'rhythm_loss': r_loss,
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> int:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_next_token_loss = 0.0
    total_harmony_loss = 0.0
    total_voicing_loss = 0.0
    total_rhythm_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)  # [batch, seq_len]
        labels = batch['labels'].to(device)  # [batch, seq_len]

        # Forward pass
        logits = model(input_ids)  # [batch, seq_len, vocab_size]

        # Compute multi-task loss
        loss_dict = criterion(logits, labels)

        total_loss_batch = loss_dict['total_loss']
        next_token_loss = loss_dict['next_token_loss']
        harmony_loss = loss_dict['harmony_loss']
        voicing_loss = loss_dict['voicing_loss']
        rhythm_loss = loss_dict['rhythm_loss']

        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_next_token_loss += next_token_loss.item()
        total_harmony_loss += harmony_loss.item()
        total_voicing_loss += voicing_loss.item()
        total_rhythm_loss += rhythm_loss.item()

        # Log to tensorboard
        if batch_idx % 10 == 0:
            writer.add_scalar('Train/TotalLoss', total_loss_batch.item(), global_step)
            writer.add_scalar('Train/NextTokenLoss', next_token_loss.item(), global_step)
            writer.add_scalar('Train/HarmonyLoss', harmony_loss.item(), global_step)
            writer.add_scalar('Train/VoicingLoss', voicing_loss.item(), global_step)
            writer.add_scalar('Train/RhythmLoss', rhythm_loss.item(), global_step)

        global_step += 1

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss_batch.item():.4f}",
            'next_token': f"{next_token_loss.item():.4f}",
            'harmony': f"{harmony_loss.item():.4f}",
        })

    # Compute averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_next_token = total_next_token_loss / num_batches
    avg_harmony = total_harmony_loss / num_batches
    avg_voicing = total_voicing_loss / num_batches
    avg_rhythm = total_rhythm_loss / num_batches

    print(f"\nEpoch {epoch} Summary:")
    print(f"  Total Loss: {avg_loss:.4f}")
    print(f"  Next Token Loss: {avg_next_token:.4f}")
    print(f"  Harmony Loss: {avg_harmony:.4f}")
    print(f"  Voicing Loss: {avg_voicing:.4f}")
    print(f"  Rhythm Loss: {avg_rhythm:.4f}")

    # Log epoch averages
    writer.add_scalar('Epoch/TotalLoss', avg_loss, epoch)
    writer.add_scalar('Epoch/NextTokenLoss', avg_next_token, epoch)
    writer.add_scalar('Epoch/HarmonyLoss', avg_harmony, epoch)
    writer.add_scalar('Epoch/VoicingLoss', avg_voicing, epoch)
    writer.add_scalar('Epoch/RhythmLoss', avg_rhythm, epoch)

    return global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    total_next_token_loss = 0.0

    for batch in tqdm(dataloader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids)

        loss_dict = criterion(logits, labels)

        total_loss += loss_dict['total_loss'].item()
        total_next_token_loss += loss_dict['next_token_loss'].item()

    # Compute averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_next_token = total_next_token_loss / num_batches

    # Compute perplexity
    perplexity = torch.exp(torch.tensor(avg_next_token)).item()

    print(f"\nValidation Results:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

    # Log to tensorboard
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Perplexity', perplexity, epoch)

    return avg_loss


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"stage3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')

    # Create model
    print("\nCreating Music Transformer with Hierarchical LoRA...")
    model = MusicTransformerWithLoRA(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_lora=True,
        lora_r=args.lora_r,
    ).to(device)

    # Print model summary
    print("\n" + model.get_model_summary())

    # Freeze base model
    if args.freeze_base:
        model.freeze_base_model()

        params = model.count_parameters()
        print(f"\nTrainable parameters: {params['trainable']:,} ({params['trainable']/params['total']*100:.2f}%)")

    # Create dummy dataloader (TODO: Replace with real MIDI dataset)
    print("\nCreating dummy dataloader (replace with real MIDI data)...")

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000, seq_len=512):
            self.num_samples = num_samples
            self.seq_len = seq_len

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Random MIDI tokens
            input_ids = torch.randint(0, 512, (self.seq_len,))
            labels = torch.randint(0, 512, (self.seq_len,))
            return {'input_ids': input_ids, 'labels': labels}

    train_dataset = DummyDataset(num_samples=args.num_train_samples)
    val_dataset = DummyDataset(num_samples=args.num_val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create multi-task loss
    criterion = MultiTaskLoss(
        lambda_harmony=args.lambda_harmony,
        lambda_voicing=args.lambda_voicing,
        lambda_rhythm=args.lambda_rhythm,
        vocab_size=args.vocab_size,
    )

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1,
    )

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*60}")

        # Train
        global_step = train_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, writer, global_step
        )

        # Validate
        if epoch % args.val_every == 0:
            val_loss = validate(
                model, val_loader, criterion,
                device, epoch, writer
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"\n✓ Best model saved to {checkpoint_path}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LearningRate', current_lr, epoch)

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"\nCheckpoint saved to {checkpoint_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print("="*60)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Hierarchical LoRA Fine-tuning")

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--lora_r', type=int, default=16)

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--freeze_base', action='store_true', default=True)

    # Multi-task loss weights
    parser.add_argument('--lambda_harmony', type=float, default=0.1)
    parser.add_argument('--lambda_voicing', type=float, default=0.1)
    parser.add_argument('--lambda_rhythm', type=float, default=0.1)

    # Data parameters (dummy data for now)
    parser.add_argument('--num_train_samples', type=int, default=1000)
    parser.add_argument('--num_val_samples', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    # Logging
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--val_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=10)

    args = parser.parse_args()

    main(args)
