#!/usr/bin/env python3
"""
PyTorch Music Transformer 학습 스크립트

순수 PyTorch로 Music Transformer를 처음부터 학습

사용법:
    python scripts/train_pytorch_transformer.py --config configs/pytorch_transformer_config.yaml
"""

import argparse
import os
import sys
import random
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.music_transformer import MusicTransformer, count_parameters
from models.midi_dataset import MIDIDataset, create_causal_mask


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Create causal mask
        seq_len = input_ids.size(1)
        mask = create_causal_mask(seq_len).to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, mask)

        # Calculate loss
        loss = criterion(
            logits.view(-1, model.vocab_size),
            labels.view(-1)
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            seq_len = input_ids.size(1)
            mask = create_causal_mask(seq_len).to(device)

            logits = model(input_ids, mask)

            loss = criterion(
                logits.view(-1, model.vocab_size),
                labels.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.midi_dir:
        config["data"]["midi_dir"] = args.midi_dir
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.use_role_dataset:
        config["data"]["use_role_dataset"] = True
    if args.role:
        config["data"]["role"] = args.role
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    return config


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Music Transformer 학습"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pytorch_transformer_config.yaml",
        help="설정 파일 경로"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="체크포인트에서 재개"
    )
    parser.add_argument(
        "--use_role_dataset",
        action="store_true",
        help="role-conditioned 데이터셋 모드 사용(data/roles/*)"
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        choices=["lead", "accompaniment", "call_response"],
        help="학습할 role 이름 (출력 디렉토리 suffix에도 사용)"
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        default=None,
        help="config.data.midi_dir override"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="config.training.output_dir override"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="config.training.epochs override"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = apply_overrides(config, args)

    seed = config.get("seed", 42)
    set_seed(seed)

    print("=" * 60)
    print("PyTorch Music Transformer 학습")
    print("=" * 60)
    print(f"설정 파일: {args.config}")
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Dataset
    print("=" * 60)
    print("데이터셋 로딩...")
    print("=" * 60)

    use_role_dataset = config["data"].get("use_role_dataset", False)
    dataset = MIDIDataset(
        midi_dir=config['data']['midi_dir'],
        max_length=config['data']['max_length'],
        augment=config['data'].get('augment', True),
        use_role_dataset=use_role_dataset,
        max_conditioning_tokens=config["data"].get("max_conditioning_tokens", 256),
    )

    print(f"Dataset mode: {'role-conditioned' if use_role_dataset else 'raw'}")
    if config["data"].get("role"):
        print(f"Role: {config['data']['role']}")
    print(f"Seed: {seed}")

    # Train/Val split
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least 2 samples for train/val split.")

    train_size = int(len(dataset) * config['data']['train_split'])
    train_size = min(max(train_size, 1), len(dataset) - 1)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    print("=" * 60)
    print("모델 초기화...")
    print("=" * 60)

    model = MusicTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['data']['max_length'],
        dropout=config['model']['dropout']
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")
    print()

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tokenizer.pad_token)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))
        print(f"Resumed from epoch {start_epoch}")
        print()

    # Training loop
    print("=" * 60)
    print("학습 시작")
    print("=" * 60)

    output_dir = config['training']['output_dir']
    if use_role_dataset and config["data"].get("role"):
        output_dir = os.path.join(output_dir, config["data"]["role"])
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1
        )

        # Validate
        val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Perplexity: {val_perplexity:.2f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                output_dir,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config
            }, checkpoint_path)

            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = os.path.join(output_dir, "best_model.pt")

            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config
            }, best_path)

            print(f"✓ New best model! Val Loss: {val_loss:.4f}")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"모델 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
