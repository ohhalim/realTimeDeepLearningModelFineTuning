# Phase 2: 빠른 실험 (2주, $10)

**목표**: 작은 모델로 첫 샘플 생성 (개념 검증)
**예상 시간**: 10-15시간 (2주)
**비용**: $10 (Colab Pro) 또는 무료 (Kaggle)

---

## ✅ 체크리스트

- [ ] 간단한 베이스라인 모델 구현
- [ ] Kaggle/Colab에서 학습 (2-3시간)
- [ ] 첫 10초 샘플 생성
- [ ] "들을 수 있는" 수준 확인
- [ ] GitHub에 업로드
- [ ] 샘플 MP3 5개 생성

---

## 🎯 목표

이 단계의 목적은 **빠르게 검증**하는 것입니다:
- ✅ 데이터가 충분한가?
- ✅ 학습이 제대로 되는가?
- ✅ 생성된 음악이 "들을 만한가"?

**만약 이 단계에서 괜찮은 결과가 나오지 않으면**, Phase 3로 넘어가기 전에 문제를 해결해야 합니다.

---

## 1. 모델 아키텍처 (Simple Baseline)

### 설계 원칙
- ❌ 복잡한 아키텍처 X
- ✅ 검증된 방법만 사용
- ✅ 빠른 학습 (2-3시간)
- ✅ 작은 모델 (<50M params)

### 아키텍처

```python
# models/simple_baseline.py
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class SimpleJazzModel(nn.Module):
    """Simple GPT-2 baseline for jazz piano generation

    Architecture:
    - GPT-2 Small (124M params, but we'll use smaller)
    - LoRA for efficient fine-tuning
    - Simple absolute position encoding
    """
    def __init__(self,
                 vocab_size=223,
                 n_embd=256,
                 n_layer=6,
                 n_head=8,
                 max_seq_len=512,
                 use_lora=True,
                 lora_r=8):
        super().__init__()

        # GPT-2 config (reduced size)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_seq_len,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=n_embd * 4,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )

        # Base model
        self.transformer = GPT2LMHeadModel(config)

        # LoRA (optional)
        if use_lora:
            self._add_lora(lora_r)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    def _add_lora(self, r=8):
        """Add LoRA to attention layers"""
        # TODO: Implement LoRA
        # For now, keep all parameters trainable
        pass

    def forward(self, input_ids, labels=None):
        """Forward pass"""
        outputs = self.transformer(input_ids=input_ids, labels=labels)
        return outputs

    @torch.no_grad()
    def generate(self,
                 prompt_ids,
                 max_new_tokens=128,
                 temperature=0.9,
                 top_p=0.95,
                 top_k=50):
        """Generate continuation"""
        self.eval()

        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Forward
            outputs = self.transformer(generated)
            logits = outputs.logits[:, -1, :]  # Last token

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if END token
            if next_token.item() == 222:  # END_TOKEN
                break

        return generated


if __name__ == "__main__":
    # Test
    model = SimpleJazzModel(vocab_size=223, n_embd=256, n_layer=6)

    # Dummy input
    batch_size = 4
    seq_len = 64
    input_ids = torch.randint(0, 223, (batch_size, seq_len))

    # Forward
    outputs = model(input_ids, labels=input_ids)
    print(f"Loss: {outputs.loss.item():.4f}")

    # Generate
    prompt = torch.randint(0, 223, (1, 16))
    generated = model.generate(prompt, max_new_tokens=32)
    print(f"Generated shape: {generated.shape}")
```

---

## 2. 데이터 로더

```python
# training/simple_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.tokenizer import SimpleMIDITokenizer

class SimpleMIDIDataset(Dataset):
    """Simple MIDI dataset for training"""
    def __init__(self,
                 data_dir,
                 tokenizer,
                 max_seq_len=512,
                 cache=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Find all MIDI files
        self.midi_files = glob.glob(f"{data_dir}/*.mid") + glob.glob(f"{data_dir}/*.midi")

        print(f"Found {len(self.midi_files)} MIDI files in {data_dir}")

        # Preprocess and cache
        if cache:
            print("Preprocessing and caching...")
            self.cached_data = []
            for i, midi_file in enumerate(self.midi_files):
                if i % 10 == 0:
                    print(f"  Processing {i}/{len(self.midi_files)}...")

                try:
                    tokens = tokenizer.encode(midi_file, max_length=max_seq_len)
                    self.cached_data.append(tokens)
                except Exception as e:
                    print(f"  Error processing {midi_file}: {e}")

            print(f"✅ Cached {len(self.cached_data)} sequences")
        else:
            self.cached_data = None

    def __len__(self):
        if self.cached_data:
            return len(self.cached_data)
        return len(self.midi_files)

    def __getitem__(self, idx):
        if self.cached_data:
            tokens = self.cached_data[idx]
        else:
            tokens = self.tokenizer.encode(self.midi_files[idx], max_length=self.max_seq_len)

        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),  # For language modeling
        }


def create_dataloaders(train_dir, val_dir, tokenizer, batch_size=8, max_seq_len=512):
    """Create train and validation dataloaders"""
    train_dataset = SimpleMIDIDataset(train_dir, tokenizer, max_seq_len)
    val_dataset = SimpleMIDIDataset(val_dir, tokenizer, max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for compatibility
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test
    tokenizer = SimpleMIDITokenizer()
    train_loader, val_loader = create_dataloaders(
        "data/splits/train",
        "data/splits/val",
        tokenizer,
        batch_size=4
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test batch
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
```

---

## 3. 학습 스크립트 (Kaggle/Colab용)

```python
# training/train_simple.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import json

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.simple_baseline import SimpleJazzModel
from preprocessing.tokenizer import SimpleMIDITokenizer
from training.simple_dataset import create_dataloaders


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs.loss

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def main():
    # Config
    config = {
        'vocab_size': 223,
        'n_embd': 256,
        'n_layer': 6,
        'n_head': 8,
        'max_seq_len': 512,
        'batch_size': 8,
        'learning_rate': 3e-4,
        'num_epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints/simple_baseline',
    }

    print("="*60)
    print("Simple Jazz Model Training")
    print("="*60)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("="*60)

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)

    # Save config
    with open(f"{config['save_dir']}/config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Tokenizer
    tokenizer = SimpleMIDITokenizer()

    # Data
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        "data/splits/train",
        "data/splits/val",
        tokenizer,
        batch_size=config['batch_size'],
        max_seq_len=config['max_seq_len']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("\nInitializing model...")
    model = SimpleJazzModel(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        max_seq_len=config['max_seq_len'],
    )
    model = model.to(config['device'])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6,
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config['device'], epoch+1)

        # Evaluate
        val_loss, val_perplexity = evaluate(model, val_loader, config['device'])

        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")
        print(f"  Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✅ New best model! (Val Loss: {val_loss:.4f})")

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
                'config': config,
            }, f"{config['save_dir']}/best_model.pt")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, f"{config['save_dir']}/checkpoint_epoch_{epoch+1}.pt")

        # Update learning rate
        scheduler.step()

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## 4. Kaggle/Colab에서 실행

### Kaggle Notebook

```python
# Cell 1: Setup
!git clone https://github.com/YOUR_USERNAME/Brad-Mehldau-AI.git
%cd Brad-Mehldau-AI

!pip install -q pretty_midi mido

# Cell 2: Upload data
# Kaggle: Add dataset manually
# Colab: Upload to Google Drive

# Cell 3: Train
!python training/train_simple.py

# Cell 4: Generate sample
!python evaluation/generate_sample.py --checkpoint checkpoints/simple_baseline/best_model.pt --output samples/phase2/sample_001.mid
```

### 예상 학습 시간
```
Kaggle P100 GPU: ~2-3 hours (20 epochs)
Colab T4 GPU: ~3-4 hours (20 epochs)
```

---

## 5. 샘플 생성

```python
# evaluation/generate_sample.py
import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.simple_baseline import SimpleJazzModel
from preprocessing.tokenizer import SimpleMIDITokenizer
import pretty_midi


def generate_sample(checkpoint_path, output_path, prompt_length=32, max_new_tokens=256):
    """Generate a MIDI sample from trained model"""
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    config = checkpoint['config']
    model = SimpleJazzModel(
        vocab_size=config['vocab_size'],
        n_embd=config['n_embd'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        max_seq_len=config['max_seq_len'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f})")

    # Create tokenizer
    tokenizer = SimpleMIDITokenizer()

    # Create prompt (START token + random notes)
    prompt = [tokenizer.START_TOKEN]
    prompt += [tokenizer.TIME_SHIFT_OFFSET + 5]  # 50ms
    prompt += [tokenizer.NOTE_ON_OFFSET + 40, tokenizer.VELOCITY_OFFSET + 20]  # C4, mf

    prompt_ids = torch.tensor([prompt])

    # Generate
    print(f"Generating {max_new_tokens} new tokens...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    prompt_ids = prompt_ids.to(device)

    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_p=0.95,
        )

    # Decode to notes
    generated_tokens = generated[0].cpu().tolist()
    notes = tokenizer.decode(generated_tokens)

    print(f"Generated {len(notes)} notes")

    # Convert to MIDI
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for pitch, onset, duration, velocity in notes:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=onset,
            end=onset + duration
        )
        piano.notes.append(note)

    midi.instruments.append(piano)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi.write(output_path)
    print(f"✅ Saved to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    generate_sample(args.checkpoint, args.output, max_new_tokens=args.max_new_tokens)
```

실행:
```bash
python evaluation/generate_sample.py \
    --checkpoint checkpoints/simple_baseline/best_model.pt \
    --output samples/phase2/sample_001.mid \
    --max_new_tokens 512
```

---

## 6. 품질 평가

생성된 MIDI를 들어보세요:
```bash
# MIDI → MP3 변환 (FluidSynth 사용)
fluidsynth -F samples/phase2/sample_001.mp3 \
    /usr/share/sounds/sf2/FluidR3_GM.sf2 \
    samples/phase2/sample_001.mid
```

### 평가 기준

✅ **최소 기준** (Phase 3로 진행):
- [ ] 화음이 어느 정도 coherent함
- [ ] 리듬이 완전히 random하지 않음
- [ ] "재즈 같은" 느낌이 조금이라도 남

❌ **재학습 필요** (다시 Phase 1/2):
- [ ] 완전히 random noise
- [ ] 화음이 전혀 없음
- [ ] 들을 수 없는 수준

---

## 🎯 Phase 2 완료 체크리스트

- [ ] Simple baseline 모델 구현
- [ ] 학습 스크립트 작성
- [ ] Kaggle/Colab에서 학습 완료 (2-3시간)
- [ ] Best model 저장 완료
- [ ] 샘플 5개 생성
- [ ] 품질 확인 (최소 기준 통과)
- [ ] GitHub에 커밋 & 푸시
- [ ] README 업데이트

---

## 🎉 Phase 2 완료!

첫 AI 재즈 샘플 생성 완료! 🎵

**다음 단계**: `../phase3_training/README.md`

본격적인 JazzFormer-CR 학습을 시작하세요! 🚀
