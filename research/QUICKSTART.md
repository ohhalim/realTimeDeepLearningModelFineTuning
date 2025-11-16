# Quick Start Guide

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: RTX 3090/4090 or A100)
- 16GB+ RAM
- 24GB+ GPU VRAM (for full model)

## Installation

### Step 1: Install PyTorch

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (not recommended for training)
pip install torch torchvision torchaudio
```

### Step 2: Install Dependencies

```bash
cd research
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python test_installation.py
```

Expected output:
```
======================================================================
Testing Hierarchical StyleLoRA-Transformer Implementation
======================================================================

[1/6] Testing dependency imports...
✓ PyTorch 2.x.x imported successfully
✓ NumPy and tqdm imported successfully

[2/6] Testing Hierarchical LoRA module...
✓ Hierarchical LoRA modules imported successfully

[3/6] Testing Music Transformer module...
✓ Music Transformer imported successfully

[4/6] Testing Style Encoder module...
✓ Style Encoder imported successfully

[5/6] Testing model instantiation...
✓ Model instantiated successfully

[6/6] Testing forward pass...
✓ Forward pass successful

======================================================================
✓ ALL TESTS PASSED!
======================================================================
```

## Quick Test (No Training)

Test the model without training:

```bash
cd research

# Test model instantiation
python -c "
from models.music_transformer import MusicTransformerWithLoRA
model = MusicTransformerWithLoRA(vocab_size=512, d_model=512, num_layers=6, use_lora=True)
print(f'Model created: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

## Training (Minimal Example)

Start training with dummy data (for testing):

```bash
cd research

python training/train_stage3_lora_finetuning.py \
    --batch_size 2 \
    --num_epochs 5 \
    --num_train_samples 100 \
    --num_val_samples 20 \
    --output_dir ./test_outputs
```

Expected time: ~2-3 minutes on GPU, ~10 minutes on CPU

## Training (Full Brad Mehldau Model)

### Step 1: Prepare MIDI Data

See `../MIDI_DATA_SOURCES.md` for data collection guide.

Place MIDI files in:
```
data/
  brad_mehldau/
    train/  (70 files)
    val/    (10 files)
    test/   (7 files)
```

### Step 2: Start Training

```bash
python training/train_stage3_lora_finetuning.py \
    --batch_size 4 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --output_dir ./outputs/full_training
```

Expected time: ~12 hours on RTX 4090

## Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/*/logs

# Check logs
tail -f outputs/*/training.log
```

## Generate Music

After training:

```bash
python evaluation/generate_samples.py \
    --checkpoint outputs/full_training/best_model.pt \
    --num_samples 10 \
    --output_dir ./generated \
    --harmony_weight 1.0 \
    --voicing_weight 0.8 \
    --rhythm_weight 0.5
```

## Interpretable Control Examples

### Example 1: Only Harmony Adaptation

```python
from models.music_transformer import MusicTransformerWithLoRA

model = MusicTransformerWithLoRA.from_pretrained("checkpoint.pt")
model.lora_controller.set_weights(
    harmony=1.0,   # Full Mehldau harmony
    voicing=0.0,   # No voicing adaptation
    rhythm=0.0,    # No rhythm adaptation
    dynamics=0.0   # No dynamics adaptation
)
```

### Example 2: Balanced Style

```python
model.lora_controller.set_weights(
    harmony=0.8,
    voicing=0.8,
    rhythm=0.5,
    dynamics=1.0
)
```

### Example 3: Extreme Mehldau Style

```python
model.lora_controller.set_weights(
    harmony=1.0,
    voicing=1.0,
    rhythm=1.0,
    dynamics=1.0
)
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or model size:
```bash
python training/train_stage3_lora_finetuning.py \
    --batch_size 1 \
    --d_model 512 \
    --num_layers 8
```

### Import Errors

Make sure you're in the research directory:
```bash
cd /path/to/research
python test_installation.py
```

### Slow Training

Enable mixed precision:
```python
# In training script, add:
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    logits = model(input_ids)
```

## Common Issues

**Q: "ModuleNotFoundError: No module named 'torch'"**
A: Install PyTorch: `pip install torch`

**Q: "CUDA out of memory"**
A: Reduce batch size or use CPU: `--batch_size 1` or set CUDA_VISIBLE_DEVICES=""

**Q: "No MIDI data found"**
A: Use dummy data for testing: `--num_train_samples 100`

## Performance Benchmarks

| Hardware | Batch Size | Time/Epoch | GPU Memory |
|----------|------------|------------|------------|
| RTX 4090 | 4 | 7 min | 18GB |
| RTX 3090 | 4 | 10 min | 20GB |
| A100 40GB | 8 | 5 min | 32GB |
| CPU | 1 | 2 hours | 16GB RAM |

## Next Steps

1. ✅ Verify installation with `test_installation.py`
2. 📊 Collect Brad Mehldau MIDI data
3. 🏋️ Train the model
4. 🎵 Generate samples
5. 📝 Write your paper!

For more details, see:
- `README.md` - Full documentation
- `../RESEARCH_ARCHITECTURE.md` - Architecture design
- `../MIDI_DATA_SOURCES.md` - Data collection guide
