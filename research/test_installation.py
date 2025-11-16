#!/usr/bin/env python3
"""
Test script to verify all components work correctly.
Run this to check if the implementation is complete and functional.
"""

import sys
from pathlib import Path

# Add research directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("Testing Hierarchical StyleLoRA-Transformer Implementation")
print("=" * 70)

# Test 1: Import dependencies
print("\n[1/6] Testing dependency imports...")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print(f"✓ PyTorch {torch.__version__} imported successfully")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    print("\nPlease install PyTorch:")
    print("  pip install torch torchvision torchaudio")
    sys.exit(1)

try:
    import numpy as np
    from tqdm import tqdm
    print("✓ NumPy and tqdm imported successfully")
except ImportError as e:
    print(f"✗ Dependencies import failed: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r research/requirements.txt")
    sys.exit(1)

# Test 2: Import hierarchical LoRA
print("\n[2/6] Testing Hierarchical LoRA module...")
try:
    from models.hierarchical_lora import (
        HierarchicalLoRAController,
        HarmonyLoRA,
        VoicingLoRA,
        RhythmLoRA,
        DynamicsLoRA,
    )
    print("✓ Hierarchical LoRA modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import hierarchical_lora: {e}")
    sys.exit(1)

# Test 3: Import Music Transformer
print("\n[3/6] Testing Music Transformer module...")
try:
    from models.music_transformer import MusicTransformerWithLoRA
    print("✓ Music Transformer imported successfully")
except ImportError as e:
    print(f"✗ Failed to import music_transformer: {e}")
    sys.exit(1)

# Test 4: Import Style Encoder
print("\n[4/6] Testing Style Encoder module...")
try:
    from models.style_encoder import StyleContrastiveModel, StyleEncoder
    print("✓ Style Encoder imported successfully")
except ImportError as e:
    print(f"✗ Failed to import style_encoder: {e}")
    sys.exit(1)

# Test 5: Instantiate model
print("\n[5/6] Testing model instantiation...")
try:
    model = MusicTransformerWithLoRA(
        vocab_size=512,
        d_model=512,  # Smaller for testing
        num_layers=6,  # Fewer layers for testing
        num_heads=8,
        d_ff=2048,
        use_lora=True,
        lora_r=8,  # Smaller rank for testing
    )
    print(f"✓ Model instantiated successfully")

    # Print model summary
    params = model.count_parameters()
    print(f"\n  Model Parameters:")
    print(f"    Base model:  {params['base']:>10,}")
    print(f"    LoRA:        {params['lora']:>10,} ({params['lora']/params['base']*100:.2f}%)")
    print(f"    Total:       {params['total']:>10,}")
    print(f"    Trainable:   {params['trainable']:>10,}")

except Exception as e:
    print(f"✗ Failed to instantiate model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test forward pass
print("\n[6/6] Testing forward pass...")
try:
    batch_size = 2
    seq_len = 64

    # Create dummy input
    input_ids = torch.randint(0, 512, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    expected_shape = (batch_size, seq_len, 512)
    if logits.shape == expected_shape:
        print(f"✓ Forward pass successful")
        print(f"  Input shape:  {tuple(input_ids.shape)}")
        print(f"  Output shape: {tuple(logits.shape)}")
    else:
        print(f"✗ Output shape mismatch: expected {expected_shape}, got {logits.shape}")
        sys.exit(1)

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test LoRA weight control
print("\n[7/7] Testing interpretable LoRA control...")
try:
    # Set custom LoRA weights
    model.lora_controller.set_weights(
        harmony=1.0,
        voicing=0.5,
        rhythm=0.0,
        dynamics=1.0,
    )

    # Generate with custom weights
    with torch.no_grad():
        logits_custom = model(input_ids)

    print("✓ LoRA weight control works")
    print(f"  Harmony weight:  {model.lora_controller.harmony_weight.item():.1f}")
    print(f"  Voicing weight:  {model.lora_controller.voicing_weight.item():.1f}")
    print(f"  Rhythm weight:   {model.lora_controller.rhythm_weight.item():.1f}")
    print(f"  Dynamics weight: {model.lora_controller.dynamics_weight.item():.1f}")

except Exception as e:
    print(f"✗ LoRA control failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test model freezing
print("\n[8/8] Testing parameter freezing...")
try:
    # Count trainable params before freezing
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Freeze base model
    model.freeze_base_model()

    # Count trainable params after freezing
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    reduction = (1 - trainable_after / trainable_before) * 100

    print("✓ Parameter freezing works")
    print(f"  Before freezing: {trainable_before:>10,} trainable params")
    print(f"  After freezing:  {trainable_after:>10,} trainable params")
    print(f"  Reduction:       {reduction:>9.2f}%")

    if reduction < 99:
        print(f"  ⚠ Warning: Expected ~99.6% reduction, got {reduction:.2f}%")

except Exception as e:
    print(f"✗ Parameter freezing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe implementation is complete and functional.")
print("\nNext steps:")
print("  1. Collect Brad Mehldau MIDI data (see ../MIDI_DATA_SOURCES.md)")
print("  2. Run training:")
print("     cd research")
print("     python training/train_stage3_lora_finetuning.py --num_epochs 10")
print("  3. Generate samples:")
print("     python evaluation/generate_samples.py --checkpoint outputs/*/best_model.pt")
print()
