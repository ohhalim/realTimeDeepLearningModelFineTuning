#!/usr/bin/env python3
"""
Test script to verify JazzFormer-RT model works correctly
"""

import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Testing JazzFormer-RT Model")
print("=" * 60)

# Test 1: Import modules
print("\n[Test 1] Importing modules...")
try:
    from src.models.jazzformer_rt import JazzFormerRT
    from src.data.dataset import MIDITokenizer, JazzMIDIDataset
    from src.evaluation.metrics import compute_perplexity
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create model
print("\n[Test 2] Creating model...")
try:
    model = JazzFormerRT(
        vocab_size=388,
        d_model=256,  # Smaller for testing
        n_heads=4,
        n_layers=2,
        d_ff=1024,
        max_seq_len=512
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Forward pass
print("\n[Test 3] Testing forward pass...")
try:
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, 388, (batch_size, seq_len))
    artist_id = torch.tensor([0, 1])
    
    logits = model(x, artist_id)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    
    assert logits.shape == (batch_size, seq_len, 388), f"Unexpected output shape: {logits.shape}"
    print(f"✓ Output shape correct")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generation
print("\n[Test 4] Testing generation...")
try:
    prompt = torch.randint(0, 388, (1, 16))
    generated = model.generate(
        prompt=prompt,
        max_length=32,
        temperature=1.0,
        artist_id=0
    )
    print(f"✓ Generation successful")
    print(f"  Prompt shape: {prompt.shape}")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated length: {generated.shape[1]}")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Save and load model
print("\n[Test 5] Testing save and load...")
try:
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pt')
        
        # Save
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'architecture': {
                    'd_model': 256,
                    'n_heads': 4,
                    'n_layers': 2,
                    'd_ff': 1024
                },
                'io': {
                    'vocab_size': 388,
                    'max_sequence_length': 512
                }
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Model saved to {checkpoint_path}")
        
        # Load
        new_model = JazzFormerRT(
            vocab_size=388,
            d_model=256,
            n_heads=4,
            n_layers=2,
            d_ff=1024,
            max_seq_len=512
        )
        loaded_checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        print(f"✓ Model loaded successfully")
        
        # Test loaded model
        with torch.no_grad():
            logits_orig = model(x, artist_id)
            logits_loaded = new_model(x, artist_id)
        
        diff = (logits_orig - logits_loaded).abs().max().item()
        print(f"✓ Max difference between original and loaded: {diff:.10f}")
        assert diff < 1e-5, f"Models differ too much: {diff}"
        
except Exception as e:
    print(f"✗ Save/load failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: MIDI Tokenizer
print("\n[Test 6] Testing MIDI Tokenizer...")
try:
    tokenizer = MIDITokenizer()
    print(f"✓ Tokenizer created")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Note on offset: {tokenizer.note_on_offset}")
    print(f"  Note off offset: {tokenizer.note_off_offset}")
    
    # Create sample tokens
    sample_tokens = torch.tensor([
        tokenizer.time_offset + 10,  # Time shift 100ms
        tokenizer.note_on_offset + 60,  # Middle C on
        tokenizer.velocity_offset + 20,  # Velocity
        tokenizer.time_offset + 50,  # Time shift 500ms
        tokenizer.note_off_offset + 60,  # Middle C off
    ])
    print(f"✓ Sample token sequence created: {sample_tokens.tolist()}")
    
except Exception as e:
    print(f"✗ Tokenizer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe model is ready to use!")
print("\nNext steps:")
print("1. Prepare MIDI data in data/brad_mehldau/train/")
print("2. Run: python scripts/train.py --data_dir data/brad_mehldau --epochs 10")
print("3. Generate: python scripts/generate.py --checkpoint models/finetuned/brad_mehldau/best.pt")
print("=" * 60)
