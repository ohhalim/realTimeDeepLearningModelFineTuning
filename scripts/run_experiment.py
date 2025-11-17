#!/usr/bin/env python3
"""
Single script to run full experiment.
Usage: python scripts/run_experiment.py --quick
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from jazzformer.model import JazzFormer, count_parameters
from jazzformer.data import create_dataloaders


def train_model(model, train_loader, val_loader, epochs=5, lr=0.001, device='cpu'):
    """Simple training loop"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining on {device}...")
    print(f"Epochs: {epochs}, LR: {lr}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("JazzFormer Experiment")
    print("=" * 60)
    
    # Create dataloaders
    print("\n[1] Creating datasets...")
    train_loader, val_loader = create_dataloaders(
        num_train=10 if args.quick else 100,
        num_val=2,
        batch_size=4,
        seq_len=128
    )
    
    # Baseline: Standard transformer (no harmonic embeddings)
    print("\n[2] Training baseline...")
    baseline = JazzFormer(d_model=256, n_heads=4, n_layers=4)
    # Disable harmonic embeddings for baseline
    baseline.harmonic_embedding = nn.Identity()
    
    print(f"Baseline params: {count_parameters(baseline):,}")
    baseline = train_model(baseline, train_loader, val_loader, epochs=3 if args.quick else 10, device=device)
    
    # Our model: With harmonic embeddings
    print("\n[3] Training JazzFormer (with harmonic embeddings)...")
    jazzformer = JazzFormer(d_model=256, n_heads=4, n_layers=4)
    print(f"JazzFormer params: {count_parameters(jazzformer):,}")
    jazzformer = train_model(jazzformer, train_loader, val_loader, epochs=3 if args.quick else 10, device=device)
    
    print("\n" + "=" * 60)
    print("✓ Experiment Complete!")
    print("=" * 60)
    print("\nNext: Compare results and generate samples")
    print("Run: python scripts/plot_results.py")


if __name__ == "__main__":
    main()
