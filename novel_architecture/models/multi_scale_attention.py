"""
Multi-Scale Temporal Attention for Music Generation

Key Innovation: Dual-path attention with local (rhythm) and global (structure) branches
- Local attention (window=16): Captures micro-patterns like syncopation, swing feel
- Global attention (full seq): Captures macro-structure like phrase boundaries, form
- Cross-scale fusion: Combines both representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SlidingWindowAttention(nn.Module):
    """Efficient sliding window attention for local patterns

    Complexity: O(T × W) where T=sequence length, W=window size
    vs O(T²) for full attention
    """
    def __init__(self, d_model: int, num_heads: int, window_size: int = 16, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply sliding window attention

        Args:
            x: [B, T, D] input tensor
            mask: Optional attention mask [B, T, T]

        Returns:
            output: [B, T, D] after local attention
        """
        B, T, D = x.shape
        W = self.window_size

        # Project to Q, K, V
        Q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        K = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention for each position with its local window
        outputs = []

        for i in range(T):
            # Define window [i - W//2, i + W//2]
            start = max(0, i - W // 2)
            end = min(T, i + W // 2 + 1)

            # Query for position i
            q_i = Q[:, :, i:i+1, :]  # [B, H, 1, d]

            # Keys and values in window
            k_window = K[:, :, start:end, :]  # [B, H, W, d]
            v_window = V[:, :, start:end, :]  # [B, H, W, d]

            # Attention scores
            scores = torch.matmul(q_i, k_window.transpose(-2, -1)) * self.scale  # [B, H, 1, W]

            # Apply mask if provided
            if mask is not None:
                mask_window = mask[:, i:i+1, start:end].unsqueeze(1)  # [B, 1, 1, W]
                scores = scores.masked_fill(mask_window == 0, float('-inf'))

            # Softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Weighted sum
            out_i = torch.matmul(attn_weights, v_window)  # [B, H, 1, d]
            outputs.append(out_i)

        # Concatenate all positions
        output = torch.cat(outputs, dim=2)  # [B, H, T, d]

        # Reshape and project
        output = output.transpose(1, 2).reshape(B, T, D)  # [B, T, D]
        output = self.out_proj(output)

        return output


class MultiScaleTemporalAttention(nn.Module):
    """Dual-path attention: local (rhythm) + global (structure)

    Architecture:
        Input → [Local Path, Global Path] → Fusion → Output
                     ↓            ↓
                 Rhythm       Structure
                 Features     Features
    """
    def __init__(self,
                 d_model: int = 768,
                 num_heads: int = 12,
                 local_window: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.local_window = local_window

        # Local attention (sliding window for rhythm)
        self.local_attn = SlidingWindowAttention(
            d_model, num_heads // 2, local_window, dropout
        )

        # Global attention (full sequence for structure)
        self.global_attn = nn.MultiheadAttention(
            d_model, num_heads // 2, dropout=dropout, batch_first=True
        )

        # Cross-scale fusion network
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with dual-path attention

        Args:
            x: [B, T, D] input sequence
            mask: Optional attention mask

        Returns:
            output: [B, T, D] with multi-scale temporal features
        """
        residual = x

        # Local attention (captures rhythm patterns, micro-timing)
        local_out = self.local_attn(x, mask)
        local_out = self.norm1(local_out)

        # Global attention (captures overall structure, phrase boundaries)
        global_out, _ = self.global_attn(x, x, x, attn_mask=mask, need_weights=False)
        global_out = self.norm2(global_out)

        # Concatenate local and global features
        combined = torch.cat([local_out, global_out], dim=-1)  # [B, T, 2D]

        # Cross-scale fusion
        fused = self.fusion(combined)  # [B, T, D]

        # Residual connection
        output = fused + residual

        return output


class MultiScaleTransformerLayer(nn.Module):
    """Transformer layer with multi-scale attention + FFN

    Replaces standard self-attention with multi-scale temporal attention.
    Compatible with GPT-2 architecture.
    """
    def __init__(self,
                 d_model: int = 768,
                 num_heads: int = 12,
                 d_ff: int = 3072,
                 local_window: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        # Multi-scale attention
        self.attn = MultiScaleTemporalAttention(
            d_model, num_heads, local_window, dropout
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-scale attention with pre-norm
        x = x + self.attn(self.norm1(x), mask)

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x


class MultiScaleTransformer(nn.Module):
    """Stack of multi-scale transformer layers"""
    def __init__(self,
                 num_layers: int = 12,
                 d_model: int = 768,
                 num_heads: int = 12,
                 d_ff: int = 3072,
                 local_window: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            MultiScaleTransformerLayer(d_model, num_heads, d_ff, local_window, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


def visualize_attention_patterns():
    """Visualize local vs global attention patterns"""
    import matplotlib.pyplot as plt
    import numpy as np

    seq_len = 64
    local_window = 16

    # Local attention pattern (sliding window)
    local_pattern = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        start = max(0, i - local_window // 2)
        end = min(seq_len, i + local_window // 2 + 1)
        local_pattern[i, start:end] = 1

    # Global attention pattern (full)
    global_pattern = np.ones((seq_len, seq_len))

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(local_pattern, cmap='Blues')
    axes[0].set_title('Local Attention (Rhythm)\nWindow=16')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    axes[1].imshow(global_pattern, cmap='Greens')
    axes[1].set_title('Global Attention (Structure)\nFull Sequence')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')

    # Combined (both patterns active)
    combined = local_pattern + global_pattern
    axes[2].imshow(combined, cmap='Purples')
    axes[2].set_title('Multi-Scale Attention\nLocal + Global')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig('multi_scale_attention_patterns.png', dpi=150)
    print("Saved visualization to multi_scale_attention_patterns.png")


def test_multi_scale_attention():
    """Test Multi-Scale Temporal Attention"""
    print("Testing Multi-Scale Temporal Attention...")

    batch_size = 2
    seq_len = 64
    d_model = 768
    num_heads = 12
    local_window = 16

    # Test single layer
    print("\n1. Testing single MultiScaleTemporalAttention layer...")
    attn = MultiScaleTemporalAttention(d_model, num_heads, local_window)

    x = torch.randn(batch_size, seq_len, d_model)
    output = attn(x)

    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    print(f"  ✓ Output shape: {output.shape}")

    # Test full transformer
    print("\n2. Testing MultiScaleTransformer (12 layers)...")
    transformer = MultiScaleTransformer(
        num_layers=12,
        d_model=d_model,
        num_heads=num_heads,
        local_window=local_window,
    )

    output = transformer(x)
    assert output.shape == x.shape
    print(f"  ✓ Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")

    # Test with mask
    print("\n3. Testing with attention mask...")
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, seq_len//2:] = 0  # Mask out second half

    output_masked = transformer(x, mask)
    assert output_masked.shape == x.shape
    print(f"  ✓ Masked output shape: {output_masked.shape}")

    # Compare local vs global contributions
    print("\n4. Analyzing local vs global contributions...")

    # Forward through local path only
    local_out = attn.local_attn(x)
    print(f"  Local output norm: {local_out.norm().item():.4f}")

    # Forward through global path only
    global_out, _ = attn.global_attn(x, x, x, need_weights=False)
    print(f"  Global output norm: {global_out.norm().item():.4f}")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_multi_scale_attention()

    # Uncomment to generate visualization
    # visualize_attention_patterns()
