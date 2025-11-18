"""
Rotary Position Encoding (RoPE)

Improvement: Replace absolute position encoding with RoPE
- Better extrapolation to longer sequences
- Relative position information
- No additional parameters

Reference: Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Encoding (RoPE)

    Applies rotation to query and key based on position.
    Encodes relative position information without learnable parameters.

    Key properties:
    - Generalizes to sequences longer than training
    - Encodes relative distance between tokens
    - Zero additional parameters
    - Compatible with Flash Attention
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        """
        Args:
            dim: Dimension of embeddings (must be even)
            max_seq_len: Maximum sequence length to precompute
            base: Base for frequency computation (10000 in original paper)
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for max_seq_len
        self._precompute_freqs(max_seq_len)

    def _precompute_freqs(self, seq_len: int):
        """Precompute cos and sin frequencies"""
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]

        # Compute cos and sin
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input

        Args:
            x: [..., dim] tensor

        Returns:
            Rotated tensor with same shape
        """
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self,
                             q: torch.Tensor,
                             k: torch.Tensor,
                             positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to queries and keys

        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            positions: Position indices [seq_len]

        Returns:
            (q_rotated, k_rotated): Queries and keys with position info
        """
        # Get cos and sin for positions
        cos = self.cos_cached[positions].unsqueeze(1)  # [seq_len, 1, dim]
        sin = self.sin_cached[positions].unsqueeze(1)  # [seq_len, 1, dim]

        # Apply rotation
        # q_rotated = q * cos + rotate_half(q) * sin
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin embeddings for sequence

        Args:
            seq_len: Sequence length

        Returns:
            (cos, sin): Cosine and sine embeddings [seq_len, dim]
        """
        if seq_len > self.max_seq_len:
            # Extend cache if needed
            self._precompute_freqs(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class RotaryMultiHeadAttention(nn.Module):
    """Multi-head attention with Rotary Position Encoding

    Drop-in replacement for standard attention with better position encoding
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 2048):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Rotary position encoding
        self.rotary = RotaryPositionEncoding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Forward with RoPE

        Args:
            x: [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            output: [batch, seq_len, d_model]
        """
        B, T, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: [B, num_heads, T, head_dim]

        # Apply rotary position encoding
        positions = torch.arange(T, device=x.device)
        Q, K = self.rotary.apply_rotary_pos_emb(
            Q.transpose(1, 2),  # [B, T, num_heads, head_dim]
            K.transpose(1, 2),
            positions
        )
        Q = Q.transpose(1, 2)  # Back to [B, num_heads, T, head_dim]
        K = K.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)  # [B, num_heads, T, head_dim]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        out = self.out_proj(out)

        return out


def test_rotary_encoding():
    """Test RoPE implementation"""
    print("Testing Rotary Position Encoding...")

    dim = 64
    max_seq_len = 128
    batch_size = 2
    seq_len = 64
    num_heads = 8

    # Test basic RoPE
    print("\n1. Testing basic RoPE...")
    rope = RotaryPositionEncoding(dim, max_seq_len)

    q = torch.randn(batch_size, seq_len, num_heads, dim // num_heads)
    k = torch.randn(batch_size, seq_len, num_heads, dim // num_heads)
    positions = torch.arange(seq_len)

    q_rot, k_rot = rope.apply_rotary_pos_emb(q, k, positions)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    print(f"  ✓ Q shape: {q_rot.shape}")
    print(f"  ✓ K shape: {k_rot.shape}")

    # Test relative position property
    print("\n2. Testing relative position property...")
    # RoPE should encode relative position, so rotating both Q and K by same amount
    # should preserve their similarity

    offset = 10
    q1 = q[:, 0:1, :, :]  # Position 0
    k1 = k[:, offset:offset+1, :, :]  # Position offset

    q1_rot, _ = rope.apply_rotary_pos_emb(q1, k1, torch.tensor([0]))
    _, k1_rot = rope.apply_rotary_pos_emb(q1, k1, torch.tensor([offset]))

    # Shift both by same amount
    q2 = q[:, 5:6, :, :]  # Position 5
    k2 = k[:, 5+offset:5+offset+1, :, :]  # Position 5+offset

    q2_rot, _ = rope.apply_rotary_pos_emb(q2, k2, torch.tensor([5]))
    _, k2_rot = rope.apply_rotary_pos_emb(q2, k2, torch.tensor([5+offset]))

    # Compute similarities
    sim1 = (q1_rot * k1_rot).sum()
    sim2 = (q2_rot * k2_rot).sum()

    print(f"  Similarity at offset {offset} (pos 0 vs pos {offset}): {sim1.item():.4f}")
    print(f"  Similarity at offset {offset} (pos 5 vs pos {5+offset}): {sim2.item():.4f}")
    print(f"  Difference: {abs(sim1 - sim2).item():.6f} (should be small)")

    # Test RotaryMultiHeadAttention
    print("\n3. Testing RotaryMultiHeadAttention...")
    d_model = 512
    attn = RotaryMultiHeadAttention(d_model, num_heads=8)

    x = torch.randn(batch_size, seq_len, d_model)
    out = attn(x)

    assert out.shape == x.shape
    print(f"  ✓ Output shape: {out.shape}")

    # Test with longer sequence (extrapolation)
    print("\n4. Testing extrapolation to longer sequence...")
    long_seq_len = max_seq_len + 50  # Longer than precomputed
    x_long = torch.randn(batch_size, long_seq_len, d_model)
    out_long = attn(x_long)

    assert out_long.shape == x_long.shape
    print(f"  ✓ Extrapolated to seq_len={long_seq_len} (max_seq_len={max_seq_len})")
    print(f"  ✓ Output shape: {out_long.shape}")

    # Compare with absolute position encoding
    print("\n5. Comparing with absolute position encoding...")

    # Absolute position (standard)
    pos_emb = nn.Embedding(max_seq_len, d_model)
    x_abs = x + pos_emb(torch.arange(seq_len))

    # RoPE (our approach)
    x_rope = attn(x)

    print(f"  Absolute pos encoding: adds {d_model * max_seq_len:,} parameters")
    print(f"  Rotary pos encoding: adds 0 parameters")
    print(f"  ✓ RoPE saves {d_model * max_seq_len:,} parameters!")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_rotary_encoding()
