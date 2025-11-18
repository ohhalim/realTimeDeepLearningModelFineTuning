"""
Contrastive Style Encoder for Artist Style Embeddings

Key Innovation: Embed artists in continuous space using triplet loss
- Brad Mehldau, Bill Evans, Oscar Peterson → 128-dim vectors
- Enables smooth interpolation: α·Mehldau + (1-α)·Evans
- Supports multi-artist conditioning: Mehldau harmony + Peterson rhythm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class ContrastiveStyleEncoder(nn.Module):
    """Encode artist style into continuous embedding space

    Training: Triplet loss pulls same-artist samples close, pushes different-artist apart
    Inference: Interpolate between artists for novel style combinations
    """
    def __init__(self,
                 d_model: int = 768,
                 style_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.style_dim = style_dim

        # Style encoder: sequence → style embedding
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, style_dim),
        )

        # Optional: Multi-head pooling for better sequence aggregation
        self.multi_head_pool = MultiHeadPooling(d_model, num_heads=4)

        # Style projection (for conditioning transformer)
        self.style_proj = nn.Linear(style_dim, d_model)

        # Triplet loss
        self.triplet_loss = nn.TripletMarginLoss(
            margin=0.2,
            p=2,  # L2 distance
        )

    def forward(self, x: torch.Tensor, pool_method: str = 'mean') -> torch.Tensor:
        """Encode sequence to style embedding

        Args:
            x: [B, T, D] input sequence (MIDI tokens embedded)
            pool_method: 'mean', 'max', 'attention'

        Returns:
            style_emb: [B, style_dim] normalized style embedding
        """
        # Pool sequence to fixed representation
        if pool_method == 'mean':
            pooled = x.mean(dim=1)  # [B, D]
        elif pool_method == 'max':
            pooled = x.max(dim=1)[0]  # [B, D]
        elif pool_method == 'attention':
            pooled = self.multi_head_pool(x)  # [B, D]
        else:
            raise ValueError(f"Unknown pool_method: {pool_method}")

        # Encode to style space
        style_emb = self.encoder(pooled)  # [B, style_dim]

        # L2 normalize (important for cosine similarity)
        style_emb = F.normalize(style_emb, p=2, dim=-1)

        return style_emb

    def compute_triplet_loss(self,
                             anchor: torch.Tensor,
                             positive: torch.Tensor,
                             negative: torch.Tensor) -> torch.Tensor:
        """Compute contrastive triplet loss

        Args:
            anchor: [B, T, D] samples from artist A
            positive: [B, T, D] different samples from artist A (same artist)
            negative: [B, T, D] samples from artist B (different artist)

        Returns:
            loss: Scalar triplet loss
        """
        # Encode to style space
        anchor_emb = self(anchor)      # [B, style_dim]
        positive_emb = self(positive)  # [B, style_dim]
        negative_emb = self(negative)  # [B, style_dim]

        # Triplet loss: d(anchor, positive) + margin < d(anchor, negative)
        loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

        return loss

    def interpolate_styles(self,
                          style_a: torch.Tensor,
                          style_b: torch.Tensor,
                          alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between two artist styles

        Args:
            style_a: [style_dim] embedding of artist A (e.g., Mehldau)
            style_b: [style_dim] embedding of artist B (e.g., Evans)
            alpha: Interpolation weight [0, 1]
                   0 = pure A, 1 = pure B, 0.5 = balanced mix

        Returns:
            interpolated: [style_dim] blended style embedding
        """
        interpolated = (1 - alpha) * style_a + alpha * style_b

        # Renormalize
        interpolated = F.normalize(interpolated, p=2, dim=-1)

        return interpolated

    def multi_artist_blend(self,
                           styles: List[torch.Tensor],
                           weights: List[float]) -> torch.Tensor:
        """Blend multiple artist styles with custom weights

        Example: 50% Mehldau + 30% Evans + 20% Peterson

        Args:
            styles: List of [style_dim] style embeddings
            weights: List of blend weights (should sum to 1.0)

        Returns:
            blended: [style_dim] multi-artist blend
        """
        assert len(styles) == len(weights), "Styles and weights must match"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights should sum to 1.0"

        # Weighted sum
        blended = sum(w * s for w, s in zip(weights, styles))

        # Renormalize
        blended = F.normalize(blended, p=2, dim=-1)

        return blended

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two style embeddings"""
        return F.cosine_similarity(emb1, emb2, dim=-1).item()


class MultiHeadPooling(nn.Module):
    """Multi-head attention pooling for sequence aggregation

    Better than simple mean pooling: learns to attend to important parts
    """
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))

        # K, V projections
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence using attention

        Args:
            x: [B, T, D]

        Returns:
            pooled: [B, D]
        """
        B, T, D = x.shape

        # Project to K, V
        K = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        V = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand query for batch
        Q = self.query.expand(B, -1, -1).unsqueeze(2)  # [B, H, 1, d]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, T]
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        pooled = torch.matmul(attn_weights, V)  # [B, H, 1, d]

        # Reshape
        pooled = pooled.squeeze(2).reshape(B, D)  # [B, D]

        return pooled


class StyleArtistDatabase:
    """Database of artist style embeddings

    Stores pre-computed style embeddings for quick lookup and interpolation
    """
    def __init__(self):
        self.artists: Dict[str, torch.Tensor] = {}

    def add_artist(self, name: str, style_emb: torch.Tensor):
        """Add artist style embedding"""
        self.artists[name] = style_emb.detach().cpu()

    def get_artist(self, name: str) -> torch.Tensor:
        """Get artist style embedding"""
        if name not in self.artists:
            raise ValueError(f"Artist '{name}' not in database. Available: {list(self.artists.keys())}")
        return self.artists[name]

    def interpolate(self, artist_a: str, artist_b: str, alpha: float = 0.5) -> torch.Tensor:
        """Interpolate between two artists"""
        style_a = self.get_artist(artist_a)
        style_b = self.get_artist(artist_b)

        interpolated = (1 - alpha) * style_a + alpha * style_b
        return F.normalize(interpolated, p=2, dim=-1)

    def blend_multiple(self, artist_weights: Dict[str, float]) -> torch.Tensor:
        """Blend multiple artists

        Example: {'brad_mehldau': 0.5, 'bill_evans': 0.3, 'oscar_peterson': 0.2}
        """
        weights_sum = sum(artist_weights.values())
        assert abs(weights_sum - 1.0) < 1e-6, f"Weights sum to {weights_sum}, not 1.0"

        styles = [self.get_artist(name) for name in artist_weights.keys()]
        weights = list(artist_weights.values())

        blended = sum(w * s for w, s in zip(weights, styles))
        return F.normalize(blended, p=2, dim=-1)

    def find_nearest(self, style_emb: torch.Tensor, top_k: int = 3) -> List[Tuple[str, float]]:
        """Find k nearest artists to given style

        Args:
            style_emb: [style_dim] query embedding
            top_k: Number of nearest neighbors

        Returns:
            List of (artist_name, similarity_score)
        """
        similarities = []

        for name, artist_emb in self.artists.items():
            sim = F.cosine_similarity(style_emb, artist_emb, dim=-1).item()
            similarities.append((name, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def visualize_space(self, save_path: str = 'style_space.png'):
        """Visualize artist style space using t-SNE"""
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Need sklearn and matplotlib for visualization")
            return

        # Collect embeddings
        names = list(self.artists.keys())
        embeddings = torch.stack([self.artists[name] for name in names]).numpy()

        # t-SNE to 2D
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)

        for i, name in enumerate(names):
            plt.annotate(name, (coords[i, 0], coords[i, 1]),
                        fontsize=12, ha='center', va='bottom')

        plt.title('Artist Style Space (t-SNE)', fontsize=14)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")


def test_style_encoder():
    """Test Contrastive Style Encoder"""
    print("Testing Contrastive Style Encoder...")

    d_model = 768
    style_dim = 128
    batch_size = 4
    seq_len = 64

    # Create encoder
    encoder = ContrastiveStyleEncoder(d_model, style_dim)

    print("\n1. Testing forward pass...")
    x = torch.randn(batch_size, seq_len, d_model)
    style_emb = encoder(x)

    assert style_emb.shape == (batch_size, style_dim)
    assert torch.allclose(style_emb.norm(dim=-1), torch.ones(batch_size), atol=1e-5), "Not normalized!"
    print(f"  ✓ Style embedding shape: {style_emb.shape}")
    print(f"  ✓ L2 norm: {style_emb.norm(dim=-1)}")

    print("\n2. Testing triplet loss...")
    anchor = torch.randn(batch_size, seq_len, d_model)
    positive = torch.randn(batch_size, seq_len, d_model)
    negative = torch.randn(batch_size, seq_len, d_model)

    loss = encoder.compute_triplet_loss(anchor, positive, negative)
    print(f"  ✓ Triplet loss: {loss.item():.4f}")

    print("\n3. Testing style interpolation...")
    style_a = encoder(torch.randn(1, seq_len, d_model)).squeeze(0)  # [style_dim]
    style_b = encoder(torch.randn(1, seq_len, d_model)).squeeze(0)

    # Interpolate at different alphas
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interpolated = encoder.interpolate_styles(style_a, style_b, alpha)
        print(f"  α={alpha:.2f}: shape={interpolated.shape}, norm={interpolated.norm().item():.4f}")

    print("\n4. Testing multi-artist blend...")
    styles = [encoder(torch.randn(1, seq_len, d_model)).squeeze(0) for _ in range(3)]
    weights = [0.5, 0.3, 0.2]

    blended = encoder.multi_artist_blend(styles, weights)
    print(f"  ✓ Blended shape: {blended.shape}")
    print(f"  ✓ Blended norm: {blended.norm().item():.4f}")

    print("\n5. Testing artist database...")
    db = StyleArtistDatabase()

    # Add artists
    db.add_artist('brad_mehldau', styles[0])
    db.add_artist('bill_evans', styles[1])
    db.add_artist('oscar_peterson', styles[2])

    # Get artist
    mehldau = db.get_artist('brad_mehldau')
    print(f"  ✓ Retrieved 'brad_mehldau': shape={mehldau.shape}")

    # Interpolate
    blend = db.interpolate('brad_mehldau', 'bill_evans', alpha=0.3)
    print(f"  ✓ Interpolated (70% Mehldau + 30% Evans): shape={blend.shape}")

    # Multi-blend
    multi_blend = db.blend_multiple({
        'brad_mehldau': 0.5,
        'bill_evans': 0.3,
        'oscar_peterson': 0.2,
    })
    print(f"  ✓ Multi-blend (50/30/20): shape={multi_blend.shape}")

    # Find nearest
    query = styles[0] + 0.1 * torch.randn_like(styles[0])  # Slightly perturbed Mehldau
    query = F.normalize(query, p=2, dim=-1)
    nearest = db.find_nearest(query, top_k=3)
    print(f"  ✓ Nearest artists to query:")
    for name, sim in nearest:
        print(f"      {name}: {sim:.4f}")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_style_encoder()
