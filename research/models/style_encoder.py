"""
Style Encoder with Contrastive Learning

Learns style embeddings that distinguish Brad Mehldau from other jazz pianists
using triplet loss and multi-scale temporal features.

Key features:
- Bar-level aggregation for capturing style
- Contrastive learning (triplet loss)
- Style classification head
- Style interpolation for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BarLevelEncoder(nn.Module):
    """
    Encode event-level sequences into bar-level representations.

    Takes MIDI events and aggregates them into bar-level features
    that capture harmonic and rhythmic patterns.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Multi-head attention for aggregation
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])

        # Feedforward layers
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # Bar boundary detection (learned query)
        self.bar_query = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor, bar_boundaries: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Event-level features [batch, seq_len, d_model]
            bar_boundaries: Optional [batch, num_bars] - indices of bar boundaries

        Returns:
            Bar-level features [batch, num_bars, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # If bar boundaries not provided, assume 4/4 time with 16 events per bar
        if bar_boundaries is None:
            events_per_bar = 16
            num_bars = seq_len // events_per_bar

            # Reshape to [batch, num_bars, events_per_bar, d_model]
            x_bars = x[:, :num_bars*events_per_bar].reshape(
                batch_size, num_bars, events_per_bar, -1
            )

            # Use attention to aggregate events within each bar
            # Query: bar_query [1, 1, d_model]
            # Key/Value: events within bar [batch*num_bars, events_per_bar, d_model]
            x_bars_flat = x_bars.reshape(batch_size * num_bars, events_per_bar, -1)

            # Expand query for each bar
            query = self.bar_query.expand(batch_size * num_bars, -1, -1)

            # Process through attention layers
            out = query
            for attn, norm, ffn in zip(self.attn_layers, self.norms, self.ffns):
                # Self-attention on events + query
                kv = torch.cat([out, x_bars_flat], dim=1)  # [batch*num_bars, 1+events_per_bar, d_model]
                attn_out, _ = attn(out, kv, kv)
                out = norm(out + attn_out)

                # FFN
                ffn_out = ffn(out)
                out = norm(out + ffn_out)

            # Reshape back to [batch, num_bars, d_model]
            bar_features = out.squeeze(1).reshape(batch_size, num_bars, -1)

        else:
            # TODO: Handle explicit bar boundaries
            raise NotImplementedError("Explicit bar boundaries not yet implemented")

        return bar_features


class StyleEncoder(nn.Module):
    """
    Encode musical sequences into style embeddings.

    Architecture:
    1. Bar-level aggregation (capture structure)
    2. Temporal pooling (average over bars)
    3. Style projection (project to style space)
    4. L2 normalization (for cosine similarity)
    """

    def __init__(
        self,
        d_model: int = 768,
        d_style: int = 256,
        num_bar_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_style = d_style

        # Bar-level encoder
        self.bar_encoder = BarLevelEncoder(
            d_model=d_model,
            num_heads=8,
            num_layers=num_bar_layers,
            dropout=dropout,
        )

        # Style projection network
        self.style_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_style),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Event-level features [batch, seq_len, d_model]

        Returns:
            style_emb: Style embedding [batch, d_style] (L2-normalized)
            bar_features: Bar-level features [batch, num_bars, d_model]
        """
        # Encode to bar level
        bar_features = self.bar_encoder(x)  # [batch, num_bars, d_model]

        # Temporal pooling (mean over bars)
        pooled = bar_features.mean(dim=1)  # [batch, d_model]

        # Project to style space
        style_emb = self.style_projection(pooled)  # [batch, d_style]

        # L2 normalize for cosine similarity
        style_emb = F.normalize(style_emb, p=2, dim=-1)

        return style_emb, bar_features


class TripletLoss(nn.Module):
    """
    Triplet loss for style contrastive learning.

    L = max(0, margin + d(anchor, positive) - d(anchor, negative))

    where d is the distance function (Euclidean or cosine).
    """

    def __init__(
        self,
        margin: float = 0.5,
        distance: str = 'cosine',
    ):
        super().__init__()

        self.margin = margin
        self.distance = distance

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings [batch, d_style]
            positive: Positive embeddings (same artist) [batch, d_style]
            negative: Negative embeddings (different artist) [batch, d_style]

        Returns:
            loss: Scalar triplet loss
        """
        if self.distance == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # Embeddings are already L2-normalized
            pos_dist = 1 - (anchor * positive).sum(dim=-1)  # [batch]
            neg_dist = 1 - (anchor * negative).sum(dim=-1)  # [batch]

        elif self.distance == 'euclidean':
            # Euclidean distance
            pos_dist = torch.norm(anchor - positive, p=2, dim=-1)
            neg_dist = torch.norm(anchor - negative, p=2, dim=-1)

        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        # Triplet loss
        loss = torch.relu(self.margin + pos_dist - neg_dist)

        return loss.mean()


class StyleClassifier(nn.Module):
    """
    Classifier head for style identification.

    Used for:
    1. Auxiliary training objective
    2. Evaluation metric (style classification accuracy)
    """

    def __init__(
        self,
        d_style: int = 256,
        num_artists: int = 4,  # Brad Mehldau, Bill Evans, Keith Jarrett, Oscar Peterson
        dropout: float = 0.1,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(d_style, d_style),
            nn.LayerNorm(d_style),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_style, num_artists),
        )

    def forward(self, style_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_emb: Style embeddings [batch, d_style]

        Returns:
            logits: Classification logits [batch, num_artists]
        """
        return self.classifier(style_emb)


class StyleContrastiveModel(nn.Module):
    """
    Complete style contrastive learning model.

    Combines:
    - Style encoder
    - Triplet loss
    - Style classification
    """

    def __init__(
        self,
        d_model: int = 768,
        d_style: int = 256,
        num_artists: int = 4,
        triplet_margin: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_style = d_style
        self.num_artists = num_artists

        # Style encoder
        self.style_encoder = StyleEncoder(
            d_model=d_model,
            d_style=d_style,
            dropout=dropout,
        )

        # Triplet loss
        self.triplet_loss = TripletLoss(margin=triplet_margin, distance='cosine')

        # Style classifier
        self.style_classifier = StyleClassifier(
            d_style=d_style,
            num_artists=num_artists,
            dropout=dropout,
        )

    def forward(
        self,
        anchor_x: torch.Tensor,
        positive_x: Optional[torch.Tensor] = None,
        negative_x: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            anchor_x: Anchor sequences [batch, seq_len, d_model]
            positive_x: Positive sequences (same artist) [batch, seq_len, d_model]
            negative_x: Negative sequences (different artist) [batch, seq_len, d_model]
            labels: Artist labels [batch] (0=Mehldau, 1=Evans, 2=Jarrett, 3=Peterson)

        Returns:
            dict with keys:
                - style_emb: Style embeddings [batch, d_style]
                - loss: Total loss (if training data provided)
                - triplet_loss: Triplet loss component
                - classification_loss: Classification loss component
                - logits: Classification logits [batch, num_artists]
        """
        # Encode anchor
        style_emb, bar_features = self.style_encoder(anchor_x)

        outputs = {
            'style_emb': style_emb,
            'bar_features': bar_features,
        }

        # Classification
        logits = self.style_classifier(style_emb)
        outputs['logits'] = logits

        # Compute losses if training data provided
        total_loss = 0.0

        # Triplet loss
        if positive_x is not None and negative_x is not None:
            pos_emb, _ = self.style_encoder(positive_x)
            neg_emb, _ = self.style_encoder(negative_x)

            triplet_loss = self.triplet_loss(style_emb, pos_emb, neg_emb)
            outputs['triplet_loss'] = triplet_loss
            total_loss += triplet_loss

        # Classification loss
        if labels is not None:
            classification_loss = F.cross_entropy(logits, labels)
            outputs['classification_loss'] = classification_loss
            total_loss += classification_loss

        if total_loss > 0:
            outputs['loss'] = total_loss

        return outputs

    def get_style_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get style embedding for a sequence (inference mode).

        Args:
            x: Sequence [batch, seq_len, d_model]

        Returns:
            Style embedding [batch, d_style]
        """
        with torch.no_grad():
            style_emb, _ = self.style_encoder(x)
        return style_emb

    def interpolate_styles(
        self,
        style_a: torch.Tensor,
        style_b: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Interpolate between two style embeddings.

        Args:
            style_a: First style [batch, d_style]
            style_b: Second style [batch, d_style]
            alpha: Interpolation weight (0.0 = all A, 1.0 = all B)

        Returns:
            Interpolated style [batch, d_style]
        """
        interpolated = (1 - alpha) * style_a + alpha * style_b
        # Re-normalize
        interpolated = F.normalize(interpolated, p=2, dim=-1)
        return interpolated


class StyleMemoryBank(nn.Module):
    """
    Memory bank for storing reference style embeddings.

    Useful for:
    - Few-shot style transfer
    - Style retrieval
    - Nearest neighbor style matching
    """

    def __init__(
        self,
        d_style: int = 256,
        num_references: int = 100,
    ):
        super().__init__()

        self.d_style = d_style
        self.num_references = num_references

        # Memory bank (embeddings for reference pieces)
        self.register_buffer(
            'memory',
            torch.zeros(num_references, d_style)
        )

        # Labels (artist IDs for each reference)
        self.register_buffer(
            'labels',
            torch.zeros(num_references, dtype=torch.long)
        )

        # Validity mask (which slots are filled)
        self.register_buffer(
            'valid',
            torch.zeros(num_references, dtype=torch.bool)
        )

        self.num_stored = 0

    def add(self, embedding: torch.Tensor, label: int):
        """
        Add a reference embedding to the memory bank.

        Args:
            embedding: Style embedding [d_style]
            label: Artist ID
        """
        if self.num_stored < self.num_references:
            idx = self.num_stored
            self.memory[idx] = embedding
            self.labels[idx] = label
            self.valid[idx] = True
            self.num_stored += 1
        else:
            # Random replacement if memory is full
            idx = torch.randint(0, self.num_references, (1,)).item()
            self.memory[idx] = embedding
            self.labels[idx] = label

    def retrieve_nearest(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve k nearest neighbors from memory bank.

        Args:
            query: Query embedding [d_style]
            k: Number of neighbors

        Returns:
            (indices, similarities) where:
                indices: [k] - indices of nearest neighbors
                similarities: [k] - cosine similarities
        """
        # Filter valid entries
        valid_memory = self.memory[self.valid]  # [num_valid, d_style]

        # Compute cosine similarities
        query_norm = F.normalize(query.unsqueeze(0), p=2, dim=-1)
        memory_norm = F.normalize(valid_memory, p=2, dim=-1)
        similarities = (query_norm @ memory_norm.T).squeeze(0)  # [num_valid]

        # Get top-k
        topk_sims, topk_indices = torch.topk(similarities, min(k, len(similarities)))

        # Map back to original indices
        valid_indices = torch.where(self.valid)[0]
        original_indices = valid_indices[topk_indices]

        return original_indices, topk_sims


if __name__ == "__main__":
    print("Testing Style Encoder with Contrastive Learning...\n")

    # Create model
    model = StyleContrastiveModel(
        d_model=768,
        d_style=256,
        num_artists=4,
        triplet_margin=0.5,
    )

    print("Model created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 256  # 16 bars × 16 events
    d_model = 768

    # Create dummy data
    anchor = torch.randn(batch_size, seq_len, d_model)
    positive = torch.randn(batch_size, seq_len, d_model)
    negative = torch.randn(batch_size, seq_len, d_model)
    labels = torch.randint(0, 4, (batch_size,))  # 4 artists

    print("\nTesting forward pass...")
    outputs = model(anchor, positive, negative, labels)

    print(f"Style embedding shape: {outputs['style_emb'].shape}")
    print(f"Bar features shape: {outputs['bar_features'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Total loss: {outputs['loss'].item():.4f}")
    print(f"  Triplet loss: {outputs['triplet_loss'].item():.4f}")
    print(f"  Classification loss: {outputs['classification_loss'].item():.4f}")

    # Test style interpolation
    print("\n" + "=" * 60)
    print("Testing style interpolation...")
    print("=" * 60)

    style_a = outputs['style_emb'][0:1]  # Brad Mehldau
    style_b = outputs['style_emb'][1:2]  # Bill Evans

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        interpolated = model.interpolate_styles(style_a, style_b, alpha)
        similarity_a = F.cosine_similarity(interpolated, style_a, dim=-1)
        similarity_b = F.cosine_similarity(interpolated, style_b, dim=-1)

        print(f"α={alpha:.2f}: sim(A)={similarity_a.item():.3f}, sim(B)={similarity_b.item():.3f}")

    # Test memory bank
    print("\n" + "=" * 60)
    print("Testing style memory bank...")
    print("=" * 60)

    memory_bank = StyleMemoryBank(d_style=256, num_references=10)

    # Add reference embeddings
    for i in range(5):
        emb = torch.randn(256)
        emb = F.normalize(emb, p=2, dim=-1)
        memory_bank.add(emb, label=i % 4)

    print(f"Stored {memory_bank.num_stored} reference embeddings")

    # Retrieve nearest neighbors
    query = torch.randn(256)
    query = F.normalize(query, p=2, dim=-1)
    indices, sims = memory_bank.retrieve_nearest(query, k=3)

    print(f"\nTop-3 nearest neighbors:")
    for idx, sim in zip(indices, sims):
        print(f"  Index {idx.item()}: similarity={sim.item():.3f}")

    print("\n✓ All tests passed!")
