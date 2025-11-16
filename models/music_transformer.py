#!/usr/bin/env python3
"""
PyTorch Music Transformer 모델 구현

Music Transformer 논문 기반:
https://arxiv.org/abs/1809.04281
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionEmbedding(nn.Module):
    """
    Relative Position Embedding for Music Transformer

    긴 시퀀스를 처리하기 위한 상대적 위치 임베딩
    """

    def __init__(self, d_model, max_relative_position=2048):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position

        # Relative position embeddings
        self.embeddings = nn.Embedding(
            2 * max_relative_position + 1,
            d_model
        )

    def forward(self, length):
        """Generate relative position embeddings"""
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).expand(length, -1)
        distance_mat = range_mat - range_mat.transpose(0, 1)

        # Clip to max relative position
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Shift to make it positive
        final_mat = distance_mat_clipped + self.max_relative_position

        embeddings = self.embeddings(final_mat)
        return embeddings


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Relative Position Embeddings
    """

    def __init__(self, d_model, num_heads, dropout=0.1, max_relative_position=2048):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        # Relative position
        self.relative_position = RelativePositionEmbedding(
            self.d_k, max_relative_position
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections and reshape
        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Get relative position embeddings
        rel_pos_emb = self.relative_position(seq_len).to(query.device)

        # Scaled dot-product attention with relative position
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative position bias
        # Simplified version - full implementation would use skewing
        rel_scores = torch.matmul(
            Q.view(batch_size * self.num_heads, seq_len, self.d_k),
            rel_pos_emb.transpose(-2, -1)
        ).view(batch_size, self.num_heads, seq_len, seq_len)
        scores = scores + rel_scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.out_linear(context)
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerLayer(nn.Module):
    """Single Transformer Layer"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, max_relative_position=2048):
        super().__init__()

        self.attention = RelativeMultiHeadAttention(
            d_model, num_heads, dropout, max_relative_position
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)

        return x


class MusicTransformer(nn.Module):
    """
    Music Transformer for MIDI Generation

    Args:
        vocab_size: 어휘 크기 (MIDI 이벤트 수)
        d_model: 모델 차원
        num_heads: Attention head 수
        num_layers: Transformer 레이어 수
        d_ff: Feed-forward 차원
        max_seq_len: 최대 시퀀스 길이
        dropout: Dropout 비율
    """

    def __init__(
        self,
        vocab_size=512,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=2048,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Forward pass

        Args:
            x: (batch_size, seq_len) - Token indices
            mask: (batch_size, 1, seq_len, seq_len) - Attention mask

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Token embedding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        primer,
        max_length=1024,
        temperature=1.0,
        top_k=40,
        top_p=0.9,
        device='cuda'
    ):
        """
        Auto-regressive generation

        Args:
            primer: (seq_len,) - Primer sequence
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            generated: (max_length,) - Generated sequence
        """
        self.eval()

        if isinstance(primer, list):
            generated = torch.tensor(primer, dtype=torch.long).to(device)
        else:
            generated = primer.to(device)

        generated = generated.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            for _ in range(max_length - len(primer)):
                # Get predictions
                logits = self.forward(generated)
                logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

        return generated.squeeze(0).cpu().numpy()


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = MusicTransformer(
        vocab_size=512,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=2048
    )

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, 512, (batch_size, seq_len))

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test generation
    primer = [1, 2, 3, 4, 5]
    generated = model.generate(primer, max_length=50, device='cpu')
    print(f"Generated sequence length: {len(generated)}")
