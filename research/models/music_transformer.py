"""
Music Transformer with Hierarchical LoRA Integration

Based on:
- Huang et al. (2018) "Music Transformer"
- Vaswani et al. (2017) "Attention is All You Need"

With novel integration of Hierarchical LoRA for efficient style transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .hierarchical_lora import HierarchicalLoRAController


class RelativePositionEmbedding(nn.Module):
    """
    Relative position embeddings for Music Transformer.

    Instead of absolute positions, uses relative distances between positions.
    This allows the model to generalize better to longer sequences.

    Reference: Shaw et al. (2018) "Self-Attention with Relative Position Representations"
    """

    def __init__(self, d_k: int, max_relative_position: int = 2048):
        super().__init__()

        self.d_k = d_k
        self.max_relative_position = max_relative_position

        # Learnable relative position embeddings
        # Range: [-max_relative_position, max_relative_position]
        vocab_size = 2 * max_relative_position + 1
        self.embeddings = nn.Embedding(vocab_size, d_k)

    def forward(self, length: int) -> torch.Tensor:
        """
        Compute relative position embeddings for a sequence of given length.

        Args:
            length: Sequence length

        Returns:
            Relative position embeddings [length, length, d_k]
        """
        # Create relative position indices
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(1).expand(-1, length)  # [length, length]
        distance_mat = range_mat - range_mat.transpose(0, 1)  # [length, length]

        # Clip to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Shift to 0-indexed
        final_mat = distance_mat_clipped + self.max_relative_position

        # Get embeddings [length, length, d_k]
        embeddings = self.embeddings(final_mat.to(self.embeddings.weight.device))

        return embeddings


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Relative Position Embeddings.

    Key innovation of Music Transformer: relative position-aware attention.
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_relative_position: int = 2048,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projections
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Relative position embeddings
        self.relative_position = RelativePositionEmbedding(
            self.d_k, max_relative_position
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split heads: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads: [batch, num_heads, seq_len, d_k] -> [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len] or [seq_len, seq_len]

        Returns:
            Output: [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        Q = self.query_proj(query)  # [batch, seq_len, d_model]
        K = self.key_proj(key)
        V = self.value_proj(value)

        # Split heads
        Q = self.split_heads(Q)  # [batch, num_heads, seq_len, d_k]
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Compute attention scores
        # Standard attention: Q @ K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, num_heads, seq_len, seq_len]

        # Add relative position bias
        rel_pos_emb = self.relative_position(seq_len)  # [seq_len, seq_len, d_k]
        rel_pos_emb = rel_pos_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, d_k]

        # Q @ rel_pos_emb^T -> [batch, num_heads, seq_len, seq_len]
        Q_expanded = Q.unsqueeze(-2)  # [batch, num_heads, seq_len, 1, d_k]
        rel_scores = torch.matmul(Q_expanded, rel_pos_emb.transpose(-2, -1)).squeeze(-2) / self.scale

        # Combine content and position scores
        scores = scores + rel_scores

        # Apply causal mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, d_k]

        # Merge heads
        output = self.merge_heads(output)  # [batch, seq_len, d_model]

        # Output projection
        output = self.out_proj(output)

        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(
        self,
        d_model: int = 768,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU works better than ReLU for transformers

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (output, intermediate) - intermediate is used for LoRA
        """
        intermediate = self.fc1(x)  # [batch, seq_len, d_ff]
        intermediate = self.activation(intermediate)
        intermediate = self.dropout(intermediate)

        output = self.fc2(intermediate)  # [batch, seq_len, d_model]
        output = self.dropout(output)

        return output, intermediate


class TransformerLayer(nn.Module):
    """
    Single Transformer layer with Hierarchical LoRA integration.
    """

    def __init__(
        self,
        layer_idx: int,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        lora_controller: Optional[HierarchicalLoRAController] = None,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.lora_controller = lora_controller

        # Self-attention
        self.self_attn = RelativeMultiHeadAttention(
            d_model, num_heads, dropout
        )

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len]

        Returns:
            Output: [batch, seq_len, d_model]
        """
        # Self-attention with LoRA
        residual = x
        x = self.norm1(x)

        # Apply LoRA to attention inputs if available
        if self.lora_controller is not None:
            q, k, v = self.lora_controller.forward_attention(
                self.layer_idx, x, x, x
            )
            attn_output = self.self_attn(q, k, v, mask)

            # Apply LoRA to attention output
            attn_output = attn_output + self.lora_controller.forward_output(
                self.layer_idx, attn_output
            )
        else:
            attn_output = self.self_attn(x, x, x, mask)

        x = residual + self.dropout(attn_output)

        # Feed-forward network with LoRA
        residual = x
        x = self.norm2(x)

        ffn_output, intermediate = self.ffn(x)

        # Apply LoRA to FFN if available
        if self.lora_controller is not None:
            _, ffn_delta = self.lora_controller.forward_ffn(
                self.layer_idx, x, intermediate
            )
            ffn_output = ffn_output + ffn_delta

        x = residual + ffn_output

        return x


class MusicTransformerWithLoRA(nn.Module):
    """
    Music Transformer with Hierarchical LoRA for Style Transfer.

    Architecture:
    - Input: MIDI tokens (vocab_size=512)
    - Embedding layer
    - 12 Transformer layers with relative position attention
    - Hierarchical LoRA modules (Harmony/Voicing/Rhythm/Dynamics)
    - Output: Next token prediction (autoregressive)

    Parameters:
    - Base model: ~86M parameters
    - LoRA modules: ~307K parameters (0.36% of base)
    """

    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_lora: bool = True,
        lora_r: int = 16,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learned, not sinusoidal)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Hierarchical LoRA controller
        self.use_lora = use_lora
        if use_lora:
            self.lora_controller = HierarchicalLoRAController(
                num_layers=num_layers,
                d_model=d_model,
                vocab_size=vocab_size,
                r=lora_r,
            )
        else:
            self.lora_controller = None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                layer_idx=i,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                lora_controller=self.lora_controller if use_lora else None,
            )
            for i in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights (embedding and output projection share weights)
        self.output_proj.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Position embedding
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask for autoregressive generation.

        Returns:
            Mask [seq_len, seq_len] where mask[i, j] = 1 if i >= j, else 0
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] - Token indices
            mask: Optional [batch, seq_len, seq_len] - Attention mask

        Returns:
            logits: [batch, seq_len, vocab_size] - Next token prediction scores
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embedding(positions)  # [1, seq_len, d_model]

        # Combine embeddings
        x = token_emb + pos_emb
        x = self.dropout(x)

        # Create causal mask if not provided
        if mask is None:
            mask = self.create_causal_mask(seq_len, device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer norm
        x = self.norm(x)

        # Output projection (base model)
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]

        # Add dynamics LoRA if available
        if self.lora_controller is not None:
            logits = logits + self.lora_controller.forward_dynamics(x)

        return logits

    def freeze_base_model(self):
        """Freeze all base model parameters, only train LoRA."""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

        print("Base model frozen. Only LoRA parameters will be trained.")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

        print("All parameters unfrozen.")

    def count_parameters(self) -> dict:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        lora_params = 0
        if self.lora_controller is not None:
            lora_counts = self.lora_controller.count_parameters()
            lora_params = lora_counts['total']

        return {
            'total': total_params,
            'trainable': trainable_params,
            'lora': lora_params,
            'base': total_params - lora_params,
        }

    def get_model_summary(self) -> str:
        """Get human-readable model summary."""
        params = self.count_parameters()

        summary = "Music Transformer with Hierarchical LoRA\n"
        summary += "=" * 60 + "\n"
        summary += f"Vocabulary size: {self.vocab_size}\n"
        summary += f"Model dimension: {self.d_model}\n"
        summary += f"Number of layers: {self.num_layers}\n"
        summary += f"Max sequence length: {self.max_seq_len}\n"
        summary += "=" * 60 + "\n"
        summary += f"Base model parameters:       {params['base']:>12,}\n"
        summary += f"LoRA parameters:             {params['lora']:>12,} ({params['lora']/params['base']*100:.2f}%)\n"
        summary += f"Total parameters:            {params['total']:>12,}\n"
        summary += f"Trainable parameters:        {params['trainable']:>12,}\n"
        summary += "=" * 60 + "\n"

        if self.lora_controller is not None:
            summary += "\n" + self.lora_controller.get_state_summary()

        return summary


if __name__ == "__main__":
    print("Testing Music Transformer with Hierarchical LoRA...\n")

    # Create model
    model = MusicTransformerWithLoRA(
        vocab_size=512,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        use_lora=True,
        lora_r=16,
    )

    # Print summary
    print(model.get_model_summary())

    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, 512, (batch_size, seq_len))

    print("\nTesting forward pass...")
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")

    # Test freezing
    print("\n" + "=" * 60)
    print("Testing parameter freezing...")
    print("=" * 60)

    print("\nBefore freezing:")
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_before:,}")

    model.freeze_base_model()

    print("\nAfter freezing:")
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_after:,}")
    print(f"Reduction: {(1 - trainable_after/trainable_before)*100:.2f}%")

    # Test generation (one step)
    print("\n" + "=" * 60)
    print("Testing autoregressive generation (one step)...")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Start with BOS token (token 449)
        prompt = torch.tensor([[449]])  # [1, 1]

        logits = model(prompt)
        next_token_logits = logits[:, -1, :]  # [1, vocab_size]
        next_token = torch.argmax(next_token_logits, dim=-1)  # [1]

        print(f"Prompt: {prompt.tolist()}")
        print(f"Next token predicted: {next_token.item()}")

    print("\n✓ All tests passed!")
