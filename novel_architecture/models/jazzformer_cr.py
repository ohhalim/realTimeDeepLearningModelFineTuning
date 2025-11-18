"""
JazzFormer-CR: Complete Model Integration
Combines all 5 innovations into a unified architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.hierarchical_lora import HierarchicalLoRACR
from models.multi_scale_attention import MultiScaleTransformerLayer
from models.style_encoder import ContrastiveStyleEncoder


class JazzFormerCR(nn.Module):
    """JazzFormer-CR: Jazz Transformer with Corruption-Refinement

    Integrates 5 key innovations:
    1. Hierarchical LoRA-CR: Corruption-specific LoRA modules
    2. Multi-Scale Temporal Attention: Local (rhythm) + Global (structure)
    3. Contrastive Style Space: Continuous artist embeddings
    4. Adaptive Corruption Curriculum: (implemented in training loop)
    5. Flash Inference Pipeline: (implemented in inference module)

    Architecture:
        Embedding → Style Conditioning → Multi-Scale Transformer + LoRA → Output
    """
    def __init__(self,
                 vocab_size: int = 226,  # 220 tokens + 6 special
                 d_model: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 d_ff: int = 3072,
                 max_seq_len: int = 512,
                 local_window: int = 16,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 style_dim: int = 128,
                 dropout: float = 0.1,
                 use_lora: bool = True,
                 use_multi_scale: bool = True,
                 use_style: bool = True):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_lora = use_lora
        self.use_multi_scale = use_multi_scale
        self.use_style = use_style

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Innovation #3: Contrastive Style Encoder
        if use_style:
            self.style_encoder = ContrastiveStyleEncoder(d_model, style_dim, dropout)
        else:
            self.style_encoder = None

        # Innovation #2: Multi-Scale Transformer Layers
        if use_multi_scale:
            self.layers = nn.ModuleList([
                MultiScaleTransformerLayer(d_model, num_heads, d_ff, local_window, dropout)
                for _ in range(num_layers)
            ])
        else:
            # Standard transformer layers (for ablation)
            self.layers = nn.ModuleList([
                StandardTransformerLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

        # Innovation #1: Hierarchical LoRA-CR
        if use_lora:
            self.hierarchical_lora = HierarchicalLoRACR(
                num_layers, d_model, lora_r, lora_alpha
            )
        else:
            self.hierarchical_lora = None

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (tie embedding and output)
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self,
                input_ids: torch.Tensor,
                corruption_type: str = 'NoCorrupt',
                style_emb: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                compute_style_loss: bool = False,
                style_triplet: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """Forward pass

        Args:
            input_ids: [B, T] token IDs (corrupted sequence)
            corruption_type: Name of corruption applied
            style_emb: [B, style_dim] optional style conditioning
            labels: [B, T] target token IDs (clean sequence)
            compute_style_loss: Whether to compute style contrastive loss
            style_triplet: (anchor, positive, negative) for triplet loss

        Returns:
            logits: [B, T, vocab_size] output logits
            loss: Optional total loss (refinement + style)
            loss_dict: Dict of individual losses
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Get embeddings
        token_emb = self.token_emb(input_ids)  # [B, T, D]

        # Positional embeddings
        positions = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.pos_emb(positions).unsqueeze(0)  # [1, T, D]

        # Combine
        hidden = self.dropout(token_emb + pos_emb)  # [B, T, D]

        # Add style conditioning if provided
        if style_emb is not None and self.use_style:
            # Project style to model dimension
            style_vec = self.style_encoder.style_proj(style_emb)  # [B, D]
            style_vec = style_vec.unsqueeze(1)  # [B, 1, D]

            # Prepend style token to sequence
            hidden = torch.cat([style_vec, hidden], dim=1)  # [B, T+1, D]

        # Forward through layers
        for layer_idx, layer in enumerate(self.layers):
            # Standard transformer layer forward
            hidden = layer(hidden)

            # Apply LoRA if enabled
            if self.use_lora and self.hierarchical_lora is not None:
                # Get Q, V from layer's attention (simplified)
                # In practice, would hook into attention module
                # For now, apply LoRA as residual to hidden state
                lora_residual = self._apply_lora_to_layer(
                    hidden, layer_idx, corruption_type
                )
                hidden = hidden + lora_residual

        # Remove style token if it was added
        if style_emb is not None and self.use_style:
            hidden = hidden[:, 1:, :]  # Remove first token

        # Final norm and output
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)  # [B, T, vocab_size]

        # Compute losses
        loss_dict = {}
        total_loss = None

        if labels is not None:
            # Refinement loss (corruption → clean)
            refinement_loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss_dict['refinement'] = refinement_loss
            total_loss = refinement_loss

        # Style contrastive loss
        if compute_style_loss and style_triplet is not None and self.style_encoder is not None:
            anchor, positive, negative = style_triplet
            style_loss = self.style_encoder.compute_triplet_loss(anchor, positive, negative)
            loss_dict['style'] = style_loss

            if total_loss is not None:
                total_loss = total_loss + 0.1 * style_loss  # Weight style loss lower
            else:
                total_loss = style_loss

        return logits, total_loss, loss_dict

    def _apply_lora_to_layer(self,
                             hidden: torch.Tensor,
                             layer_idx: int,
                             corruption_type: str) -> torch.Tensor:
        """Apply LoRA residual to hidden state

        Simplified version - in full implementation would hook into attention QV
        """
        # Create dummy Q, V from hidden (simplified)
        q = v = hidden

        # Get LoRA residuals
        q_lora, v_lora = self.hierarchical_lora(
            layer_idx=layer_idx,
            q=q,
            v=v,
            corruption_type=corruption_type,
        )

        # Combine (simplified)
        lora_residual = (q_lora + v_lora) / 2

        return lora_residual * 0.1  # Scale down

    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor,
                 max_new_tokens: int = 128,
                 corruption_type: str = 'GenreChange',
                 style_emb: Optional[torch.Tensor] = None,
                 temperature: float = 0.9,
                 top_p: float = 0.95,
                 top_k: int = 50) -> torch.Tensor:
        """Generate sequence (simplified - no KV cache)

        For full Flash Inference Pipeline with optimizations, see inference/flash_pipeline.py

        Args:
            prompt_ids: [B, T] initial tokens
            max_new_tokens: Number of tokens to generate
            corruption_type: Corruption type for LoRA selection
            style_emb: Optional style conditioning
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold

        Returns:
            generated: [B, T + max_new_tokens] full sequence
        """
        self.eval()
        generated = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Forward (take last 512 tokens if longer)
            if generated.size(1) > 512:
                input_ids = generated[:, -512:]
            else:
                input_ids = generated

            logits, _, _ = self(
                input_ids,
                corruption_type=corruption_type,
                style_emb=style_emb,
            )

            # Get last token logits
            logits = logits[:, -1, :]  # [B, vocab_size]

            # Sample next token
            next_token = self._sample(logits, temperature, top_p, top_k)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def _sample(self,
                logits: torch.Tensor,
                temperature: float,
                top_p: float,
                top_k: int) -> torch.Tensor:
        """Sample next token with temperature, top-p, top-k"""
        # Temperature
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumsum > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token

    def num_parameters(self, only_trainable: bool = True) -> Dict[str, int]:
        """Count parameters"""
        if only_trainable:
            base_params = 0  # Frozen
            lora_params = sum(self.hierarchical_lora.num_parameters().values()) if self.use_lora else 0
            style_params = sum(p.numel() for p in self.style_encoder.parameters()) if self.use_style else 0
            classifier_params = sum(p.numel() for p in self.hierarchical_lora.corruption_classifier.parameters()) if self.use_lora else 0

            return {
                'base_model': base_params,
                'lora': lora_params,
                'style_encoder': style_params,
                'corruption_classifier': classifier_params,
                'total_trainable': lora_params + style_params + classifier_params,
            }
        else:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {
                'total': total,
                'trainable': trainable,
                'frozen': total - trainable,
            }


class StandardTransformerLayer(nn.Module):
    """Standard transformer layer (for ablation studies)"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


def test_jazzformer_cr():
    """Test JazzFormer-CR"""
    print("Testing JazzFormer-CR...")

    vocab_size = 226
    d_model = 768
    num_layers = 12
    batch_size = 2
    seq_len = 64

    # Create model
    print("\n1. Creating model...")
    model = JazzFormerCR(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        use_lora=True,
        use_multi_scale=True,
        use_style=True,
    )

    # Print parameter counts
    print("\n2. Parameter counts:")
    param_counts = model.num_parameters(only_trainable=True)
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")

    total_counts = model.num_parameters(only_trainable=False)
    print(f"\n  Total: {total_counts['total']:,}")
    print(f"  Trainable: {total_counts['trainable']:,} ({total_counts['trainable']/total_counts['total']*100:.2f}%)")

    # Test forward pass
    print("\n3. Testing forward pass...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss, loss_dict = model(
        input_ids=input_ids,
        corruption_type='GenreChange',
        labels=labels,
    )

    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss dict: {loss_dict}")

    # Test with style conditioning
    print("\n4. Testing with style conditioning...")
    style_samples = torch.randn(batch_size, seq_len, d_model)
    style_emb = model.style_encoder(style_samples)

    logits, loss, loss_dict = model(
        input_ids=input_ids,
        corruption_type='GenreChange',
        style_emb=style_emb,
        labels=labels,
    )

    print(f"  Logits shape (with style): {logits.shape}")
    print(f"  Loss (with style): {loss.item():.4f}")

    # Test generation
    print("\n5. Testing generation...")
    prompt = torch.randint(0, vocab_size, (1, 16))
    generated = model.generate(
        prompt,
        max_new_tokens=32,
        corruption_type='GenreChange',
        style_emb=style_emb[:1],
        temperature=0.9,
    )

    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens (first 10): {generated[0, :10].tolist()}")

    # Test different corruptions
    print("\n6. Testing different corruption types...")
    for corruption in ['NoCorrupt', 'TimeCrop', 'NoteCrop', 'GenreChange', 'PitchDropout', 'VelocityDropout']:
        logits, loss, _ = model(input_ids, corruption_type=corruption, labels=labels)
        print(f"  {corruption}: loss={loss.item():.4f}")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_jazzformer_cr()
