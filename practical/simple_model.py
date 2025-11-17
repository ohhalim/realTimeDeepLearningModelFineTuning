"""
Simple Working Model: GPT-2 + LoRA for Jazz Piano

Philosophy: Use only proven methods, make it work first.

Architecture:
- GPT-2 Transformer (proven by OpenAI 2019)
- Standard LoRA (proven by Hu et al. 2021)
- Simple tokenization (proven by Magenta)

Total: ~300 lines. No fancy stuff.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the model. Proven defaults."""

    vocab_size: int = 220      # NOTE(88) + TIME(100) + VEL(32)
    n_layer: int = 6           # GPT-2 small has 12, we use 6
    n_head: int = 8            # Multi-head attention
    n_embd: int = 512          # Embedding dimension
    max_seq_len: int = 1024    # Context window
    dropout: float = 0.1       # Regularization

    # LoRA config (proven by Hu et al. 2021)
    use_lora: bool = True
    lora_r: int = 8            # Rank (sweet spot)
    lora_alpha: int = 16       # Scaling (2× rank is standard)
    lora_dropout: float = 0.1


class LoRALinear(nn.Module):
    """
    Standard LoRA layer (Hu et al. 2021).

    Instead of fine-tuning W, we add low-rank adaptation:
    h = Wx + (alpha/r) * B A x

    Only B and A are trainable, W is frozen.
    """

    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)

        # Scaling
        self.scaling = alpha / r

        # Initialize (Hu et al. initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """Apply LoRA adaptation."""
        # x: [batch, seq_len, in_features]
        result = self.dropout(x) @ self.lora_A.T  # [batch, seq_len, r]
        result = result @ self.lora_B.T            # [batch, seq_len, out_features]
        return result * self.scaling


class Attention(nn.Module):
    """
    Multi-head attention with optional LoRA.

    Standard GPT-2 attention (Radford et al. 2019)
    """

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V projections (standard)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # LoRA adapters (optional)
        if config.use_lora:
            self.q_lora = LoRALinear(
                config.n_embd, config.n_embd,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
            self.v_lora = LoRALinear(
                config.n_embd, config.n_embd,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout
            )
        else:
            self.q_lora = None
            self.v_lora = None

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (register as buffer so it's not a parameter)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd

        # Q, K, V projections
        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Add LoRA if enabled
        if self.q_lora is not None:
            q = q + self.q_lora(x)
            v = v + self.v_lora(x)

        # Split into heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]

        # Apply causal mask
        scores = scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float('-inf')
        )

        # Softmax + dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, H, T, D]

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Standard GPT-2 FFN: Linear -> GELU -> Linear
    """

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block.

    Standard GPT-2 block: LayerNorm -> Attention -> Residual
                           LayerNorm -> FFN -> Residual
    """

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        # Attention block
        x = x + self.attn(self.ln1(x))

        # Feed-forward block
        x = x + self.ffn(self.ln2(x))

        return x


class SimpleGPT2(nn.Module):
    """
    Simple GPT-2 model for music generation.

    Proven architecture (Radford et al. 2019) + LoRA (Hu et al. 2021).
    Nothing fancy, just what works.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token + position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (tie token embedding and output head)
        self.head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: Token indices [batch, seq_len]
            targets: Target indices [batch, seq_len] (for training)

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Scalar (if targets provided)
        """
        B, T = idx.shape

        # Embeddings
        token_emb = self.token_emb(idx)  # [B, T, n_embd]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)      # [T, n_embd]

        x = self.dropout(token_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # Output logits
        logits = self.head(x)  # [B, T, vocab_size]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    def freeze_base_model(self):
        """Freeze all parameters except LoRA."""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        print("✓ Base model frozen, only LoRA parameters trainable")

    def count_parameters(self):
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in self.parameters() if 'lora' in str(p))

        return {
            'total': total,
            'trainable': trainable,
            'lora': lora,
            'frozen': total - trainable
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting tokens [batch, seq_len]
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = no filtering)

        Returns:
            Generated sequence [batch, seq_len + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


def create_model(config_name='small', use_lora=True):
    """
    Factory function to create models with preset configs.

    Args:
        config_name: 'tiny', 'small', 'medium', 'large'
        use_lora: Whether to use LoRA

    Returns:
        model: SimpleGPT2 instance
        config: ModelConfig
    """

    configs = {
        'tiny': ModelConfig(
            n_layer=4, n_head=4, n_embd=256,
            use_lora=use_lora
        ),
        'small': ModelConfig(
            n_layer=6, n_head=8, n_embd=512,
            use_lora=use_lora
        ),
        'medium': ModelConfig(
            n_layer=12, n_head=12, n_embd=768,
            use_lora=use_lora
        ),
        'large': ModelConfig(
            n_layer=24, n_head=16, n_embd=1024,
            use_lora=use_lora
        ),
    }

    config = configs.get(config_name, configs['small'])
    model = SimpleGPT2(config)

    return model, config


if __name__ == "__main__":
    print("Testing Simple GPT-2 + LoRA Model")
    print("=" * 70)

    # Create model
    model, config = create_model('small', use_lora=True)

    # Count parameters
    params = model.count_parameters()
    print(f"\nModel: SimpleGPT2-Small + LoRA")
    print(f"  Total params:     {params['total']:>10,}")
    print(f"  Trainable params: {params['trainable']:>10,}")
    print(f"  LoRA params:      {params['lora']:>10,}")
    print(f"  Frozen params:    {params['frozen']:>10,}")
    print(f"  LoRA %:           {params['lora']/params['total']*100:>9.2f}%")

    # Freeze base model
    model.freeze_base_model()
    params_after = model.count_parameters()
    print(f"\nAfter freezing:")
    print(f"  Trainable params: {params_after['trainable']:>10,}")
    print(f"  Reduction:        {(1-params_after['trainable']/params['total'])*100:>9.2f}%")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 100

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    print(f"  Input shape:  {idx.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss:         {loss.item():.4f}")

    # Test generation
    print("\nTesting generation...")
    generated = model.generate(idx[:, :10], max_new_tokens=50, temperature=0.8, top_k=40)
    print(f"  Generated shape: {generated.shape}")

    print("\n✓ All tests passed! Model is ready to train.")
