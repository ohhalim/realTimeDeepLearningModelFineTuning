"""
Hierarchical LoRA-CR (Corruption-Refinement)
Specialized LoRA modules for each corruption type

Key Innovation: Instead of shared LoRA across all tasks, each corruption
gets dedicated parameters for specialized refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LoRALayer(nn.Module):
    """Standard LoRA layer (Hu et al. 2021)

    Decomposes weight update as: W' = W + BA
    where A ∈ R^(r×d), B ∈ R^(d×r), r << d
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Gaussian, B with zeros
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA: output = x @ A^T @ B^T * scaling"""
        # x: [B, T, in_features]
        result = (x @ self.lora_A.T) @ self.lora_B.T  # [B, T, out_features]
        return result * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with optional LoRA"""
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, use_lora: bool = True):
        super().__init__()
        self.base_layer = base_layer
        self.use_lora = use_lora

        if use_lora:
            self.lora = LoRALayer(
                base_layer.in_features,
                base_layer.out_features,
                r=r,
                alpha=alpha
            )

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen)
        base_out = self.base_layer(x)

        # Add LoRA residual if enabled
        if self.use_lora:
            lora_out = self.lora(x)
            return base_out + lora_out
        else:
            return base_out


class CorruptionSpecificLoRA(nn.Module):
    """LoRA modules for a single corruption type across all layers"""
    def __init__(self, num_layers: int, d_model: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.r = r

        # LoRA for Q and V projections in each layer
        # (K is often omitted in practice, O can be added if needed)
        self.q_loras = nn.ModuleList([
            LoRALayer(d_model, d_model, r, alpha) for _ in range(num_layers)
        ])
        self.v_loras = nn.ModuleList([
            LoRALayer(d_model, d_model, r, alpha) for _ in range(num_layers)
        ])

    def forward(self, layer_idx: int, q: torch.Tensor, v: torch.Tensor):
        """Apply LoRA to Q and V at specified layer"""
        q_lora = self.q_loras[layer_idx](q)
        v_lora = self.v_loras[layer_idx](v)
        return q_lora, v_lora

    def num_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HierarchicalLoRACR(nn.Module):
    """Hierarchical LoRA for Corruption-Refinement

    Key Innovation: Separate LoRA modules per corruption type
    - NoCorrupt: Baseline LoRA (r=8)
    - TimeCrop: Temporal coherence LoRA (r=8)
    - NoteCrop: Harmonic reasoning LoRA (r=8)
    - GenreChange: Style transfer LoRA (r=16, 2× capacity)
    - PitchDropout: Melody generation LoRA (r=8)
    - VelocityDropout: Dynamics modeling LoRA (r=8)

    Total params: ~1.8M (1.43% of GPT-2 124M)
    """
    def __init__(self,
                 num_layers: int = 12,
                 d_model: int = 768,
                 lora_r: int = 8,
                 lora_alpha: int = 16):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # Corruption-specific LoRA modules
        self.corruption_loras = nn.ModuleDict({
            'NoCorrupt': CorruptionSpecificLoRA(num_layers, d_model, lora_r, lora_alpha),
            'TimeCrop': CorruptionSpecificLoRA(num_layers, d_model, lora_r, lora_alpha),
            'NoteCrop': CorruptionSpecificLoRA(num_layers, d_model, lora_r, lora_alpha),
            'GenreChange': CorruptionSpecificLoRA(num_layers, d_model, lora_r * 2, lora_alpha * 2),  # 2× for style
            'PitchDropout': CorruptionSpecificLoRA(num_layers, d_model, lora_r, lora_alpha),
            'VelocityDropout': CorruptionSpecificLoRA(num_layers, d_model, lora_r, lora_alpha),
        })

        # Corruption type classifier (for automatic detection at inference)
        self.corruption_classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6),  # 6 corruption types
        )

        # Corruption type name to index
        self.corruption_to_idx = {
            'NoCorrupt': 0,
            'TimeCrop': 1,
            'NoteCrop': 2,
            'GenreChange': 3,
            'PitchDropout': 4,
            'VelocityDropout': 5,
        }

    def forward(self,
                layer_idx: int,
                q: torch.Tensor,
                v: torch.Tensor,
                corruption_type: Optional[str] = None,
                hidden_state: Optional[torch.Tensor] = None):
        """Apply corruption-specific LoRA to Q and V

        Args:
            layer_idx: Which transformer layer (0-11)
            q: Query tensor [B, T, D]
            v: Value tensor [B, T, D]
            corruption_type: Name of corruption (e.g., 'GenreChange')
            hidden_state: For auto-detection if corruption_type not provided

        Returns:
            q_lora, v_lora: LoRA residuals to add to base Q and V
        """
        # Auto-detect corruption if not provided
        if corruption_type is None:
            assert hidden_state is not None, "Need hidden_state for auto-detection"
            corruption_type = self._detect_corruption(hidden_state)

        # Select specialized LoRA
        lora_module = self.corruption_loras[corruption_type]

        # Apply LoRA
        q_lora, v_lora = lora_module(layer_idx, q, v)

        return q_lora, v_lora

    def _detect_corruption(self, hidden_state: torch.Tensor) -> str:
        """Automatically detect corruption type from hidden state

        Uses a small classifier trained to recognize corruption patterns.
        Useful for zero-shot inference where corruption type is unknown.
        """
        # Pool sequence: [B, T, D] → [B, D]
        pooled = hidden_state.mean(dim=1)

        # Classify
        logits = self.corruption_classifier(pooled)  # [B, 6]
        pred_idx = logits.argmax(dim=-1).item()

        # Convert index to name
        idx_to_corruption = {v: k for k, v in self.corruption_to_idx.items()}
        return idx_to_corruption[pred_idx]

    def get_lora_parameters(self, corruption_type: Optional[str] = None):
        """Get parameters for specific corruption (or all)"""
        if corruption_type is not None:
            return self.corruption_loras[corruption_type].parameters()
        else:
            return self.parameters()

    def num_parameters(self, corruption_type: Optional[str] = None) -> Dict[str, int]:
        """Count parameters per corruption type"""
        if corruption_type is not None:
            return {
                corruption_type: self.corruption_loras[corruption_type].num_parameters()
            }
        else:
            return {
                name: module.num_parameters()
                for name, module in self.corruption_loras.items()
            }

    def parameter_efficiency(self, base_model_params: int = 124_000_000) -> str:
        """Report parameter efficiency"""
        total_lora_params = sum(self.num_parameters().values())
        classifier_params = sum(p.numel() for p in self.corruption_classifier.parameters())
        total_params = total_lora_params + classifier_params

        efficiency = (total_params / base_model_params) * 100

        report = f"Hierarchical LoRA-CR Parameter Report:\n"
        report += f"  Base model: {base_model_params:,} params\n"
        report += f"  LoRA params:\n"
        for name, count in self.num_parameters().items():
            report += f"    - {name}: {count:,}\n"
        report += f"  Classifier: {classifier_params:,}\n"
        report += f"  Total trainable: {total_params:,} ({efficiency:.2f}% of base)\n"

        return report


def test_hierarchical_lora():
    """Test Hierarchical LoRA-CR"""
    print("Testing Hierarchical LoRA-CR...")

    # Create model
    num_layers = 12
    d_model = 768
    batch_size = 2
    seq_len = 32

    lora_cr = HierarchicalLoRACR(
        num_layers=num_layers,
        d_model=d_model,
        lora_r=8,
        lora_alpha=16,
    )

    # Print parameter report
    print(lora_cr.parameter_efficiency())

    # Test forward pass for each corruption
    q = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    for corruption_type in lora_cr.corruption_loras.keys():
        print(f"\nTesting {corruption_type}...")

        for layer_idx in range(num_layers):
            q_lora, v_lora = lora_cr(
                layer_idx=layer_idx,
                q=q,
                v=v,
                corruption_type=corruption_type,
            )

            assert q_lora.shape == q.shape, f"Q shape mismatch at layer {layer_idx}"
            assert v_lora.shape == v.shape, f"V shape mismatch at layer {layer_idx}"

        print(f"  ✓ All {num_layers} layers passed")

    # Test auto-detection
    print("\nTesting auto-detection...")
    hidden = torch.randn(batch_size, seq_len, d_model)
    detected = lora_cr._detect_corruption(hidden)
    print(f"  Detected corruption: {detected}")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_hierarchical_lora()
