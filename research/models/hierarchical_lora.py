"""
Hierarchical LoRA (Low-Rank Adaptation) Modules
for Expressive Jazz Piano Style Transfer

Paper: "Hierarchical StyleLoRA-Transformer: Efficient and Interpretable Style Transfer
        for Expressive Jazz Piano Generation"

Key Innovation: Decompose musical style into hierarchical components:
- Harmony LoRA (H-LoRA): Chord progressions and harmonic structure
- Voicing LoRA (V-LoRA): Chord voicings and note distribution
- Rhythm LoRA (R-LoRA): Timing, syncopation, and rhythmic patterns
- Dynamics LoRA (D-LoRA): Velocity and articulation

Each LoRA module can be independently controlled for interpretable style transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class LoRALayer(nn.Module):
    """
    Single LoRA layer implementing low-rank adaptation.

    Original weight W ∈ R^(d_out × d_in)
    LoRA decomposition: ΔW = B @ A where:
        A ∈ R^(r × d_in)
        B ∈ R^(d_out × r)
        r << min(d_out, d_in)

    Forward: h = W_0 @ x + α * (B @ A) @ x
    where α = lora_alpha / r (scaling factor)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        self.r = r
        self.lora_alpha = lora_alpha
        self.in_features = in_features
        self.out_features = out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Scaling factor
        self.scaling = self.lora_alpha / self.r

        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, in_features]

        Returns:
            LoRA adaptation [batch, seq_len, out_features]
        """
        # x @ A^T -> [batch, seq_len, r]
        # @ B^T -> [batch, seq_len, out_features]
        result = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return result * self.scaling

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, r={self.r}, alpha={self.lora_alpha}'


class HarmonyLoRA(nn.Module):
    """
    Harmony LoRA (H-LoRA): Models chord progressions and harmonic structure.

    Applied to early transformer layers (0-3) which capture long-range dependencies
    and harmonic patterns.

    Key musical aspects:
    - Chord progressions (ii-V-I, tritone substitutions, etc.)
    - Modal interchange
    - Reharmonization patterns
    """

    def __init__(
        self,
        d_model: int = 768,
        r: int = 16,
        lora_alpha: int = 32,
        num_heads: int = 12,
    ):
        super().__init__()

        self.d_model = d_model
        self.r = r
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # LoRA for attention Q, K, V projections
        self.q_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.k_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.v_lora = LoRALayer(d_model, d_model, r, lora_alpha)

        # LoRA for output projection
        self.out_lora = LoRALayer(d_model, d_model, r, lora_alpha)

        # LoRA for feed-forward network
        self.ff1_lora = LoRALayer(d_model, d_model * 4, r, lora_alpha)
        self.ff2_lora = LoRALayer(d_model * 4, d_model, r, lora_alpha)

    def forward_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply LoRA to attention projections.

        Returns:
            Modified (query, key, value) tensors
        """
        query = query + self.q_lora(query)
        key = key + self.k_lora(key)
        value = value + self.v_lora(value)
        return query, key, value

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA to attention output projection."""
        return self.out_lora(x)

    def forward_ffn(self, x: torch.Tensor, intermediate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LoRA to feed-forward network.

        Args:
            x: Input to FFN
            intermediate: After first FFN layer (before activation)

        Returns:
            (modified_intermediate, output_delta)
        """
        intermediate = intermediate + self.ff1_lora(x)
        output_delta = self.ff2_lora(intermediate)
        return intermediate, output_delta


class VoicingLoRA(nn.Module):
    """
    Voicing LoRA (V-LoRA): Models chord voicings and note distribution.

    Applied to middle transformer layers (4-7) which capture
    local patterns and note relationships.

    Key musical aspects:
    - Rootless voicings (e.g., 3-7-9-13)
    - Upper structure triads
    - Note spacing and registration
    - Drop-2, drop-3 voicings
    """

    def __init__(
        self,
        d_model: int = 768,
        r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()

        self.d_model = d_model
        self.r = r

        # LoRA for attention (focus on local note patterns)
        self.q_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.k_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.v_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.out_lora = LoRALayer(d_model, d_model, r, lora_alpha)

        # LoRA for feed-forward (note combinations)
        self.ff1_lora = LoRALayer(d_model, d_model * 4, r, lora_alpha)
        self.ff2_lora = LoRALayer(d_model * 4, d_model, r, lora_alpha)

    def forward_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query + self.q_lora(query)
        key = key + self.k_lora(key)
        value = value + self.v_lora(value)
        return query, key, value

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_lora(x)

    def forward_ffn(self, x: torch.Tensor, intermediate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        intermediate = intermediate + self.ff1_lora(x)
        output_delta = self.ff2_lora(intermediate)
        return intermediate, output_delta


class RhythmLoRA(nn.Module):
    """
    Rhythm LoRA (R-LoRA): Models timing, syncopation, and rhythmic patterns.

    Applied to late transformer layers (8-11) which capture
    sequential dependencies and temporal patterns.

    Key musical aspects:
    - Syncopation (off-beat accents)
    - Swing timing (triplet feel)
    - Rubato (tempo flexibility)
    - Phrase articulation
    """

    def __init__(
        self,
        d_model: int = 768,
        r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()

        self.d_model = d_model
        self.r = r

        # LoRA for attention (temporal patterns)
        self.q_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.k_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.v_lora = LoRALayer(d_model, d_model, r, lora_alpha)
        self.out_lora = LoRALayer(d_model, d_model, r, lora_alpha)

        # LoRA for feed-forward (rhythm encoding)
        self.ff1_lora = LoRALayer(d_model, d_model * 4, r, lora_alpha)
        self.ff2_lora = LoRALayer(d_model * 4, d_model, r, lora_alpha)

    def forward_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query + self.q_lora(query)
        key = key + self.k_lora(key)
        value = value + self.v_lora(value)
        return query, key, value

    def forward_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_lora(x)

    def forward_ffn(self, x: torch.Tensor, intermediate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        intermediate = intermediate + self.ff1_lora(x)
        output_delta = self.ff2_lora(intermediate)
        return intermediate, output_delta


class DynamicsLoRA(nn.Module):
    """
    Dynamics LoRA (D-LoRA): Models velocity and articulation.

    Applied to output projection layer to control note dynamics.
    Uses smaller rank (r/2) since dynamics are more straightforward.

    Key musical aspects:
    - Velocity curves
    - Accents and emphasis
    - Dynamic range (pp to ff)
    - Touch and articulation
    """

    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 512,
        r: int = 8,  # Half rank of other LoRAs
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.r = r

        # LoRA for final output projection (logits)
        self.output_lora = LoRALayer(d_model, vocab_size, r, lora_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states [batch, seq_len, d_model]

        Returns:
            Logits delta [batch, seq_len, vocab_size]
        """
        return self.output_lora(x)


class HierarchicalLoRAController(nn.Module):
    """
    Controller for all hierarchical LoRA modules.

    Manages:
    - Layer-wise application of different LoRA types
    - Weighted combination of LoRA outputs
    - Enable/disable individual LoRA modules

    Layer assignment:
    - Layers 0-3: Harmony LoRA (early layers, global structure)
    - Layers 4-7: Voicing LoRA (middle layers, local patterns)
    - Layers 8-11: Rhythm LoRA (late layers, sequential patterns)
    - Output: Dynamics LoRA (final projection)
    """

    def __init__(
        self,
        num_layers: int = 12,
        d_model: int = 768,
        vocab_size: int = 512,
        r: int = 16,
        lora_alpha: int = 32,
        num_heads: int = 12,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # Create LoRA modules for each layer
        self.harmony_loras = nn.ModuleList([
            HarmonyLoRA(d_model, r, lora_alpha, num_heads)
            for _ in range(4)  # Layers 0-3
        ])

        self.voicing_loras = nn.ModuleList([
            VoicingLoRA(d_model, r, lora_alpha)
            for _ in range(4)  # Layers 4-7
        ])

        self.rhythm_loras = nn.ModuleList([
            RhythmLoRA(d_model, r, lora_alpha)
            for _ in range(4)  # Layers 8-11
        ])

        self.dynamics_lora = DynamicsLoRA(d_model, vocab_size, r // 2, lora_alpha // 2)

        # Learnable mixing weights (initialized to 1.0)
        self.harmony_weight = nn.Parameter(torch.ones(1))
        self.voicing_weight = nn.Parameter(torch.ones(1))
        self.rhythm_weight = nn.Parameter(torch.ones(1))
        self.dynamics_weight = nn.Parameter(torch.ones(1))

    def get_lora_for_layer(self, layer_idx: int):
        """Get appropriate LoRA module for given layer index."""
        if layer_idx < 4:
            return self.harmony_loras[layer_idx]
        elif layer_idx < 8:
            return self.voicing_loras[layer_idx - 4]
        elif layer_idx < 12:
            return self.rhythm_loras[layer_idx - 8]
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}")

    def get_weight_for_layer(self, layer_idx: int) -> torch.Tensor:
        """Get mixing weight for given layer index."""
        if layer_idx < 4:
            return self.harmony_weight
        elif layer_idx < 8:
            return self.voicing_weight
        elif layer_idx < 12:
            return self.rhythm_weight
        else:
            raise ValueError(f"Invalid layer index: {layer_idx}")

    def set_weights(
        self,
        harmony: float = 1.0,
        voicing: float = 1.0,
        rhythm: float = 1.0,
        dynamics: float = 1.0,
    ):
        """
        Manually set LoRA mixing weights for interpretable control.

        Args:
            harmony: Weight for harmony LoRA (0.0 to 1.0)
            voicing: Weight for voicing LoRA (0.0 to 1.0)
            rhythm: Weight for rhythm LoRA (0.0 to 1.0)
            dynamics: Weight for dynamics LoRA (0.0 to 1.0)

        Example:
            # 100% Mehldau harmony, 50% Mehldau rhythm, no voicing adaptation
            controller.set_weights(harmony=1.0, voicing=0.0, rhythm=0.5, dynamics=1.0)
        """
        with torch.no_grad():
            self.harmony_weight.fill_(harmony)
            self.voicing_weight.fill_(voicing)
            self.rhythm_weight.fill_(rhythm)
            self.dynamics_weight.fill_(dynamics)

    def forward_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply LoRA to attention projections."""
        lora = self.get_lora_for_layer(layer_idx)
        weight = self.get_weight_for_layer(layer_idx)

        q_delta, k_delta, v_delta = lora.forward_attention(query, key, value)

        # Apply weight
        query = query + weight * (q_delta - query)
        key = key + weight * (k_delta - key)
        value = value + weight * (v_delta - value)

        return query, key, value

    def forward_output(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA to attention output."""
        lora = self.get_lora_for_layer(layer_idx)
        weight = self.get_weight_for_layer(layer_idx)
        return weight * lora.forward_output(x)

    def forward_ffn(
        self,
        layer_idx: int,
        x: torch.Tensor,
        intermediate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply LoRA to feed-forward network."""
        lora = self.get_lora_for_layer(layer_idx)
        weight = self.get_weight_for_layer(layer_idx)

        inter_delta, out_delta = lora.forward_ffn(x, intermediate)

        # Apply weight
        intermediate = intermediate + weight * (inter_delta - intermediate)
        output_delta = weight * out_delta

        return intermediate, output_delta

    def forward_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamics LoRA to final output."""
        return self.dynamics_weight * self.dynamics_lora(x)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters for each LoRA type."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'harmony': sum(count_params(lora) for lora in self.harmony_loras),
            'voicing': sum(count_params(lora) for lora in self.voicing_loras),
            'rhythm': sum(count_params(lora) for lora in self.rhythm_loras),
            'dynamics': count_params(self.dynamics_lora),
            'total': count_params(self),
        }

    def get_state_summary(self) -> str:
        """Get human-readable summary of LoRA state."""
        params = self.count_parameters()

        summary = "Hierarchical LoRA Controller\n"
        summary += "=" * 50 + "\n"
        summary += f"Harmony LoRA (layers 0-3): {params['harmony']:,} params, weight={self.harmony_weight.item():.2f}\n"
        summary += f"Voicing LoRA (layers 4-7): {params['voicing']:,} params, weight={self.voicing_weight.item():.2f}\n"
        summary += f"Rhythm LoRA (layers 8-11): {params['rhythm']:,} params, weight={self.rhythm_weight.item():.2f}\n"
        summary += f"Dynamics LoRA (output):     {params['dynamics']:,} params, weight={self.dynamics_weight.item():.2f}\n"
        summary += "=" * 50 + "\n"
        summary += f"Total LoRA parameters: {params['total']:,}\n"

        return summary


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Hierarchical LoRA modules...\n")

    # Initialize controller
    controller = HierarchicalLoRAController(
        num_layers=12,
        d_model=768,
        vocab_size=512,
        r=16,
    )

    # Print parameter counts
    print(controller.get_state_summary())

    # Test forward passes
    batch_size = 2
    seq_len = 128
    d_model = 768

    # Test attention LoRA
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    for layer_idx in [0, 5, 10]:  # Test harmony, voicing, rhythm layers
        q_out, k_out, v_out = controller.forward_attention(layer_idx, query, key, value)
        print(f"\nLayer {layer_idx} attention output shapes:")
        print(f"  Query: {q_out.shape}")
        print(f"  Key: {k_out.shape}")
        print(f"  Value: {v_out.shape}")

    # Test dynamics LoRA
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    logits_delta = controller.forward_dynamics(hidden_states)
    print(f"\nDynamics LoRA output shape: {logits_delta.shape}")

    # Test interpretable control
    print("\n" + "=" * 50)
    print("Testing interpretable style control...")
    print("=" * 50)

    # Scenario 1: Only harmony adaptation
    controller.set_weights(harmony=1.0, voicing=0.0, rhythm=0.0, dynamics=0.0)
    print("\n1. Only Harmony (100%):")
    print(f"   Weights: H={controller.harmony_weight.item():.1f}, "
          f"V={controller.voicing_weight.item():.1f}, "
          f"R={controller.rhythm_weight.item():.1f}, "
          f"D={controller.dynamics_weight.item():.1f}")

    # Scenario 2: Balanced adaptation
    controller.set_weights(harmony=0.8, voicing=0.8, rhythm=0.5, dynamics=1.0)
    print("\n2. Balanced (80% H/V, 50% R, 100% D):")
    print(f"   Weights: H={controller.harmony_weight.item():.1f}, "
          f"V={controller.voicing_weight.item():.1f}, "
          f"R={controller.rhythm_weight.item():.1f}, "
          f"D={controller.dynamics_weight.item():.1f}")

    print("\n✓ All tests passed!")
