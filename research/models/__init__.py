"""
Model implementations for Hierarchical StyleLoRA-Transformer
"""

from .hierarchical_lora import (
    LoRALayer,
    HarmonyLoRA,
    VoicingLoRA,
    RhythmLoRA,
    DynamicsLoRA,
    HierarchicalLoRAController,
)

from .music_transformer import (
    MusicTransformerWithLoRA,
    RelativeMultiHeadAttention,
    TransformerLayer,
)

from .style_encoder import (
    StyleEncoder,
    StyleContrastiveModel,
    TripletLoss,
    StyleMemoryBank,
)

__all__ = [
    # Hierarchical LoRA
    'LoRALayer',
    'HarmonyLoRA',
    'VoicingLoRA',
    'RhythmLoRA',
    'DynamicsLoRA',
    'HierarchicalLoRAController',
    # Music Transformer
    'MusicTransformerWithLoRA',
    'RelativeMultiHeadAttention',
    'TransformerLayer',
    # Style Encoder
    'StyleEncoder',
    'StyleContrastiveModel',
    'TripletLoss',
    'StyleMemoryBank',
]
