"""
JazzFormer: Simple, Honest Jazz Generation Model

This is a minimal implementation focusing on one key innovation:
harmonic embeddings for jazz-aware music generation.
"""

from .model import JazzFormer, count_parameters
from .data import SimpleMIDIDataset, create_dataloaders

__version__ = "0.1.0"
__all__ = [
    "JazzFormer",
    "count_parameters",
    "SimpleMIDIDataset",
    "create_dataloaders",
]
