"""
ReaLJazz: Real-Time Jazz Jam Bot

Real-time AI jazz accompaniment powered by Anticipatory Transformers

Based on latest SOTA research:
- Anticipatory Music Transformer (Stanford, 2024)
- ReaLJam (2025)
- MusicGen Streaming (Meta, 2024)
"""

from .model import AnticipativeJazzFormer, count_parameters
from .midi_io import create_midi_input, create_midi_output
from .jamming import JamSession, JamConfig, jam

__version__ = "0.1.0"
__all__ = [
    "AnticipativeJazzFormer",
    "count_parameters",
    "create_midi_input",
    "create_midi_output",
    "JamSession",
    "JamConfig",
    "jam",
]

# Quick start
def quick_jam():
    """Quick start a virtual jam session"""
    jam(mode="virtual")
