from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConditioningNote:
    time_ms: float
    pitch: int
    velocity: int


@dataclass
class GenerationRequest:
    request_id: int
    created_at_ms: float
    role: str
    tempo_bpm: float
    primer_tokens: List[int]
    max_length: int
    context_window: int
    temperature: float
    top_k: int
    top_p: float


@dataclass
class GenerationResult:
    request_id: int
    created_at_ms: float
    completed_at_ms: float
    status: str
    role: str
    tempo_bpm: float
    primer_tokens: List[int]
    generated_tokens: List[int]
    continuation_tokens: List[int]
    cache_hit: bool
    inference_ms: float
    error: Optional[str] = None


@dataclass(order=True)
class ScheduledMidiEvent:
    send_at_ms: float
    message_type: str = field(compare=False)
    note: int = field(compare=False)
    velocity: int = field(compare=False, default=0)
    channel: int = field(compare=False, default=0)
    source_request_id: int = field(compare=False, default=-1)

