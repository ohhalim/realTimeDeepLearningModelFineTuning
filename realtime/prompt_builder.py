from collections import deque
from typing import Deque, Dict, Optional

from models.midi_dataset import MIDITokenizer
from realtime.types import ConditioningNote, GenerationRequest


class PromptBuilder:
    """
    Build generation requests from incoming MIDI messages.

    Stage A strategy:
    - buffer chord/gesture for `chord_buffer_ms`
    - trigger request on debounced note activity
    """

    def __init__(
        self,
        tokenizer: MIDITokenizer,
        chord_buffer_ms: float = 100.0,
        lookback_notes: int = 60,
    ):
        self.tokenizer = tokenizer
        self.chord_buffer_ms = float(chord_buffer_ms)
        self.lookback_notes = int(lookback_notes)

        self._recent_notes: Deque[ConditioningNote] = deque(maxlen=self.lookback_notes * 2)
        self._active_notes: Dict[int, int] = {}
        self._dirty = False
        self._last_input_ms = 0.0
        self._last_trigger_ms = -1e9
        self._request_id = 0

    def ingest_midi(self, msg, timestamp_ms: float) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            note = int(msg.note)
            velocity = int(msg.velocity)
            self._active_notes[note] = velocity
            self._recent_notes.append(
                ConditioningNote(time_ms=float(timestamp_ms), pitch=note, velocity=velocity)
            )
            self._dirty = True
            self._last_input_ms = float(timestamp_ms)
            return

        if msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            self._active_notes.pop(int(msg.note), None)
            self._last_input_ms = float(timestamp_ms)

    def _encode_conditioning_tokens(self, max_conditioning_tokens: int) -> list:
        notes = list(self._recent_notes)[-self.lookback_notes :]
        if not notes:
            return []

        tokens = []
        prev_time_ms = notes[0].time_ms

        for note in notes:
            delta_ms = max(0.0, note.time_ms - prev_time_ms)
            delta_steps = int(delta_ms / 10.0)
            while delta_steps > 0:
                step = min(delta_steps, 127)
                tokens.append(self.tokenizer.time_shift_offset + step)
                delta_steps -= step

            tokens.append(self.tokenizer.note_on_offset + int(note.pitch))
            velocity_bin = max(1, min(63, int(note.velocity) // 2))
            tokens.append(self.tokenizer.velocity_offset + velocity_bin)
            tokens.append(self.tokenizer.time_shift_offset + 12)
            tokens.append(self.tokenizer.note_off_offset + int(note.pitch))
            prev_time_ms = note.time_ms

            if len(tokens) >= max_conditioning_tokens:
                break
        return tokens[:max_conditioning_tokens]

    def _build_primer(
        self,
        role: str,
        tempo_bpm: float,
        max_conditioning_tokens: int,
    ) -> list:
        role_token = self.tokenizer.role_to_token(role)
        chord_token = self.tokenizer.chord_to_token(None)
        tempo_token = self.tokenizer.tempo_to_token(tempo_bpm)
        sep = self.tokenizer.control_token("COND_SEP")
        cond_tokens = self._encode_conditioning_tokens(max_conditioning_tokens)
        return [self.tokenizer.bos_token, role_token, chord_token, tempo_token, sep] + cond_tokens + [sep]

    def flush_if_ready(
        self,
        now_ms: float,
        role: str,
        tempo_bpm: float,
        max_conditioning_tokens: int,
        max_length: int,
        context_window: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Optional[GenerationRequest]:
        if not self._dirty:
            return None

        if (now_ms - self._last_input_ms) < self.chord_buffer_ms:
            return None
        if (now_ms - self._last_trigger_ms) < self.chord_buffer_ms:
            return None

        primer = self._build_primer(
            role=role,
            tempo_bpm=tempo_bpm,
            max_conditioning_tokens=max_conditioning_tokens,
        )
        if len(primer) >= max_length:
            primer = primer[: max(8, max_length // 2)]

        self._request_id += 1
        self._dirty = False
        self._last_trigger_ms = now_ms
        return GenerationRequest(
            request_id=self._request_id,
            created_at_ms=now_ms,
            role=role,
            tempo_bpm=tempo_bpm,
            primer_tokens=primer,
            max_length=max_length,
            context_window=context_window,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

