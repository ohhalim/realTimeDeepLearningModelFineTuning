import heapq
import threading
import time
from queue import Empty, Queue
from typing import Dict, List, Tuple

import mido

from models.midi_dataset import MIDITokenizer
from realtime.types import GenerationResult, ScheduledMidiEvent


class MIDIScheduler(threading.Thread):
    """
    Schedules generated MIDI events to output device.
    """

    def __init__(
        self,
        clock_now_ms,
        midi_send_fn,
        result_queue: Queue,
        target_horizon_bars: float = 1.0,
        min_output_horizon_ms: float = 250.0,
        min_output_notes: int = 8,
        dead_air_threshold_ms: float = 180.0,
        fallback_enabled: bool = True,
        channel: int = 0,
    ):
        super().__init__(name="midi-scheduler", daemon=True)
        self.clock_now_ms = clock_now_ms
        self.midi_send_fn = midi_send_fn
        self.result_queue = result_queue

        self.target_horizon_bars = float(target_horizon_bars)
        self.min_output_horizon_ms = float(min_output_horizon_ms)
        self.min_output_notes = int(min_output_notes)
        self.dead_air_threshold_ms = float(dead_air_threshold_ms)
        self.fallback_enabled = fallback_enabled
        self.channel = int(channel)

        self.stop_event = threading.Event()
        self.tokenizer = MIDITokenizer()
        self._event_heap: List[ScheduledMidiEvent] = []
        self._last_send_ms = 0.0
        self._last_fallback_ms = -1e9

    def stop(self) -> None:
        self.stop_event.set()

    @staticmethod
    def _bar_ms(tempo_bpm: float) -> float:
        bpm = max(30.0, float(tempo_bpm))
        beat_ms = 60000.0 / bpm
        return beat_ms * 4.0

    def _decode_tokens_to_note_events(self, tokens: List[int]) -> Tuple[List[Tuple[float, str, int, int]], float]:
        current_time_sec = 0.0
        active_notes: Dict[int, Tuple[float, int]] = {}
        events: List[Tuple[float, str, int, int]] = []
        last_velocity = 80

        for token in tokens:
            if token in (
                self.tokenizer.pad_token,
                self.tokenizer.bos_token,
                self.tokenizer.eos_token,
                self.tokenizer.mask_token,
            ) or token in self.tokenizer.control_token_ids:
                continue

            if self.tokenizer.note_on_offset <= token < self.tokenizer.note_off_offset:
                pitch = token - self.tokenizer.note_on_offset
                if pitch in active_notes:
                    start_sec, vel = active_notes[pitch]
                    events.append((start_sec * 1000.0, "note_on", pitch, vel))
                    events.append((current_time_sec * 1000.0, "note_off", pitch, 0))
                active_notes[pitch] = (current_time_sec, last_velocity)

            elif self.tokenizer.note_off_offset <= token < self.tokenizer.time_shift_offset:
                pitch = token - self.tokenizer.note_off_offset
                if pitch in active_notes:
                    start_sec, vel = active_notes[pitch]
                    events.append((start_sec * 1000.0, "note_on", pitch, vel))
                    events.append((current_time_sec * 1000.0, "note_off", pitch, 0))
                    del active_notes[pitch]

            elif self.tokenizer.time_shift_offset <= token < self.tokenizer.velocity_offset:
                current_time_sec += (token - self.tokenizer.time_shift_offset) / 100.0

            elif self.tokenizer.velocity_offset <= token < self.tokenizer.pad_token:
                last_velocity = max(1, min(127, (token - self.tokenizer.velocity_offset) * 2))

        for pitch, (start_sec, vel) in active_notes.items():
            end_sec = current_time_sec + 0.12
            events.append((start_sec * 1000.0, "note_on", pitch, vel))
            events.append((end_sec * 1000.0, "note_off", pitch, 0))

        events.sort(key=lambda x: x[0])
        duration_ms = events[-1][0] if events else 0.0
        return events, duration_ms

    def _enqueue_events(self, base_time_ms: float, note_events, request_id: int) -> None:
        for offset_ms, event_type, note, velocity in note_events:
            heapq.heappush(
                self._event_heap,
                ScheduledMidiEvent(
                    send_at_ms=base_time_ms + offset_ms,
                    message_type=event_type,
                    note=note,
                    velocity=velocity,
                    channel=self.channel,
                    source_request_id=request_id,
                ),
            )

    def _schedule_fallback_phrase(self, now_ms: float) -> None:
        phrase = [
            (0.0, "note_on", 60, 96),
            (110.0, "note_off", 60, 0),
            (120.0, "note_on", 64, 96),
            (230.0, "note_off", 64, 0),
            (240.0, "note_on", 67, 96),
            (360.0, "note_off", 67, 0),
            (380.0, "note_on", 72, 98),
            (520.0, "note_off", 72, 0),
        ]
        self._enqueue_events(now_ms + 10.0, phrase, request_id=-1)
        self._last_fallback_ms = now_ms

    def _handle_result(self, result: GenerationResult) -> None:
        now_ms = self.clock_now_ms()
        target_horizon_ms = max(
            self.target_horizon_bars * self._bar_ms(result.tempo_bpm),
            self.min_output_horizon_ms,
        )

        events, duration_ms = self._decode_tokens_to_note_events(result.continuation_tokens)
        note_on_count = sum(1 for e in events if e[1] == "note_on")

        too_short = duration_ms < self.min_output_horizon_ms
        too_few_notes = note_on_count < self.min_output_notes
        should_fallback = (result.status != "ok") or too_short or too_few_notes

        if should_fallback and self.fallback_enabled:
            self._schedule_fallback_phrase(now_ms)
            return
        if not events:
            return

        play_from_ms = now_ms + target_horizon_ms
        self._enqueue_events(play_from_ms, events, request_id=result.request_id)

    def _flush_due_events(self) -> None:
        now_ms = self.clock_now_ms()
        while self._event_heap and self._event_heap[0].send_at_ms <= now_ms:
            event = heapq.heappop(self._event_heap)
            msg = mido.Message(
                event.message_type,
                note=int(event.note),
                velocity=int(event.velocity),
                channel=int(event.channel),
            )
            self.midi_send_fn(msg)
            self._last_send_ms = now_ms

        if self.fallback_enabled:
            idle_ms = now_ms - self._last_send_ms
            if (
                idle_ms >= self.dead_air_threshold_ms
                and not self._event_heap
                and (now_ms - self._last_fallback_ms) >= self.dead_air_threshold_ms
            ):
                self._schedule_fallback_phrase(now_ms)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=0.01)
                self._handle_result(result)
            except Empty:
                pass
            self._flush_due_events()
            time.sleep(0.001)

