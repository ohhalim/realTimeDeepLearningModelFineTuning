#!/usr/bin/env python3
"""
Stage A realtime runtime entrypoint.

Keyboard MIDI input -> prompt builder -> generation worker -> MIDI scheduler output.
"""

import argparse
import signal
import threading
import time
from queue import Queue

import yaml

from models.midi_dataset import MIDITokenizer
from realtime.clock import GlobalClock
from realtime.generation_worker import GenerationWorker
from realtime.midi_input import MIDIIO
from realtime.prompt_builder import PromptBuilder
from realtime.scheduler import MIDIScheduler


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Run Stage A realtime MIDI runtime")
    parser.add_argument(
        "--inference_config",
        type=str,
        default="configs/inference/realtime_stage_a.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override model checkpoint path",
    )
    parser.add_argument("--input_port", type=str, default=None)
    parser.add_argument("--output_port", type=str, default=None)
    parser.add_argument("--list_ports", action="store_true")
    parser.add_argument("--role", type=str, default=None)
    parser.add_argument("--tempo_bpm", type=float, default=128.0)
    parser.add_argument("--passthrough", action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        print("Input Ports:")
        for p in MIDIIO.list_input_ports():
            print(f"  - {p}")
        print("Output Ports:")
        for p in MIDIIO.list_output_ports():
            print(f"  - {p}")
        return

    cfg = load_yaml(args.inference_config)
    model_cfg = cfg.get("model", {})
    generation_cfg = cfg.get("generation", {})
    scheduler_cfg = cfg.get("scheduler", {})
    fallback_cfg = cfg.get("fallback", {})

    checkpoint_path = args.checkpoint or model_cfg.get("checkpoint")
    role = args.role or model_cfg.get("role", "lead")
    if not checkpoint_path:
        raise ValueError("checkpoint path is required (config.model.checkpoint or --checkpoint)")

    max_length = int(generation_cfg.get("max_length", 512))
    context_window = int(generation_cfg.get("context_window", 256))
    temperature = float(generation_cfg.get("temperature", 0.9))
    top_k = int(generation_cfg.get("top_k", 40))
    top_p = float(generation_cfg.get("top_p", 0.9))
    max_conditioning_tokens = int(generation_cfg.get("max_conditioning_tokens", 256))

    chord_buffer_ms = float(scheduler_cfg.get("chord_buffer_ms", 100.0))
    max_inference_step_ms = float(scheduler_cfg.get("max_inference_step_ms", 60.0))

    request_queue: Queue = Queue(maxsize=8)
    result_queue: Queue = Queue(maxsize=16)

    clock = GlobalClock()
    clock.start()

    tokenizer = MIDITokenizer()
    prompt_builder = PromptBuilder(
        tokenizer=tokenizer,
        chord_buffer_ms=chord_buffer_ms,
        lookback_notes=60,
    )

    midi = MIDIIO(
        input_port_name=args.input_port,
        output_port_name=args.output_port,
        passthrough=args.passthrough,
    )

    scheduler = MIDIScheduler(
        clock_now_ms=clock.now_ms,
        midi_send_fn=midi.send,
        result_queue=result_queue,
        target_horizon_bars=float(scheduler_cfg.get("target_horizon_bars", 1.0)),
        min_output_horizon_ms=float(scheduler_cfg.get("min_output_horizon_ms", 250.0)),
        min_output_notes=int(scheduler_cfg.get("min_output_notes", 8)),
        dead_air_threshold_ms=float(scheduler_cfg.get("dead_air_threshold_ms", 180.0)),
        fallback_enabled=bool(fallback_cfg.get("enabled", True)),
        channel=0,
    )

    worker = GenerationWorker(
        request_queue=request_queue,
        result_queue=result_queue,
        checkpoint_path=checkpoint_path,
        max_inference_step_ms=max_inference_step_ms,
        cache_enabled=True,
    )

    stop_event = threading.Event()

    def on_midi_message(msg, timestamp_ms):
        prompt_builder.ingest_midi(msg, timestamp_ms)

    midi.set_callback(on_midi_message)

    def shutdown():
        stop_event.set()
        try:
            midi.stop()
        except Exception:
            pass
        worker.stop()
        scheduler.stop()
        clock.stop()

    def handle_signal(_sig, _frame):
        shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    midi.start(clock.now_ms)
    scheduler.start()
    worker.start()

    print("=" * 60)
    print("Realtime runtime started")
    print(f"Input port:  {midi.input_port_name}")
    print(f"Output port: {midi.output_port_name}")
    print(f"Role: {role}, Tempo: {args.tempo_bpm}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    try:
        while not stop_event.is_set():
            now_ms = clock.now_ms()
            req = prompt_builder.flush_if_ready(
                now_ms=now_ms,
                role=role,
                tempo_bpm=args.tempo_bpm,
                max_conditioning_tokens=max_conditioning_tokens,
                max_length=max_length,
                context_window=context_window,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if req is not None:
                try:
                    request_queue.put_nowait(req)
                except Exception:
                    # Queue is full; skip stale request to preserve low latency.
                    pass
            time.sleep(0.005)
    finally:
        shutdown()
        print("Realtime runtime stopped")


if __name__ == "__main__":
    main()

