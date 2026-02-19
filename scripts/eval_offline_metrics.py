#!/usr/bin/env python3
"""
Offline MIDI generation quality metrics for fast iteration.

Metrics (per file):
- ttfn_ms
- note_density_notes_per_sec
- pitch_mean/std/min/max
- repetition_ratio (pitch n-gram based)
- dead_air_event_count
- dead_air_ratio
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import pretty_midi


VALID_EXTENSIONS = {".mid", ".midi"}


def list_midi_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTENSIONS:
            return [input_path]
        return []
    files = [
        p
        for p in sorted(input_path.rglob("*"))
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]
    return files


def collect_notes(midi_path: Path) -> List[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: List[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def repetition_ratio(pitches: Sequence[int], ngram: int) -> float:
    if ngram <= 0 or len(pitches) < ngram:
        return 0.0
    seqs: List[Tuple[int, ...]] = [
        tuple(pitches[i : i + ngram]) for i in range(len(pitches) - ngram + 1)
    ]
    if not seqs:
        return 0.0
    counts: Dict[Tuple[int, ...], int] = {}
    for s in seqs:
        counts[s] = counts.get(s, 0) + 1
    repeated_occurrences = sum(c for c in counts.values() if c > 1)
    return repeated_occurrences / len(seqs)


def compute_metrics(
    notes: Sequence[pretty_midi.Note],
    dead_air_threshold_ms: float,
    ngram: int,
) -> Dict[str, float]:
    if not notes:
        return {
            "note_count": 0,
            "duration_sec": 0.0,
            "ttfn_ms": 0.0,
            "note_density_notes_per_sec": 0.0,
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "repetition_ratio": 0.0,
            "dead_air_event_count": 0,
            "dead_air_ratio": 1.0,
            "max_gap_ms": 0.0,
        }

    first_start = min(n.start for n in notes)
    end_time = max(n.end for n in notes)
    duration = max(end_time - first_start, 1e-6)
    pitches = [int(n.pitch) for n in notes]

    gaps_sec: List[float] = []
    prev_end = notes[0].end
    for note in notes[1:]:
        gap = max(0.0, note.start - prev_end)
        gaps_sec.append(gap)
        prev_end = max(prev_end, note.end)

    threshold_sec = dead_air_threshold_ms / 1000.0
    dead_air_gaps = [g for g in gaps_sec if g >= threshold_sec]
    dead_air_total = sum(dead_air_gaps)
    dead_air_ratio_value = min(dead_air_total / duration, 1.0)

    pitch_avg = sum(pitches) / len(pitches)
    pitch_var = sum((p - pitch_avg) ** 2 for p in pitches) / len(pitches)

    return {
        "note_count": len(notes),
        "duration_sec": round(duration, 4),
        "ttfn_ms": round(first_start * 1000.0, 2),
        "note_density_notes_per_sec": round(len(notes) / duration, 4),
        "pitch_mean": round(pitch_avg, 3),
        "pitch_std": round(pitch_var ** 0.5, 3),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "repetition_ratio": round(repetition_ratio(pitches, ngram), 4),
        "dead_air_event_count": len(dead_air_gaps),
        "dead_air_ratio": round(dead_air_ratio_value, 4),
        "max_gap_ms": round((max(gaps_sec) if gaps_sec else 0.0) * 1000.0, 2),
    }


def aggregate_metrics(per_file: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not per_file:
        return {}

    numeric_keys = [
        "note_count",
        "duration_sec",
        "ttfn_ms",
        "note_density_notes_per_sec",
        "pitch_mean",
        "pitch_std",
        "repetition_ratio",
        "dead_air_event_count",
        "dead_air_ratio",
        "max_gap_ms",
    ]
    agg: Dict[str, float] = {}
    for key in numeric_keys:
        values = [float(v[key]) for v in per_file.values() if key in v]
        if values:
            agg[f"mean_{key}"] = round(mean(values), 4)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline MIDI quality evaluation")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="MIDI file or directory",
    )
    parser.add_argument(
        "--dead_air_threshold_ms",
        type=float,
        default=180.0,
        help="Gap >= this threshold is counted as dead-air event",
    )
    parser.add_argument(
        "--ngram",
        type=int,
        default=4,
        help="Pitch n-gram size for repetition ratio",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/offline_eval_report.json",
        help="JSON report output path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    files = list_midi_files(input_path)
    if not files:
        raise ValueError(f"No MIDI files found at: {input_path}")

    per_file: Dict[str, Dict[str, float]] = {}
    failed: List[Dict[str, str]] = []

    for midi_file in files:
        try:
            notes = collect_notes(midi_file)
            metrics = compute_metrics(notes, args.dead_air_threshold_ms, args.ngram)
            per_file[str(midi_file)] = metrics
        except Exception as exc:
            failed.append({"file": str(midi_file), "error": str(exc)})

    report = {
        "input": str(input_path),
        "dead_air_threshold_ms": args.dead_air_threshold_ms,
        "ngram": args.ngram,
        "file_count": len(files),
        "success_count": len(per_file),
        "failed_count": len(failed),
        "aggregate": aggregate_metrics(per_file),
        "per_file": per_file,
        "failed": failed,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved report: {output_path}")
    print(f"Processed: {len(per_file)}/{len(files)} files")
    if failed:
        print(f"Failed: {len(failed)} files")


if __name__ == "__main__":
    main()
