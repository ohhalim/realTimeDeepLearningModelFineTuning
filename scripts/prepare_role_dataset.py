#!/usr/bin/env python3
"""
Prepare role-based MIDI dataset for real-time jam training.

Output structure:
  data/roles/<role>/<sample_id>/
    - conditioning.mid
    - target.mid
    - meta.json

MVP design:
  - Start with a single role (`lead`) and a simple conditioning signal.
  - Use lower-register note extraction as a lightweight chord/bass proxy.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Iterable, List

import pretty_midi


VALID_EXTENSIONS = (".mid", ".midi")


def find_midi_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            files.append(path)
    return files


def clone_time_signatures(src: pretty_midi.PrettyMIDI, dst: pretty_midi.PrettyMIDI) -> None:
    for ts in src.time_signature_changes:
        dst.time_signature_changes.append(
            pretty_midi.TimeSignature(ts.numerator, ts.denominator, ts.time)
        )


def clone_key_signatures(src: pretty_midi.PrettyMIDI, dst: pretty_midi.PrettyMIDI) -> None:
    for ks in src.key_signature_changes:
        dst.key_signature_changes.append(
            pretty_midi.KeySignature(ks.key_number, ks.time)
        )


def all_non_drum_notes(midi_obj: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
    notes: List[pretty_midi.Note] = []
    for inst in midi_obj.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def clone_with_selected_notes(
    src_midi: pretty_midi.PrettyMIDI,
    selected_notes: Iterable[pretty_midi.Note],
) -> pretty_midi.PrettyMIDI:
    tempo = 120.0
    try:
        tempo = float(src_midi.estimate_tempo()) or 120.0
    except Exception:
        tempo = 120.0

    dst = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    clone_time_signatures(src_midi, dst)
    clone_key_signatures(src_midi, dst)

    program = 0
    for inst in src_midi.instruments:
        if not inst.is_drum:
            program = inst.program
            break

    out_inst = pretty_midi.Instrument(program=program, is_drum=False, name="piano")
    for note in selected_notes:
        out_inst.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(note.start),
                end=float(note.end),
            )
        )

    dst.instruments.append(out_inst)
    return dst


def transpose_midi(src_midi: pretty_midi.PrettyMIDI, semitones: int) -> pretty_midi.PrettyMIDI:
    dst = copy.deepcopy(src_midi)
    for inst in dst.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            new_pitch = max(0, min(127, note.pitch + semitones))
            note.pitch = int(new_pitch)
    return dst


def build_conditioning_midi(
    target_midi: pretty_midi.PrettyMIDI,
    conditioning_mode: str,
) -> pretty_midi.PrettyMIDI:
    notes = all_non_drum_notes(target_midi)
    if not notes:
        return clone_with_selected_notes(target_midi, [])

    if conditioning_mode == "full":
        selected = notes
    else:
        pitches = sorted(note.pitch for note in notes)
        cutoff = pitches[int(0.4 * (len(pitches) - 1))]
        selected = [n for n in notes if n.pitch <= cutoff]
        if len(selected) < max(8, int(0.1 * len(notes))):
            selected = notes[::2] if len(notes) > 1 else notes

    return clone_with_selected_notes(target_midi, selected)


def safe_estimate_tempo(midi_obj: pretty_midi.PrettyMIDI) -> float:
    try:
        tempo = float(midi_obj.estimate_tempo())
        return tempo if tempo > 0 else 120.0
    except Exception:
        return 120.0


def first_time_signature(midi_obj: pretty_midi.PrettyMIDI) -> str:
    if midi_obj.time_signature_changes:
        ts = midi_obj.time_signature_changes[0]
        return f"{ts.numerator}/{ts.denominator}"
    return "4/4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare role-based MIDI dataset")
    parser.add_argument("--input_dir", type=str, default="data/raw_midi")
    parser.add_argument("--output_dir", type=str, default="data/roles")
    parser.add_argument(
        "--role",
        type=str,
        default="lead",
        choices=["lead", "accompaniment", "call_response"],
    )
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="lower_register",
        choices=["lower_register", "full"],
        help="How to build conditioning.mid",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="regular_interval",
        help="Interaction strategy metadata label",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="brad_mehldau",
        help="Style metadata label",
    )
    parser.add_argument(
        "--transpose_all_keys",
        action="store_true",
        help="Create 12-key augmented samples (-5..+6 semitones)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output role directory before writing",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    role_dir = output_root / args.role

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    midi_files = find_midi_files(input_dir)
    if args.limit is not None:
        midi_files = midi_files[: max(0, args.limit)]

    if not midi_files:
        raise ValueError(f"No MIDI files found under: {input_dir}")

    if args.overwrite and role_dir.exists():
        shutil.rmtree(role_dir)
    role_dir.mkdir(parents=True, exist_ok=True)

    semitone_values = list(range(-5, 7)) if args.transpose_all_keys else [0]

    sample_index = 0
    written_count = 0
    skipped_count = 0

    print(f"Found {len(midi_files)} MIDI files")
    print(f"Role: {args.role}")
    print(f"Conditioning mode: {args.conditioning_mode}")
    print(f"Augmentation keys: {len(semitone_values)}")
    print(f"Output: {role_dir}")

    for midi_path in midi_files:
        try:
            original = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as exc:
            print(f"[SKIP] Failed to read {midi_path}: {exc}")
            skipped_count += 1
            continue

        for semitones in semitone_values:
            sample_index += 1
            sample_id = f"{sample_index:05d}_{midi_path.stem}_t{semitones:+d}"
            sample_dir = role_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            target_midi = transpose_midi(original, semitones)
            conditioning_midi = build_conditioning_midi(
                target_midi=target_midi,
                conditioning_mode=args.conditioning_mode,
            )

            target_path = sample_dir / "target.mid"
            conditioning_path = sample_dir / "conditioning.mid"
            meta_path = sample_dir / "meta.json"

            target_midi.write(str(target_path))
            conditioning_midi.write(str(conditioning_path))

            target_note_count = len(all_non_drum_notes(target_midi))
            conditioning_note_count = len(all_non_drum_notes(conditioning_midi))

            meta = {
                "sample_id": sample_id,
                "role": args.role,
                "style": args.style,
                "strategy": args.strategy,
                "key": "unknown",
                "chord_progression": [],
                "tempo": round(safe_estimate_tempo(target_midi), 2),
                "time_signature": first_time_signature(target_midi),
                "conditioning_mode": args.conditioning_mode,
                "source_file": str(midi_path),
                "transpose_semitones": semitones,
                "target_note_count": target_note_count,
                "conditioning_note_count": conditioning_note_count,
            }
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            written_count += 1

    print(
        f"Done. written={written_count}, skipped_files={skipped_count}, role_dir={role_dir}"
    )


if __name__ == "__main__":
    main()
