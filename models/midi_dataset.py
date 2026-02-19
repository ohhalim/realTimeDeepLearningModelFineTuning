#!/usr/bin/env python3
"""
MIDI dataset utilities for PyTorch Music Transformer.

Supports:
1) Raw MIDI training (`data/raw_midi`)
2) Role-conditioned training (`data/roles/...`) for real-time jam use-cases
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List

import pretty_midi
import torch
from torch.utils.data import Dataset


class MIDITokenizer:
    """
    MIDI tokenizer with lightweight control tokens for role-conditioned training.

    Event types:
    - NOTE_ON: 0-127
    - NOTE_OFF: 128-255
    - TIME_SHIFT: 256-383 (0-127 * 10ms)
    - VELOCITY: 384-447 (0-63)
    - Special: 448+
    """

    def __init__(self, vocab_size: int = 512):
        self.vocab_size = vocab_size

        self.note_on_offset = 0
        self.note_off_offset = 128
        self.time_shift_offset = 256
        self.velocity_offset = 384

        # Core special tokens (backward-compatible)
        self.pad_token = 448
        self.bos_token = 449
        self.eos_token = 450
        self.mask_token = 451

        # Stage A control tokens
        base_control = 452
        self.control_tokens: Dict[str, int] = {
            "ROLE_LEAD": base_control + 0,
            "ROLE_ACCOMPANIMENT": base_control + 1,
            "ROLE_CALL_RESPONSE": base_control + 2,
            "CHORD_UNKNOWN": base_control + 3,
            "TEMPO_SLOW": base_control + 4,
            "TEMPO_MEDIUM": base_control + 5,
            "TEMPO_FAST": base_control + 6,
            "COND_SEP": base_control + 7,
        }
        self.control_token_ids = set(self.control_tokens.values())

        max_used_token = max([self.mask_token] + list(self.control_token_ids))
        if vocab_size <= max_used_token:
            raise ValueError(
                f"vocab_size={vocab_size} is too small; requires at least {max_used_token + 1}"
            )

    def control_token(self, name: str) -> int:
        return self.control_tokens[name]

    def role_to_token(self, role: str) -> int:
        role_norm = (role or "lead").strip().lower()
        if role_norm == "lead":
            return self.control_tokens["ROLE_LEAD"]
        if role_norm == "accompaniment":
            return self.control_tokens["ROLE_ACCOMPANIMENT"]
        if role_norm == "call_response":
            return self.control_tokens["ROLE_CALL_RESPONSE"]
        return self.control_tokens["ROLE_LEAD"]

    def chord_to_token(self, _chord_progression) -> int:
        return self.control_tokens["CHORD_UNKNOWN"]

    def tempo_to_token(self, tempo_bpm) -> int:
        try:
            bpm = float(tempo_bpm)
        except (TypeError, ValueError):
            bpm = 120.0

        if bpm < 90:
            return self.control_tokens["TEMPO_SLOW"]
        if bpm < 140:
            return self.control_tokens["TEMPO_MEDIUM"]
        return self.control_tokens["TEMPO_FAST"]

    def _collect_notes(self, midi_obj: pretty_midi.PrettyMIDI) -> List[pretty_midi.Note]:
        notes: List[pretty_midi.Note] = []
        for instrument in midi_obj.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)
        notes.sort(key=lambda x: (x.start, x.pitch))
        return notes

    def encode_compact(
        self,
        midi_path: str,
        max_tokens: int = 2048,
        include_bos: bool = True,
        include_eos: bool = True,
    ) -> List[int]:
        """Encode MIDI to a compact (unpadded) token sequence."""
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as exc:
            print(f"Error encoding {midi_path}: {exc}")
            fallback = [self.bos_token] if include_bos else []
            if include_eos:
                fallback.append(self.eos_token)
            return fallback[: max_tokens if max_tokens > 0 else 1]

        tokens: List[int] = [self.bos_token] if include_bos else []
        all_notes = self._collect_notes(midi)
        current_time = 0.0

        for note in all_notes:
            time_diff = int((note.start - current_time) * 100)
            while time_diff > 0:
                shift = min(time_diff, 127)
                tokens.append(self.time_shift_offset + shift)
                time_diff -= shift

            tokens.append(self.note_on_offset + int(note.pitch))
            velocity = min(int(note.velocity) // 2, 63)
            tokens.append(self.velocity_offset + velocity)

            duration = int((note.end - note.start) * 100)
            while duration > 0:
                shift = min(duration, 127)
                tokens.append(self.time_shift_offset + shift)
                duration -= shift

            tokens.append(self.note_off_offset + int(note.pitch))
            current_time = float(note.end)

            if len(tokens) >= max_tokens - (1 if include_eos else 0):
                break

        if include_eos:
            tokens.append(self.eos_token)

        if not tokens:
            tokens = [self.bos_token, self.eos_token] if include_eos else [self.bos_token]

        return tokens[:max_tokens]

    def encode(self, midi_path: str, max_length: int = 2048) -> List[int]:
        """Encode MIDI into fixed-length sequence (padded)."""
        tokens = self.encode_compact(
            midi_path=midi_path,
            max_tokens=max_length,
            include_bos=True,
            include_eos=True,
        )
        if len(tokens) < max_length:
            tokens.extend([self.pad_token] * (max_length - len(tokens)))
        else:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, tokens: List[int], output_path: str, tempo: int = 120) -> None:
        """Decode token sequence into MIDI."""
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)

        current_time = 0.0
        active_notes = {}

        for token in tokens:
            if token in (
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.mask_token,
            ):
                continue
            if token in self.control_token_ids:
                continue

            if self.note_on_offset <= token < self.note_off_offset:
                pitch = token - self.note_on_offset
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    piano.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=current_time,
                        )
                    )
                active_notes[pitch] = (current_time, 80)

            elif self.note_off_offset <= token < self.time_shift_offset:
                pitch = token - self.note_off_offset
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    piano.notes.append(
                        pretty_midi.Note(
                            velocity=velocity,
                            pitch=pitch,
                            start=start_time,
                            end=current_time,
                        )
                    )
                    del active_notes[pitch]

            elif self.time_shift_offset <= token < self.velocity_offset:
                time_shift = (token - self.time_shift_offset) / 100.0
                current_time += time_shift

            elif self.velocity_offset <= token < self.pad_token:
                velocity = (token - self.velocity_offset) * 2
                if active_notes:
                    last_pitch = list(active_notes.keys())[-1]
                    start_time, _ = active_notes[last_pitch]
                    active_notes[last_pitch] = (start_time, velocity)

        for pitch, (start_time, velocity) in active_notes.items():
            piano.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=current_time + 0.5,
                )
            )

        midi.instruments.append(piano)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        midi.write(output_path)


class MIDIDataset(Dataset):
    """PyTorch dataset for raw MIDI and role-conditioned MIDI."""

    def __init__(
        self,
        midi_dir: str,
        max_length: int = 2048,
        augment: bool = True,
        use_role_dataset: bool = False,
        max_conditioning_tokens: int = 256,
    ):
        self.midi_dir = midi_dir
        self.max_length = max_length
        self.augment = augment
        self.use_role_dataset = use_role_dataset
        self.max_conditioning_tokens = max_conditioning_tokens

        self.tokenizer = MIDITokenizer()
        self.encoded_files: List[List[int]] = []

        if self.use_role_dataset:
            self._load_role_dataset()
        else:
            self._load_raw_dataset()

    def _load_raw_dataset(self) -> None:
        midi_files: List[Path] = []
        for ext in ("*.mid", "*.midi"):
            midi_files.extend(Path(self.midi_dir).rglob(ext))
        midi_files = sorted(midi_files)

        if not midi_files:
            raise ValueError(f"No MIDI files found in {self.midi_dir}")

        print(f"Found {len(midi_files)} MIDI files (raw mode)")
        print("Encoding MIDI files...")
        for midi_file in midi_files:
            tokens = self.tokenizer.encode(str(midi_file), self.max_length)
            self.encoded_files.append(tokens)

    def _load_role_dataset(self) -> None:
        role_root = Path(self.midi_dir)
        target_files = sorted(role_root.rglob("target.mid"))
        if not target_files:
            raise ValueError(
                f"No role samples found in {self.midi_dir} (expected */target.mid files)"
            )

        print(f"Found {len(target_files)} role samples (role mode)")
        print("Encoding role-conditioned samples...")
        for target_path in target_files:
            conditioning_path = target_path.with_name("conditioning.mid")
            meta_path = target_path.with_name("meta.json")
            if not conditioning_path.exists():
                print(f"Skip sample without conditioning.mid: {target_path.parent}")
                continue

            meta = self._read_meta(meta_path)
            tokens = self._encode_role_sample(
                conditioning_path=conditioning_path,
                target_path=target_path,
                meta=meta,
            )
            self.encoded_files.append(tokens)

        if not self.encoded_files:
            raise ValueError(
                f"No valid role samples encoded from {self.midi_dir}. "
                "Check conditioning.mid/target.mid/meta.json files."
            )

    def _read_meta(self, meta_path: Path) -> Dict:
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: failed to read meta {meta_path}: {exc}")
            return {}

    def _encode_role_sample(
        self,
        conditioning_path: Path,
        target_path: Path,
        meta: Dict,
    ) -> List[int]:
        role_token = self.tokenizer.role_to_token(meta.get("role", "lead"))
        chord_token = self.tokenizer.chord_to_token(meta.get("chord_progression"))
        tempo_token = self.tokenizer.tempo_to_token(meta.get("tempo", 120))
        sep_token = self.tokenizer.control_token("COND_SEP")

        conditioning_tokens = self.tokenizer.encode_compact(
            str(conditioning_path),
            max_tokens=self.max_conditioning_tokens,
            include_bos=False,
            include_eos=False,
        )
        target_tokens = self.tokenizer.encode_compact(
            str(target_path),
            max_tokens=self.max_length,
            include_bos=False,
            include_eos=True,
        )

        tokens = [
            self.tokenizer.bos_token,
            role_token,
            chord_token,
            tempo_token,
            sep_token,
        ]
        tokens.extend(conditioning_tokens)
        tokens.append(sep_token)
        tokens.extend(target_tokens)

        if self.tokenizer.eos_token not in tokens:
            tokens.append(self.tokenizer.eos_token)

        if len(tokens) < self.max_length:
            tokens.extend([self.tokenizer.pad_token] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[: self.max_length]
            if tokens[-1] not in (self.tokenizer.eos_token, self.tokenizer.pad_token):
                tokens[-1] = self.tokenizer.eos_token
        return tokens

    def __len__(self) -> int:
        return len(self.encoded_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.encoded_files[idx].copy()

        if self.augment and random.random() > 0.5:
            tokens = self._transpose(tokens, random.randint(-5, 5))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    def _transpose(self, tokens: List[int], semitones: int) -> List[int]:
        transposed: List[int] = []
        for token in tokens:
            if self.tokenizer.note_on_offset <= token < self.tokenizer.note_off_offset:
                new_pitch = token - self.tokenizer.note_on_offset + semitones
                if 0 <= new_pitch <= 127:
                    transposed.append(self.tokenizer.note_on_offset + new_pitch)
                else:
                    transposed.append(token)
            elif self.tokenizer.note_off_offset <= token < self.tokenizer.time_shift_offset:
                new_pitch = token - self.tokenizer.note_off_offset + semitones
                if 0 <= new_pitch <= 127:
                    transposed.append(self.tokenizer.note_off_offset + new_pitch)
                else:
                    transposed.append(token)
            else:
                transposed.append(token)
        return transposed


def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Create causal attention mask.

    Returns shape: (1, 1, seq_len, seq_len)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    tokenizer = MIDITokenizer()
    print("Testing MIDI tokenizer...")

    try:
        dataset = MIDIDataset("data/raw_midi", max_length=512, use_role_dataset=False)
        print(f"Raw dataset size: {len(dataset)}")
        sample = dataset[0]
        print(f"Input shape: {sample['input_ids'].shape}")
        print(f"Label shape: {sample['labels'].shape}")
    except ValueError as exc:
        print(f"Note: {exc}")
