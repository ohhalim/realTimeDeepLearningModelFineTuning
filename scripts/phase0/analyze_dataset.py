#!/usr/bin/env python3
"""
Analyze MIDI dataset statistics

Provides:
- Total duration
- Average song length
- Tempo distribution
- Note range
- Polyphony statistics
"""

import argparse
import mido
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict


def analyze_midi_file(midi_path):
    """Extract statistics from a single MIDI file"""
    try:
        mid = mido.MidiFile(midi_path)

        stats = {
            'duration': mid.length,
            'tempo': 120,  # default
            'time_signature': (4, 4),
            'note_count': 0,
            'min_pitch': 127,
            'max_pitch': 0,
            'max_polyphony': 0,
            'total_time_secs': mid.length
        }

        # Extract tempo and time signature
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    stats['tempo'] = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    stats['time_signature'] = (msg.numerator, msg.denominator)

        # Analyze notes
        active_notes = set()
        max_poly = 0

        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on':
                    if msg.velocity > 0:
                        stats['note_count'] += 1
                        stats['min_pitch'] = min(stats['min_pitch'], msg.note)
                        stats['max_pitch'] = max(stats['max_pitch'], msg.note)
                        active_notes.add(msg.note)
                        max_poly = max(max_poly, len(active_notes))
                    else:
                        active_notes.discard(msg.note)
                elif msg.type == 'note_off':
                    active_notes.discard(msg.note)

        stats['max_polyphony'] = max_poly

        return stats

    except Exception as e:
        return None


def analyze_dataset(input_dir, output_file=None, verbose=True):
    """
    Analyze all MIDI files in directory

    Args:
        input_dir: Directory with MIDI files
        output_file: JSON file to save stats (optional)
        verbose: Print progress

    Returns:
        Statistics dictionary
    """
    input_path = Path(input_dir)

    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(input_path.glob(ext))

    if verbose:
        print(f"Analyzing {len(midi_files)} MIDI files...")

    all_stats = []
    errors = 0

    for midi_file in tqdm(midi_files, desc="Analyzing", disable=not verbose):
        stats = analyze_midi_file(midi_file)
        if stats:
            all_stats.append(stats)
        else:
            errors += 1

    if not all_stats:
        print("Error: No valid MIDI files found!")
        return None

    # Aggregate statistics
    durations = [s['duration'] for s in all_stats]
    tempos = [s['tempo'] for s in all_stats]
    note_counts = [s['note_count'] for s in all_stats]
    min_pitches = [s['min_pitch'] for s in all_stats if s['min_pitch'] < 127]
    max_pitches = [s['max_pitch'] for s in all_stats if s['max_pitch'] > 0]
    polyphonies = [s['max_polyphony'] for s in all_stats]

    # Note names
    def note_name(pitch):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = pitch // 12 - 1
        return f"{notes[pitch % 12]}{octave}"

    dataset_stats = {
        'files': {
            'total': len(midi_files),
            'valid': len(all_stats),
            'errors': errors,
        },
        'duration': {
            'total_seconds': sum(durations),
            'total_hours': sum(durations) / 3600,
            'average_seconds': np.mean(durations),
            'median_seconds': np.median(durations),
            'min_seconds': min(durations),
            'max_seconds': max(durations),
        },
        'tempo': {
            'average_bpm': np.mean(tempos),
            'median_bpm': np.median(tempos),
            'min_bpm': min(tempos),
            'max_bpm': max(tempos),
        },
        'notes': {
            'total_notes': sum(note_counts),
            'average_per_file': np.mean(note_counts),
            'min_pitch': min(min_pitches) if min_pitches else None,
            'max_pitch': max(max_pitches) if max_pitches else None,
            'min_pitch_name': note_name(min(min_pitches)) if min_pitches else None,
            'max_pitch_name': note_name(max(max_pitches)) if max_pitches else None,
        },
        'polyphony': {
            'average': np.mean(polyphonies),
            'median': np.median(polyphonies),
            'max': max(polyphonies),
        }
    }

    # Print results
    if verbose:
        print(f"\n" + "=" * 70)
        print(f"Dataset Statistics")
        print(f"=" * 70)

        print(f"\n📁 Files")
        print(f"  Total: {dataset_stats['files']['total']}")
        print(f"  Valid: {dataset_stats['files']['valid']}")
        print(f"  Errors: {dataset_stats['files']['errors']}")

        print(f"\n⏱️  Duration")
        hours = int(dataset_stats['duration']['total_hours'])
        minutes = int((dataset_stats['duration']['total_hours'] - hours) * 60)
        print(f"  Total: {hours}h {minutes}m")
        print(f"  Average per file: {dataset_stats['duration']['average_seconds']:.1f}s")
        print(f"  Range: {dataset_stats['duration']['min_seconds']:.1f}s - {dataset_stats['duration']['max_seconds']:.1f}s")

        print(f"\n🎵 Tempo")
        print(f"  Average: {dataset_stats['tempo']['average_bpm']:.1f} BPM")
        print(f"  Range: {dataset_stats['tempo']['min_bpm']:.1f} - {dataset_stats['tempo']['max_bpm']:.1f} BPM")

        print(f"\n🎹 Notes")
        print(f"  Total notes: {dataset_stats['notes']['total_notes']:,}")
        print(f"  Average per file: {dataset_stats['notes']['average_per_file']:.0f}")
        print(f"  Pitch range: {dataset_stats['notes']['min_pitch_name']} - {dataset_stats['notes']['max_pitch_name']}")

        print(f"\n🎼 Polyphony")
        print(f"  Average voices: {dataset_stats['polyphony']['average']:.1f}")
        print(f"  Max voices: {dataset_stats['polyphony']['max']}")

        print("=" * 70)

    # Save to JSON
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(dataset_stats, f, indent=2)

        if verbose:
            print(f"\n💾 Statistics saved to: {output_file}")

    return dataset_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MIDI dataset statistics"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Directory with MIDI files'
    )
    parser.add_argument(
        '--output',
        help='JSON file to save statistics (optional)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    stats = analyze_dataset(
        args.input,
        output_file=args.output,
        verbose=not args.quiet
    )

    return 0 if stats else 1


if __name__ == '__main__':
    exit(main())
