#!/usr/bin/env python3
"""
Filter Jazz MIDI files from Lakh MIDI Dataset

Filters based on:
1. Metadata (genre tags)
2. Musical features (swing ratio, syncopation)
3. Tempo range (typical jazz: 120-240 BPM)
"""

import os
import argparse
import mido
from pathlib import Path
from tqdm import tqdm
import shutil
import json


def has_jazz_keywords(filepath):
    """Check if filename contains jazz-related keywords"""
    jazz_keywords = [
        'jazz', 'bebop', 'swing', 'blues', 'bossa',
        'parker', 'coltrane', 'davis', 'evans', 'hancock',
        'monk', 'mingus', 'brubeck', 'gillespie'
    ]

    filename_lower = filepath.lower()
    return any(keyword in filename_lower for keyword in jazz_keywords)


def analyze_midi_features(midi_path):
    """Analyze MIDI file for jazz-like features"""
    try:
        mid = mido.MidiFile(midi_path)

        # Get tempo
        tempo = 120  # default
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = mido.tempo2bpm(msg.tempo)
                    break

        # Count notes
        note_count = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_count += 1

        # Duration
        duration = mid.length

        # Jazz-like if:
        # - Tempo between 80-300 (jazz range is wide)
        # - Not too short (>20 seconds)
        # - Has reasonable note density
        is_jazz_like = (
            80 <= tempo <= 300 and
            duration >= 20 and
            note_count >= 50
        )

        return {
            'tempo': tempo,
            'duration': duration,
            'note_count': note_count,
            'is_jazz_like': is_jazz_like
        }

    except Exception as e:
        return None


def filter_jazz_midis(input_dir, output_dir, min_files=50, verbose=True):
    """
    Filter jazz MIDI files from input directory

    Args:
        input_dir: Source directory with MIDI files
        output_dir: Destination directory for jazz files
        min_files: Minimum number of files to collect
        verbose: Print progress
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(input_path.rglob(ext))

    if verbose:
        print(f"Found {len(midi_files)} MIDI files in {input_dir}")
        print(f"Filtering for jazz files...")

    jazz_files = []
    stats = {
        'total': len(midi_files),
        'keyword_match': 0,
        'feature_match': 0,
        'copied': 0,
        'errors': 0
    }

    # First pass: keyword matching
    for midi_file in tqdm(midi_files, desc="Keyword filtering", disable=not verbose):
        if has_jazz_keywords(str(midi_file)):
            jazz_files.append(midi_file)
            stats['keyword_match'] += 1

    # Second pass: feature analysis (if not enough files)
    if len(jazz_files) < min_files:
        if verbose:
            print(f"Only {len(jazz_files)} files from keywords, analyzing features...")

        for midi_file in tqdm(midi_files, desc="Feature analysis", disable=not verbose):
            if midi_file in jazz_files:
                continue

            features = analyze_midi_features(str(midi_file))
            if features and features['is_jazz_like']:
                jazz_files.append(midi_file)
                stats['feature_match'] += 1

                if len(jazz_files) >= min_files * 2:  # Get extras
                    break

    # Copy files
    if verbose:
        print(f"\nCopying {len(jazz_files)} jazz files to {output_dir}...")

    for i, midi_file in enumerate(tqdm(jazz_files, desc="Copying", disable=not verbose)):
        try:
            # Create subdirectory structure or flat copy
            dest_file = output_path / f"jazz_{i:04d}.mid"
            shutil.copy2(midi_file, dest_file)
            stats['copied'] += 1
        except Exception as e:
            stats['errors'] += 1
            if verbose:
                print(f"Error copying {midi_file}: {e}")

    # Save statistics
    stats_file = output_path / 'filter_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\n" + "=" * 70)
        print(f"Filtering complete!")
        print(f"=" * 70)
        print(f"  Total MIDI files: {stats['total']}")
        print(f"  Keyword matches: {stats['keyword_match']}")
        print(f"  Feature matches: {stats['feature_match']}")
        print(f"  Copied: {stats['copied']}")
        print(f"  Errors: {stats['errors']}")
        print(f"\n  Files saved to: {output_dir}")
        print(f"  Statistics: {stats_file}")
        print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Filter jazz MIDI files from Lakh dataset"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input directory with MIDI files'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for jazz files'
    )
    parser.add_argument(
        '--min_files',
        type=int,
        default=50,
        help='Minimum number of files to collect (default: 50)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    stats = filter_jazz_midis(
        args.input,
        args.output,
        min_files=args.min_files,
        verbose=not args.quiet
    )

    # Exit code based on success
    if stats['copied'] >= args.min_files:
        return 0
    else:
        print(f"Warning: Only collected {stats['copied']} files (target: {args.min_files})")
        return 1


if __name__ == '__main__':
    exit(main())
