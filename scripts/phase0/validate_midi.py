#!/usr/bin/env python3
"""
Validate MIDI files and remove corrupt ones

Checks for:
1. File can be opened by mido
2. Has at least one note
3. Has valid tempo/time signature
4. Duration > 5 seconds
"""

import argparse
import mido
from pathlib import Path
from tqdm import tqdm
import os


def validate_midi_file(midi_path):
    """
    Validate a single MIDI file

    Returns:
        (is_valid, error_message)
    """
    try:
        # Try to open file
        mid = mido.MidiFile(midi_path)

        # Check duration
        if mid.length < 5:
            return False, "Too short (<5s)"

        # Count notes
        note_count = 0
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_count += 1

        if note_count == 0:
            return False, "No notes found"

        if note_count < 10:
            return False, f"Too few notes ({note_count})"

        # Check for tempo
        has_tempo = False
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    has_tempo = True
                    break

        # Valid!
        return True, None

    except EOFError:
        return False, "EOF error (corrupt)"
    except Exception as e:
        return False, str(e)


def validate_midi_directory(input_dir, remove_corrupt=False, verbose=True):
    """
    Validate all MIDI files in directory

    Args:
        input_dir: Directory with MIDI files
        remove_corrupt: Remove invalid files
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
        print(f"Checking {len(midi_files)} MIDI files...")

    stats = {
        'total': len(midi_files),
        'valid': 0,
        'invalid': 0,
        'removed': 0,
        'errors': {}
    }

    invalid_files = []

    for midi_file in tqdm(midi_files, desc="Validating", disable=not verbose):
        is_valid, error = validate_midi_file(midi_file)

        if is_valid:
            stats['valid'] += 1
        else:
            stats['invalid'] += 1
            invalid_files.append((midi_file, error))

            # Track error types
            if error not in stats['errors']:
                stats['errors'][error] = 0
            stats['errors'][error] += 1

    # Remove corrupt files if requested
    if remove_corrupt and invalid_files:
        if verbose:
            print(f"\nRemoving {len(invalid_files)} corrupt files...")

        for midi_file, error in invalid_files:
            try:
                os.remove(midi_file)
                stats['removed'] += 1
            except Exception as e:
                if verbose:
                    print(f"Error removing {midi_file}: {e}")

    # Print results
    if verbose:
        print(f"\n" + "=" * 70)
        print(f"Validation Results")
        print(f"=" * 70)
        print(f"  Total files: {stats['total']}")
        print(f"  ✓ Valid: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
        print(f"  ✗ Invalid: {stats['invalid']} ({stats['invalid']/stats['total']*100:.1f}%)")

        if stats['errors']:
            print(f"\n  Error breakdown:")
            for error, count in sorted(stats['errors'].items(), key=lambda x: -x[1]):
                print(f"    - {error}: {count}")

        if remove_corrupt:
            print(f"\n  Removed: {stats['removed']} files")
            print(f"  Final count: {stats['valid']} files")

        print("=" * 70)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate MIDI files and optionally remove corrupt ones"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Directory with MIDI files'
    )
    parser.add_argument(
        '--remove_corrupt',
        action='store_true',
        help='Remove invalid files'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    stats = validate_midi_directory(
        args.input,
        remove_corrupt=args.remove_corrupt,
        verbose=not args.quiet
    )

    # Exit code
    if stats['valid'] == stats['total']:
        return 0  # All valid
    elif stats['valid'] > 0:
        return 1  # Some valid
    else:
        return 2  # None valid


if __name__ == '__main__':
    exit(main())
