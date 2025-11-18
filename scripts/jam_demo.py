#!/usr/bin/env python3
"""
ReaLJazz Demo Script

Interactive demo for real-time jazz jam bot

Usage:
    # Virtual keyboard (no MIDI hardware needed)
    python scripts/jam_demo.py --mode virtual

    # Real MIDI keyboard
    python scripts/jam_demo.py --mode midi

    # List available MIDI devices
    python scripts/jam_demo.py --list-devices
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from realjazz import JamSession, JamConfig


def list_midi_devices():
    """List available MIDI devices"""
    try:
        import mido
    except ImportError:
        print("⚠ mido not installed")
        print("Install with: pip install mido python-rtmidi")
        return

    print("\n" + "="*70)
    print("Available MIDI Devices")
    print("="*70)

    input_devices = mido.get_input_names()
    output_devices = mido.get_output_names()

    print("\nInput devices:")
    if input_devices:
        for i, dev in enumerate(input_devices):
            print(f"  [{i}] {dev}")
    else:
        print("  (none)")

    print("\nOutput devices:")
    if output_devices:
        for i, dev in enumerate(output_devices):
            print(f"  [{i}] {dev}")
    else:
        print("  (none)")

    print("\n" + "="*70)
    print("\nUsage:")
    print('  python scripts/jam_demo.py --mode midi --input "Device Name"')
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="ReaLJazz: Real-Time Jazz Jam Bot Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Virtual keyboard (recommended for testing)
  python scripts/jam_demo.py --mode virtual

  # Real MIDI keyboard
  python scripts/jam_demo.py --mode midi

  # List available MIDI devices
  python scripts/jam_demo.py --list-devices

  # Custom configuration
  python scripts/jam_demo.py --mode virtual --temperature 0.8 --response-length 10
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="virtual",
        choices=["virtual", "midi"],
        help="MIDI mode: virtual (keyboard) or midi (real device)"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="MIDI input device name"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="MIDI output device name"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to pretrained model weights (.pt file)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (0.7-1.0, default: 0.9)"
    )

    parser.add_argument(
        "--response-length",
        type=int,
        default=8,
        help="How many notes AI plays (default: 8)"
    )

    parser.add_argument(
        "--min-notes",
        type=int,
        default=3,
        help="Minimum user notes before AI responds (default: 3)"
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available MIDI devices and exit"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on (cpu or cuda)"
    )

    args = parser.parse_args()

    # List devices and exit
    if args.list_devices:
        list_midi_devices()
        return

    # Print banner
    print("\n" + "="*70)
    print("██████╗ ███████╗ █████╗ ██╗         ██╗ █████╗ ███████╗███████╗")
    print("██╔══██╗██╔════╝██╔══██╗██║         ██║██╔══██╗╚══███╔╝╚══███╔╝")
    print("██████╔╝█████╗  ███████║██║         ██║███████║  ███╔╝   ███╔╝ ")
    print("██╔══██╗██╔══╝  ██╔══██║██║    ██   ██║██╔══██║ ███╔╝   ███╔╝  ")
    print("██║  ██║███████╗██║  ██║███████╗╚█████╔╝██║  ██║███████╗███████╗")
    print("╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚════╝ ╚═╝  ╚═╝╚══════╝╚══════╝")
    print("="*70)
    print("Real-Time AI Jazz Jam Bot")
    print("Based on Anticipatory Music Transformer (Stanford, 2024)")
    print("="*70)

    # Create configuration
    config = JamConfig(
        midi_mode=args.mode,
        input_device=args.input,
        output_device=args.output,
        model_path=args.model,
        device=args.device,
        temperature=args.temperature,
        response_length=args.response_length,
        min_user_notes=args.min_notes
    )

    # Create and start session
    try:
        session = JamSession(config)
        session.start()
        session.stop()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
