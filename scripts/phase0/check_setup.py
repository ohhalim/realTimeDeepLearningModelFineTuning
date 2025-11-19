#!/usr/bin/env python3
"""
Check Phase 0 setup completion

Verifies:
1. Python version
2. Dependencies installed
3. Data collected
4. RunPod ready (manual check)
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.9"""
    version = sys.version_info
    required = (3, 9)

    if version >= required:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (need >= 3.9)")
        return False


def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name} {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} not found")
        return False


def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA not available (CPU mode - OK for development)")

        return True
    except ImportError:
        print(f"❌ PyTorch not found")
        return False


def check_data_directory(data_dir='data/jazz_midi'):
    """Check if data directory has MIDI files"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False

    # Count MIDI files
    midi_files = []
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(data_path.glob(ext))

    count = len(midi_files)

    if count == 0:
        print(f"❌ No MIDI files in {data_dir}")
        return False
    elif count < 50:
        print(f"⚠️  Only {count} MIDI files (recommended: 50+)")
        return True
    else:
        print(f"✅ {count} MIDI files in {data_dir}")

        # Calculate total duration
        try:
            import mido
            total_duration = 0
            for f in midi_files[:min(count, 100)]:  # Sample max 100 for speed
                try:
                    mid = mido.MidiFile(f)
                    total_duration += mid.length
                except:
                    pass

            hours = total_duration / 3600
            print(f"  ✅ Total duration: {hours:.1f}h (sampled)")
        except:
            pass

        return True


def check_runpod():
    """Manual check for RunPod account"""
    print("⚠️  RunPod account: Check manually at https://runpod.io")
    print("   - Account created?")
    print("   - Credit added ($10+)?")
    print("   - SSH key registered?")
    return None  # Manual check


def main():
    print("=" * 70)
    print("Phase 0 Setup Check")
    print("=" * 70)
    print()

    checks = []

    # Python version
    checks.append(check_python_version())

    # Core dependencies
    print()
    checks.append(check_pytorch_cuda())
    checks.append(check_package('numpy'))
    checks.append(check_package('mido'))
    checks.append(check_package('tqdm'))

    # Data
    print()
    checks.append(check_data_directory())

    # RunPod (manual)
    print()
    check_runpod()

    # Summary
    print()
    print("=" * 70)

    passed = sum(1 for c in checks if c)
    total = len(checks)

    if passed == total:
        print("🎉 All checks passed! Ready for Phase 1.")
        print()
        print("Next step:")
        print("  Read guides/PHASE1_TRAINING.md")
        print("=" * 70)
        return 0
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print()
        print("Fix the issues above before starting Phase 1")
        print("=" * 70)
        return 1


if __name__ == '__main__':
    exit(main())
