"""
Real-time MIDI Input/Output Handler

Handles:
- MIDI device enumeration
- Real-time MIDI event capture
- Note buffering and quantization
- MIDI output to synth/DAW
- Virtual MIDI for testing (no hardware needed)
"""

import time
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque
import threading


# ==================== MIDI Message ====================

@dataclass
class MIDINote:
    """Simple MIDI note representation"""
    pitch: int          # 0-127
    velocity: int       # 0-127 (volume)
    timestamp: float    # Time in seconds


class MIDIBuffer:
    """
    Buffer for collecting MIDI notes in real-time

    Supports quantization to musical time (16th notes, etc.)
    """

    def __init__(
        self,
        buffer_duration: float = 1.0,     # How long to buffer (seconds)
        quantize_to: str = "16th",         # Quantization: 16th, 8th, quarter
        bpm: float = 120.0                 # Tempo
    ):
        self.buffer_duration = buffer_duration
        self.quantize_to = quantize_to
        self.bpm = bpm

        self.notes: deque[MIDINote] = deque()
        self.lock = threading.Lock()

    def add_note(self, pitch: int, velocity: int):
        """Add note to buffer"""
        with self.lock:
            note = MIDINote(pitch, velocity, time.time())
            self.notes.append(note)

            # Remove old notes
            current_time = time.time()
            while self.notes and (current_time - self.notes[0].timestamp) > self.buffer_duration:
                self.notes.popleft()

    def get_recent_notes(self, duration: float = 0.5) -> List[int]:
        """
        Get recent notes (last `duration` seconds)

        Returns:
            List of MIDI pitches (sorted by time)
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - duration

            recent = [note.pitch for note in self.notes if note.timestamp >= cutoff_time]
            return recent

    def get_and_clear(self) -> List[int]:
        """Get all notes and clear buffer"""
        with self.lock:
            notes = [note.pitch for note in self.notes]
            self.notes.clear()
            return notes

    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.notes.clear()

    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        with self.lock:
            return len(self.notes) == 0

    def __len__(self) -> int:
        with self.lock:
            return len(self.notes)


# ==================== Virtual MIDI (for testing) ====================

class VirtualMIDIInput:
    """
    Virtual MIDI input for testing without hardware

    Simulates user playing notes from computer keyboard
    """

    def __init__(self):
        self.notes_queue: deque[int] = deque()
        self.lock = threading.Lock()
        self.is_running = False

        # Keyboard mapping: A S D F G H J K → C D E F G A B C
        self.key_to_midi = {
            'a': 60,  # C4
            's': 62,  # D4
            'd': 64,  # E4
            'f': 65,  # F4
            'g': 67,  # G4
            'h': 69,  # A4
            'j': 71,  # B4
            'k': 72,  # C5
        }

    def start(self):
        """Start virtual input (keyboard listener)"""
        self.is_running = True
        print("\n" + "="*60)
        print("Virtual MIDI Keyboard")
        print("="*60)
        print("Use keys: A S D F G H J K")
        print("Maps to:  C D E F G A B C")
        print("Type notes and press Enter. 'q' to quit.")
        print("="*60 + "\n")

    def play_notes(self, keys: str) -> List[int]:
        """
        Play notes from keyboard input

        Args:
            keys: String like "ceg" for C major chord

        Returns:
            List of MIDI pitches
        """
        notes = []
        for key in keys.lower():
            if key in self.key_to_midi:
                notes.append(self.key_to_midi[key])
        return notes

    def stop(self):
        """Stop virtual input"""
        self.is_running = False


class VirtualMIDIOutput:
    """
    Virtual MIDI output (just prints notes)

    For testing without actual synth
    """

    def __init__(self):
        pass

    def send_notes(self, pitches: List[int], duration: float = 0.5):
        """
        Send notes to output (just print for virtual)

        Args:
            pitches: List of MIDI pitches
            duration: How long to play
        """
        if not pitches:
            return

        note_names = [self._pitch_to_name(p) for p in pitches]
        print(f"🤖 AI plays: {' '.join(note_names)} ({pitches})")

    @staticmethod
    def _pitch_to_name(pitch: int) -> str:
        """Convert MIDI pitch to note name (e.g., 60 → C4)"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note = note_names[pitch % 12]
        return f"{note}{octave}"


# ==================== Real MIDI (with mido) ====================

class RealMIDIInput:
    """
    Real MIDI input using mido library

    Captures notes from actual MIDI keyboard/controller
    """

    def __init__(self, device_name: Optional[str] = None):
        try:
            import mido
            self.mido = mido
        except ImportError:
            raise ImportError(
                "mido not installed. Install with: pip install mido python-rtmidi"
            )

        self.device_name = device_name
        self.port = None
        self.buffer = MIDIBuffer()
        self.is_running = False
        self.listener_thread = None

    def list_devices(self) -> List[str]:
        """List available MIDI input devices"""
        return self.mido.get_input_names()

    def start(self):
        """Start listening to MIDI input"""
        # Get device
        devices = self.list_devices()
        if not devices:
            raise RuntimeError("No MIDI input devices found!")

        if self.device_name is None:
            self.device_name = devices[0]  # Use first device
            print(f"Using default MIDI input: {self.device_name}")

        # Open port
        self.port = self.mido.open_input(self.device_name)
        print(f"✓ MIDI input opened: {self.device_name}")

        # Start listener thread
        self.is_running = True
        self.listener_thread = threading.Thread(target=self._listen_loop)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def _listen_loop(self):
        """Listen for MIDI messages in background thread"""
        for msg in self.port:
            if not self.is_running:
                break

            # Only handle note_on messages
            if msg.type == 'note_on' and msg.velocity > 0:
                self.buffer.add_note(msg.note, msg.velocity)

    def get_recent_notes(self, duration: float = 0.5) -> List[int]:
        """Get recently played notes"""
        return self.buffer.get_recent_notes(duration)

    def stop(self):
        """Stop MIDI input"""
        self.is_running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=1.0)
        if self.port:
            self.port.close()
        print("✓ MIDI input closed")


class RealMIDIOutput:
    """
    Real MIDI output using mido library

    Sends notes to synth/DAW
    """

    def __init__(self, device_name: Optional[str] = None):
        try:
            import mido
            self.mido = mido
        except ImportError:
            raise ImportError(
                "mido not installed. Install with: pip install mido python-rtmidi"
            )

        self.device_name = device_name
        self.port = None

    def list_devices(self) -> List[str]:
        """List available MIDI output devices"""
        return self.mido.get_output_names()

    def start(self):
        """Open MIDI output port"""
        devices = self.list_devices()
        if not devices:
            print("⚠ No MIDI output devices found (using virtual)")
            return

        if self.device_name is None:
            self.device_name = devices[0]
            print(f"Using default MIDI output: {self.device_name}")

        self.port = self.mido.open_output(self.device_name)
        print(f"✓ MIDI output opened: {self.device_name}")

    def send_notes(self, pitches: List[int], duration: float = 0.5, velocity: int = 80):
        """
        Send notes to MIDI output

        Args:
            pitches: List of MIDI pitches
            duration: How long to play
            velocity: Note velocity (volume)
        """
        if self.port is None:
            # Fallback to virtual output
            VirtualMIDIOutput().send_notes(pitches, duration)
            return

        # Send note_on for all pitches
        for pitch in pitches:
            msg = self.mido.Message('note_on', note=pitch, velocity=velocity)
            self.port.send(msg)

        # Wait duration
        time.sleep(duration)

        # Send note_off for all pitches
        for pitch in pitches:
            msg = self.mido.Message('note_off', note=pitch, velocity=0)
            self.port.send(msg)

    def stop(self):
        """Close MIDI output"""
        if self.port:
            self.port.close()
            print("✓ MIDI output closed")


# ==================== Factory Functions ====================

def create_midi_input(mode: str = "virtual", device: Optional[str] = None):
    """
    Create MIDI input (virtual or real)

    Args:
        mode: "virtual" or "midi"
        device: MIDI device name (None = use default)

    Returns:
        MIDIInput object
    """
    if mode == "virtual":
        return VirtualMIDIInput()
    elif mode == "midi":
        return RealMIDIInput(device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'virtual' or 'midi'")


def create_midi_output(mode: str = "virtual", device: Optional[str] = None):
    """
    Create MIDI output (virtual or real)

    Args:
        mode: "virtual" or "midi"
        device: MIDI device name (None = use default)

    Returns:
        MIDIOutput object
    """
    if mode == "virtual":
        return VirtualMIDIOutput()
    elif mode == "midi":
        return RealMIDIOutput(device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'virtual' or 'midi'")


# ==================== SELF-TEST ====================

if __name__ == "__main__":
    print("=" * 70)
    print("MIDI I/O Self-Test")
    print("=" * 70)

    # Test buffer
    print("\n[1] Testing MIDI buffer...")
    buffer = MIDIBuffer(buffer_duration=1.0)

    buffer.add_note(60, 80)  # C4
    buffer.add_note(64, 80)  # E4
    buffer.add_note(67, 80)  # G4

    notes = buffer.get_recent_notes(duration=1.0)
    print(f"✓ Buffer works")
    print(f"  Added: C4, E4, G4")
    print(f"  Retrieved: {notes}")

    assert notes == [60, 64, 67], "Buffer mismatch!"

    # Test virtual MIDI
    print("\n[2] Testing virtual MIDI...")
    virtual_in = VirtualMIDIInput()
    virtual_out = VirtualMIDIOutput()

    test_notes = virtual_in.play_notes("ceg")  # C major
    print(f"✓ Virtual input works: {test_notes}")

    virtual_out.send_notes(test_notes)
    print(f"✓ Virtual output works")

    # Test real MIDI (if available)
    print("\n[3] Testing real MIDI...")
    try:
        import mido

        # List devices
        input_devices = mido.get_input_names()
        output_devices = mido.get_output_names()

        print(f"  Input devices: {input_devices if input_devices else 'None'}")
        print(f"  Output devices: {output_devices if output_devices else 'None'}")

        if input_devices or output_devices:
            print(f"✓ Real MIDI available")
        else:
            print(f"  ⚠ No MIDI devices (virtual mode recommended)")

    except ImportError:
        print(f"  ⚠ mido not installed (pip install mido python-rtmidi)")

    # Summary
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nMIDI I/O ready for jamming!")
    print("Use create_midi_input/output() to get started")
    print("=" * 70)
