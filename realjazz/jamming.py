"""
Real-Time Jazz Jam Session

Main loop that connects:
- MIDI input (user playing)
- AnticipativeJazzFormer (AI)
- MIDI output (AI response)

Target latency: <100ms
"""

import torch
import time
from typing import Optional, List
from dataclasses import dataclass, field

from .model import AnticipativeJazzFormer, count_parameters
from .midi_io import create_midi_input, create_midi_output


@dataclass
class JamConfig:
    """Configuration for jam session"""
    # Model
    model_path: Optional[str] = None      # Path to pretrained weights
    device: str = "cpu"                    # cpu or cuda
    temperature: float = 0.9               # Sampling temperature
    top_p: float = 0.95                    # Nucleus sampling

    # Generation
    response_length: int = 8               # How many notes AI plays
    min_user_notes: int = 3                # Min notes before AI responds

    # MIDI
    midi_mode: str = "virtual"             # virtual or midi
    input_device: Optional[str] = None
    output_device: Optional[str] = None

    # Performance
    latency_target: float = 100.0          # Target latency (ms)
    measure_latency: bool = True

    # Jamming behavior
    wait_for_pause: bool = True            # Wait for user to pause
    pause_duration: float = 0.5            # How long to wait (seconds)


@dataclass
class JamStats:
    """Statistics for jam session"""
    num_exchanges: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        """Average latency"""
        return self.total_latency_ms / max(1, self.num_exchanges)

    def add_exchange(self, latency_ms: float):
        """Record a jam exchange"""
        self.num_exchanges += 1
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)

    def summary(self) -> str:
        """Get summary string"""
        if self.num_exchanges == 0:
            return "No exchanges yet"

        return f"""
Jam Session Stats:
  Exchanges: {self.num_exchanges}
  Avg latency: {self.avg_latency:.1f}ms
  Min latency: {min(self.latencies):.1f}ms
  Max latency: {max(self.latencies):.1f}ms
  Target: <100ms {'✓' if self.avg_latency < 100 else '⚠'}
"""


class JamSession:
    """
    Real-time jazz jam session

    Usage:
        session = JamSession(config)
        session.start()
        # Jam with the AI!
        session.stop()
    """

    def __init__(self, config: Optional[JamConfig] = None):
        self.config = config or JamConfig()
        self.stats = JamStats()

        # Create model
        print("\n" + "="*70)
        print("ReaLJazz: Real-Time Jazz Jam Bot")
        print("="*70)

        print("\n[1] Loading model...")
        self.model = AnticipativeJazzFormer(
            vocab_size=128,
            d_model=256,
            n_heads=4,
            n_layers=4
        )

        # Load pretrained weights if available
        if self.config.model_path:
            try:
                state_dict = torch.load(self.config.model_path, map_location=self.config.device)
                self.model.load_state_dict(state_dict)
                print(f"  ✓ Loaded pretrained weights from {self.config.model_path}")
            except Exception as e:
                print(f"  ⚠ Could not load weights: {e}")
                print(f"  Using randomly initialized model")

        self.model.to(self.config.device)
        self.model.eval()

        n_params = count_parameters(self.model)
        print(f"  ✓ Model loaded ({n_params:,} parameters, ~{n_params/1e6:.1f}M)")

        # Create MIDI I/O
        print(f"\n[2] Setting up MIDI I/O ({self.config.midi_mode} mode)...")
        self.midi_in = create_midi_input(self.config.midi_mode, self.config.input_device)
        self.midi_out = create_midi_output(self.config.midi_mode, self.config.output_device)

        print(f"  ✓ MIDI input: {self.config.midi_mode}")
        print(f"  ✓ MIDI output: {self.config.midi_mode}")
        print(f"  ✓ Latency target: <{self.config.latency_target}ms")

        self.is_running = False
        self.last_user_time = 0.0

    def start(self):
        """Start jam session"""
        print("\n" + "="*70)
        print("Session starting...")
        print("="*70)

        self.midi_in.start()
        if hasattr(self.midi_out, 'start'):
            self.midi_out.start()

        self.is_running = True

        if self.config.midi_mode == "virtual":
            self._run_virtual_session()
        else:
            self._run_real_session()

    def _run_virtual_session(self):
        """Virtual keyboard session (interactive)"""
        print("\nVirtual Keyboard Guide:")
        print("  Type notes: a s d f g h j k")
        print("  Maps to:    C D E F G A B C")
        print("  Example: 'ceg' = C major chord")
        print("  Type 'q' to quit")
        print("\n" + "="*70)

        while self.is_running:
            try:
                # Get user input
                user_input = input("\n🎹 You play: ")

                if user_input.lower() == 'q':
                    break

                # Convert to MIDI notes
                user_notes = self.midi_in.play_notes(user_input)

                if not user_notes:
                    print("  ⚠ No valid notes (use: a s d f g h j k)")
                    continue

                if len(user_notes) < self.config.min_user_notes:
                    print(f"  ⚠ Play at least {self.config.min_user_notes} notes")
                    continue

                # Show what user played
                note_names = [self._pitch_to_name(p) for p in user_notes]
                chord_name = self._guess_chord(user_notes)
                print(f"  Notes: {' '.join(note_names)} ({chord_name})")

                # AI responds
                ai_response, latency = self._generate_response(user_notes)

                # Show AI response
                ai_note_names = [self._pitch_to_name(p) for p in ai_response]
                ai_chord = self._guess_chord(ai_response)
                print(f"\n🤖 AI responds: {' '.join(ai_note_names)} ({ai_chord})")
                print(f"   Latency: {latency:.1f}ms", end="")
                if latency < self.config.latency_target:
                    print(" ✓")
                else:
                    print(" ⚠")

                # Play AI response
                self.midi_out.send_notes(ai_response)

                # Update stats
                self.stats.add_exchange(latency)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Error: {e}")

    def _run_real_session(self):
        """Real MIDI keyboard session"""
        print("\nListening to MIDI input...")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")

        try:
            while self.is_running:
                # Check for user input
                if hasattr(self.midi_in, 'buffer'):
                    recent_notes = self.midi_in.buffer.get_recent_notes(
                        duration=self.config.pause_duration
                    )

                    if len(recent_notes) >= self.config.min_user_notes:
                        # User played something
                        current_time = time.time()

                        # Check if user paused (stopped playing)
                        if self.config.wait_for_pause:
                            time_since_last = current_time - self.last_user_time
                            if time_since_last < self.config.pause_duration:
                                # Still playing, wait
                                continue

                        # User paused, AI responds
                        user_notes = self.midi_in.buffer.get_and_clear()

                        # Show what user played
                        print(f"\n🎹 You played: {user_notes}")

                        # Generate AI response
                        ai_response, latency = self._generate_response(user_notes)

                        print(f"🤖 AI responds: {ai_response}  [Latency: {latency:.1f}ms]")

                        # Play AI response
                        self.midi_out.send_notes(ai_response)

                        # Update stats
                        self.stats.add_exchange(latency)

                        self.last_user_time = current_time

                # Small sleep to avoid busy-wait
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\nStopping...")

    def _generate_response(self, user_notes: List[int]) -> tuple[List[int], float]:
        """
        Generate AI response to user input

        Args:
            user_notes: List of MIDI pitches user played

        Returns:
            (ai_notes, latency_ms)
        """
        # Convert to tensor
        user_tokens = torch.tensor([user_notes], dtype=torch.long, device=self.config.device)

        # Generate with model
        ai_tokens, latency = self.model.generate_anticipatory(
            user_tokens,
            max_new_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )

        # Convert back to list
        ai_notes = ai_tokens[0].cpu().tolist()

        return ai_notes, latency

    @staticmethod
    def _pitch_to_name(pitch: int) -> str:
        """Convert MIDI pitch to note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (pitch // 12) - 1
        note = note_names[pitch % 12]
        return f"{note}{octave}"

    @staticmethod
    def _guess_chord(pitches: List[int]) -> str:
        """Guess chord name from pitches (simple heuristic)"""
        if not pitches:
            return "?"

        # Get unique pitch classes
        pcs = sorted(set(p % 12 for p in pitches))

        # Simple chord detection
        if len(pcs) >= 3:
            root = pcs[0]
            intervals = [(pc - root) % 12 for pc in pcs]

            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            root_name = note_names[root]

            # Major: 0, 4, 7
            if {0, 4, 7}.issubset(intervals):
                return f"{root_name}maj"
            # Minor: 0, 3, 7
            elif {0, 3, 7}.issubset(intervals):
                return f"{root_name}m"
            # Dominant 7th: 0, 4, 7, 10
            elif {0, 4, 7, 10}.issubset(intervals):
                return f"{root_name}7"
            # Minor 7th: 0, 3, 7, 10
            elif {0, 3, 7, 10}.issubset(intervals):
                return f"{root_name}m7"
            else:
                return f"{root_name}..."

        return "?"

    def stop(self):
        """Stop jam session"""
        print("\n" + "="*70)
        print("Stopping jam session...")
        print("="*70)

        self.is_running = False

        self.midi_in.stop()
        if hasattr(self.midi_out, 'stop'):
            self.midi_out.stop()

        # Print stats
        print(self.stats.summary())

        print("\nThanks for jamming! 🎹🎷🎺")
        print("="*70)

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.stop()


# ==================== CONVENIENCE FUNCTION ====================

def jam(
    mode: str = "virtual",
    temperature: float = 0.9,
    response_length: int = 8
):
    """
    Quick start a jam session

    Args:
        mode: "virtual" or "midi"
        temperature: Sampling temperature (0.7-1.0)
        response_length: How many notes AI plays

    Example:
        >>> jam(mode="virtual", temperature=0.9)
    """
    config = JamConfig(
        midi_mode=mode,
        temperature=temperature,
        response_length=response_length
    )

    session = JamSession(config)
    session.start()
    session.stop()


# ==================== SELF-TEST ====================

if __name__ == "__main__":
    print("Testing JamSession...")

    # Create config
    config = JamConfig(
        midi_mode="virtual",
        temperature=0.9,
        response_length=6
    )

    # Create session
    session = JamSession(config)

    # Test response generation
    print("\nTesting AI response generation...")
    user_notes = [60, 64, 67]  # C major
    ai_notes, latency = session._generate_response(user_notes)

    print(f"User: {user_notes} (C major)")
    print(f"AI:   {ai_notes}")
    print(f"Latency: {latency:.1f}ms")

    if latency < 100:
        print("✓ Latency under 100ms target!")
    else:
        print("⚠ Latency over 100ms (consider smaller model or GPU)")

    print("\n" + "="*70)
    print("JamSession ready!")
    print("Run: python -m realjazz.jamming")
    print("Or:  from realjazz.jamming import jam; jam()")
    print("="*70)
