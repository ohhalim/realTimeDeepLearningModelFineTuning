"""
Aria Tokenizer Configuration

⚠️ IMPORTANT: Token range assumptions are currently UNVERIFIED against official Aria tokenizer.

This module provides configurable token ranges based on reverse-engineering from:
1. ImprovNet paper (mentions "chunked absolute encoding" with 10ms resolution)
2. Analysis of Aria-MIDI dataset papers
3. Common MIDI tokenization patterns

Until verified against official ariautils.tokenizer.AbsTokenizer source code, these
ranges should be considered BEST-EFFORT ESTIMATES.

Status: UNVERIFIED - USE WITH CAUTION
TODO: Verify against https://github.com/EleutherAI/aria (ariautils package)
"""

from dataclasses import dataclass
from typing import Optional
from enum import IntEnum


class TokenType(IntEnum):
    """Token types in Aria encoding"""
    ONSET = 0
    DURATION = 1
    PITCH_VELOCITY = 2
    SPECIAL = 3


@dataclass
class AriaTokenizerConfig:
    """
    Configuration for Aria tokenizer ranges

    ⚠️ WARNING: These ranges are ASSUMPTIONS based on:
    - ImprovNet paper: "10ms quantization"
    - Common practice: 5-second chunks = 500 time steps @ 10ms
    - MIDI standard: 128 pitches
    - Velocity quantization: Likely 8-16 bins (assumed 8)

    UNVERIFIED ASSUMPTIONS:
    - onset_start=0, onset_end=500 (0-5000ms)
    - duration_start=501, duration_end=1000 (0-5000ms)
    - pitch_velocity_start=1001, pitch_velocity_end=1896 (128*7 = 896 tokens)
    - special_start=1897 (<T>, <sep>, <mask>, etc.)

    Total vocab: ~2048 tokens (rough estimate)

    To verify, check:
    1. ariautils.tokenizer.AbsTokenizer().vocab_size
    2. ariautils.tokenizer.AbsTokenizer().config (if available)
    """

    # Onset tokens (absolute time in 10ms units)
    onset_start: int = 0
    onset_end: int = 500  # 0-5000ms @ 10ms resolution

    # Duration tokens (in 10ms units)
    duration_start: int = 501
    duration_end: int = 1000  # 0-5000ms @ 10ms resolution

    # Pitch+Velocity merged tokens
    pitch_velocity_start: int = 1001
    pitch_velocity_end: int = 1896  # 128 pitches × 7 velocities = 896 tokens

    # Special tokens
    special_start: int = 1897
    special_end: int = 2047

    # Special token IDs (tentative)
    chunk_separator_token: int = 1897  # <T> token (resets onset to 0)
    mask_token: int = 1898  # <MASK> token
    whole_mask_token: int = 1899  # <WHOLE_MASK> token (for continuation/infilling)
    sep_token: int = 1900  # <SEP> token
    pad_token: int = 1901  # <PAD> token
    bos_token: int = 1902  # <BOS> token (begin of sequence)
    eos_token: int = 1903  # <EOS> token (end of sequence)

    # Derived parameters
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size"""
        return self.special_end + 1  # 2048

    @property
    def max_onset_ms(self) -> int:
        """Maximum onset time in milliseconds"""
        return (self.onset_end - self.onset_start) * 10  # 5000ms = 5 seconds

    @property
    def max_duration_ms(self) -> int:
        """Maximum duration in milliseconds"""
        return (self.duration_end - self.duration_start) * 10  # 5000ms = 5 seconds

    @property
    def num_pitches(self) -> int:
        """Number of MIDI pitches (0-127)"""
        return 128

    @property
    def num_velocity_bins(self) -> int:
        """Number of velocity bins"""
        total_pv_tokens = self.pitch_velocity_end - self.pitch_velocity_start + 1
        return total_pv_tokens // self.num_pitches  # 896 / 128 = 7

    def identify_token_type(self, token: int) -> TokenType:
        """
        Identify the type of a token

        Args:
            token: Token ID

        Returns:
            TokenType enum value
        """
        if self.onset_start <= token <= self.onset_end:
            return TokenType.ONSET
        elif self.duration_start <= token <= self.duration_end:
            return TokenType.DURATION
        elif self.pitch_velocity_start <= token <= self.pitch_velocity_end:
            return TokenType.PITCH_VELOCITY
        elif token >= self.special_start:
            return TokenType.SPECIAL
        else:
            raise ValueError(f"Token {token} outside known ranges (vocab_size={self.vocab_size})")

    def token_to_onset_ms(self, token: int) -> Optional[int]:
        """Convert onset token to milliseconds"""
        if self.identify_token_type(token) != TokenType.ONSET:
            return None
        return (token - self.onset_start) * 10

    def onset_ms_to_token(self, onset_ms: int) -> int:
        """Convert milliseconds to onset token"""
        token = self.onset_start + (onset_ms // 10)
        if token > self.onset_end:
            raise ValueError(f"Onset {onset_ms}ms exceeds max {self.max_onset_ms}ms")
        return token

    def token_to_duration_ms(self, token: int) -> Optional[int]:
        """Convert duration token to milliseconds"""
        if self.identify_token_type(token) != TokenType.DURATION:
            return None
        return (token - self.duration_start) * 10

    def duration_ms_to_token(self, duration_ms: int) -> int:
        """Convert milliseconds to duration token"""
        token = self.duration_start + (duration_ms // 10)
        if token > self.duration_end:
            raise ValueError(f"Duration {duration_ms}ms exceeds max {self.max_duration_ms}ms")
        return token

    def token_to_pitch_velocity(self, token: int) -> Optional[Tuple[int, int]]:
        """
        Convert pitch-velocity token to (pitch, velocity_bin)

        ⚠️ WARNING: This assumes linear encoding: token = base + pitch * num_vel_bins + vel_bin
        This is UNVERIFIED and may be incorrect!
        """
        if self.identify_token_type(token) != TokenType.PITCH_VELOCITY:
            return None

        offset = token - self.pitch_velocity_start
        pitch = offset // self.num_velocity_bins
        velocity_bin = offset % self.num_velocity_bins

        return (pitch, velocity_bin)

    def pitch_velocity_to_token(self, pitch: int, velocity_bin: int) -> int:
        """
        Convert (pitch, velocity_bin) to token

        Args:
            pitch: MIDI pitch (0-127)
            velocity_bin: Velocity bin (0-num_velocity_bins-1)

        Returns:
            Pitch-velocity token ID
        """
        if not (0 <= pitch < self.num_pitches):
            raise ValueError(f"Pitch {pitch} out of range [0, {self.num_pitches})")
        if not (0 <= velocity_bin < self.num_velocity_bins):
            raise ValueError(f"Velocity bin {velocity_bin} out of range [0, {self.num_velocity_bins})")

        offset = pitch * self.num_velocity_bins + velocity_bin
        return self.pitch_velocity_start + offset


# Default configuration (UNVERIFIED ASSUMPTIONS)
DEFAULT_ARIA_CONFIG = AriaTokenizerConfig()


def verify_against_real_tokenizer():
    """
    Placeholder function to verify our assumptions against real Aria tokenizer

    TODO: Implement this by:
    1. pip install ariautils (if available)
    2. from ariautils.tokenizer import AbsTokenizer
    3. Compare AbsTokenizer().vocab_size, token ranges, etc.
    4. Update AriaTokenizerConfig with correct values
    """
    try:
        # Try to import real tokenizer
        from ariautils.tokenizer import AbsTokenizer

        real_tokenizer = AbsTokenizer()
        real_vocab_size = real_tokenizer.vocab_size

        print(f"✓ Found real Aria tokenizer!")
        print(f"  Real vocab_size: {real_vocab_size}")
        print(f"  Our assumption:  {DEFAULT_ARIA_CONFIG.vocab_size}")

        if real_vocab_size != DEFAULT_ARIA_CONFIG.vocab_size:
            print(f"⚠️  WARNING: Vocab size mismatch!")
            print(f"  Please update AriaTokenizerConfig with correct values")
            return False

        return True

    except ImportError:
        print("⚠️  ariautils package not available")
        print("  Cannot verify token ranges against real Aria tokenizer")
        print("  Using UNVERIFIED assumptions")
        return None


if __name__ == "__main__":
    print("=" * 70)
    print("Aria Tokenizer Configuration Test")
    print("=" * 70)

    config = DEFAULT_ARIA_CONFIG

    print(f"\n📊 Configuration Summary:")
    print(f"  Total vocab size: {config.vocab_size}")
    print(f"  Onset range:      {config.onset_start}-{config.onset_end} (max {config.max_onset_ms}ms)")
    print(f"  Duration range:   {config.duration_start}-{config.duration_end} (max {config.max_duration_ms}ms)")
    print(f"  Pitch-vel range:  {config.pitch_velocity_start}-{config.pitch_velocity_end}")
    print(f"  Special range:    {config.special_start}-{config.special_end}")
    print(f"  Num pitches:      {config.num_pitches}")
    print(f"  Velocity bins:    {config.num_velocity_bins}")

    print(f"\n🧪 Testing conversions:")

    # Test onset
    onset_token = config.onset_ms_to_token(1230)  # 1.23 seconds
    onset_ms = config.token_to_onset_ms(onset_token)
    print(f"  1230ms -> token {onset_token} -> {onset_ms}ms (rounded to {onset_ms}ms)")

    # Test duration
    dur_token = config.duration_ms_to_token(500)  # 0.5 seconds
    dur_ms = config.token_to_duration_ms(dur_token)
    print(f"  500ms -> token {dur_token} -> {dur_ms}ms")

    # Test pitch-velocity
    pv_token = config.pitch_velocity_to_token(60, 3)  # Middle C, velocity bin 3
    pitch, vel_bin = config.token_to_pitch_velocity(pv_token)
    print(f"  (pitch=60, vel_bin=3) -> token {pv_token} -> (pitch={pitch}, vel_bin={vel_bin})")

    print(f"\n⚠️  VERIFICATION STATUS:")
    result = verify_against_real_tokenizer()
    if result is True:
        print("  ✓ Configuration VERIFIED against real tokenizer")
    elif result is False:
        print("  ✗ Configuration MISMATCH with real tokenizer")
    else:
        print("  ? Configuration UNVERIFIED (real tokenizer not available)")
        print("  Using best-effort assumptions - proceed with caution!")

    print("\n" + "=" * 70)
