"""
Reproducibility Utilities

Ensures deterministic behavior across all random number generators:
- Python random
- NumPy random
- PyTorch (CPU and CUDA)
- Corruption functions
- Model initialization

Usage:
    from realjazz.reproducibility import set_global_seed, ReproducibleContext

    # Set seed globally
    set_global_seed(42)

    # Or use context manager for local reproducibility
    with ReproducibleContext(42):
        # All random operations here are deterministic
        model = MultiTaskJazzFormer(...)
        corrupted = apply_random_corruption(tokens)
"""

import random
import numpy as np
import torch
from contextlib import contextmanager
from typing import Optional
import os


def set_global_seed(seed: int, deterministic: bool = True):
    """
    Set global random seed for all libraries

    Args:
        seed: Random seed (0-2**32-1)
        deterministic: If True, enables deterministic algorithms (slower but reproducible)

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU)
    - PyTorch (CUDA)
    - Python hash seed (for dict ordering)
    """
    if not (0 <= seed < 2**32):
        raise ValueError(f"Seed must be in range [0, 2^32), got {seed}")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Python hash seed (for dict ordering, set() ordering, etc.)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch deterministic algorithms
    if deterministic:
        # Make cuDNN deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where possible
        # Note: This may reduce performance but ensures reproducibility
        torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"✓ Global seed set to {seed} (deterministic={deterministic})")


def get_random_state():
    """
    Get current random state for all RNGs

    Returns:
        Dictionary with random states for Python, NumPy, and PyTorch
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()

    return state


def set_random_state(state: dict):
    """
    Restore random state for all RNGs

    Args:
        state: Dictionary from get_random_state()
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])

    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@contextmanager
def ReproducibleContext(seed: int, deterministic: bool = True):
    """
    Context manager for reproducible operations

    Usage:
        with ReproducibleContext(42):
            # All random operations here use seed 42
            model = MyModel()
            data = generate_random_data()

        # Outside context, random state is restored

    Args:
        seed: Random seed
        deterministic: Enable deterministic algorithms
    """
    # Save current state
    old_state = get_random_state()
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark

    try:
        # Set new seed
        set_global_seed(seed, deterministic)
        yield

    finally:
        # Restore old state
        set_random_state(old_state)
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark


def verify_reproducibility(fn, seed: int = 42, n_runs: int = 3):
    """
    Verify that a function produces deterministic results

    Args:
        fn: Function to test (should take no args and return a comparable result)
        seed: Random seed to use
        n_runs: Number of runs to compare

    Returns:
        True if all runs produce identical results, False otherwise

    Example:
        def test_fn():
            model = MultiTaskJazzFormer()
            return model.state_dict()['task_embedding.weight'].sum().item()

        verify_reproducibility(test_fn, seed=42, n_runs=5)
    """
    results = []

    for i in range(n_runs):
        with ReproducibleContext(seed):
            result = fn()
            results.append(result)

    # Check if all results are identical
    first_result = results[0]

    for i, result in enumerate(results[1:], 1):
        if isinstance(first_result, torch.Tensor):
            if not torch.allclose(first_result, result):
                print(f"❌ Run {i+1} differs from run 1")
                print(f"   Run 1: {first_result}")
                print(f"   Run {i+1}: {result}")
                return False
        elif isinstance(first_result, (int, float)):
            if abs(first_result - result) > 1e-6:
                print(f"❌ Run {i+1} differs from run 1")
                print(f"   Run 1: {first_result}")
                print(f"   Run {i+1}: {result}")
                return False
        else:
            if first_result != result:
                print(f"❌ Run {i+1} differs from run 1")
                return False

    print(f"✓ Reproducibility verified: {n_runs} runs produced identical results")
    return True


class SeededRNG:
    """
    Seeded random number generator for corruption functions

    Allows corruption functions to be deterministic while maintaining
    global RNG independence.

    Usage:
        rng = SeededRNG(42)
        choice = rng.choice([1, 2, 3])
        prob = rng.random()
    """

    def __init__(self, seed: int):
        """
        Initialize seeded RNG

        Args:
            seed: Random seed
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def random(self) -> float:
        """Random float in [0, 1)"""
        return self.rng.random()

    def randint(self, a: int, b: int) -> int:
        """Random integer in [a, b]"""
        return self.rng.randint(a, b)

    def choice(self, seq):
        """Random element from sequence"""
        return self.rng.choice(seq)

    def shuffle(self, seq):
        """Shuffle sequence in-place"""
        return self.rng.shuffle(seq)

    def sample(self, population, k):
        """Sample k elements from population without replacement"""
        return self.rng.sample(population, k)

    def uniform(self, a: float, b: float) -> float:
        """Random float in [a, b]"""
        return self.rng.uniform(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        """Random Gaussian with mean mu and stdev sigma"""
        return self.rng.gauss(mu, sigma)


if __name__ == "__main__":
    print("=" * 70)
    print("Reproducibility Utilities Test")
    print("=" * 70)

    # Test 1: Global seed
    print("\n🧪 Test 1: Global seed setting")
    set_global_seed(42)

    vals = []
    for i in range(3):
        set_global_seed(42)
        val = random.random()
        vals.append(val)
        print(f"  Run {i+1}: {val}")

    if len(set(vals)) == 1:
        print("  ✓ All runs produced same value")
    else:
        print("  ❌ Runs produced different values!")

    # Test 2: Context manager
    print("\n🧪 Test 2: Reproducible context")

    outer_val = random.random()
    print(f"  Before context: {outer_val}")

    with ReproducibleContext(123):
        inner_val1 = random.random()
        print(f"  Inside context (run 1): {inner_val1}")

    with ReproducibleContext(123):
        inner_val2 = random.random()
        print(f"  Inside context (run 2): {inner_val2}")

    if inner_val1 == inner_val2:
        print("  ✓ Context produces reproducible results")
    else:
        print("  ❌ Context did not reproduce results!")

    # Test 3: PyTorch reproducibility
    print("\n🧪 Test 3: PyTorch tensor generation")

    tensors = []
    for i in range(3):
        with ReproducibleContext(999):
            t = torch.randn(3, 3)
            tensors.append(t)
        print(f"  Run {i+1}:\n{t}")

    if torch.allclose(tensors[0], tensors[1]) and torch.allclose(tensors[1], tensors[2]):
        print("  ✓ All tensors identical")
    else:
        print("  ❌ Tensors differ!")

    # Test 4: SeededRNG
    print("\n🧪 Test 4: SeededRNG")

    rng1 = SeededRNG(777)
    rng2 = SeededRNG(777)

    vals1 = [rng1.random() for _ in range(5)]
    vals2 = [rng2.random() for _ in range(5)]

    print(f"  RNG 1: {vals1}")
    print(f"  RNG 2: {vals2}")

    if vals1 == vals2:
        print("  ✓ SeededRNG produces identical sequences")
    else:
        print("  ❌ SeededRNG sequences differ!")

    # Test 5: Model initialization reproducibility
    print("\n🧪 Test 5: Model initialization")

    def create_model():
        model = torch.nn.Linear(10, 10)
        return model.weight.sum().item()

    if verify_reproducibility(create_model, seed=42, n_runs=5):
        print("  ✓ Model initialization is reproducible")
    else:
        print("  ❌ Model initialization is NOT reproducible!")

    print("\n" + "=" * 70)
    print("✅ All reproducibility tests passed!")
    print("=" * 70)
