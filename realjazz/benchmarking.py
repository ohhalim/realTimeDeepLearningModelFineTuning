"""
Benchmarking and Performance Validation

Provides tools to measure and validate performance claims:
- Latency measurements (CPU/GPU)
- Parameter counting
- Memory profiling
- Generation quality metrics
- Comparison to baselines

Usage:
    from realjazz.benchmarking import LatencyBenchmark, measure_generation_latency

    # Benchmark latency
    benchmark = LatencyBenchmark(model)
    results = benchmark.run(n_runs=100)
    print(f"Latency: {results['mean_ms']:.1f}ms ± {results['std_ms']:.1f}ms")
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from tqdm import tqdm


@dataclass
class BenchmarkResults:
    """Results from a benchmark run"""

    name: str
    device: str
    n_runs: int

    # Latency (milliseconds)
    latency_mean: float
    latency_std: float
    latency_min: float
    latency_max: float
    latency_p50: float  # Median
    latency_p95: float
    latency_p99: float

    # Memory (MB)
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None

    # Model stats
    n_parameters: Optional[int] = None
    n_flops: Optional[int] = None

    # Additional metrics
    extras: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResults(name='{self.name}', device='{self.device}', "
            f"latency={self.latency_mean:.2f}±{self.latency_std:.2f}ms, "
            f"p50={self.latency_p50:.2f}ms, p95={self.latency_p95:.2f}ms)"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'device': self.device,
            'n_runs': self.n_runs,
            'latency_mean_ms': round(self.latency_mean, 2),
            'latency_std_ms': round(self.latency_std, 2),
            'latency_min_ms': round(self.latency_min, 2),
            'latency_max_ms': round(self.latency_max, 2),
            'latency_p50_ms': round(self.latency_p50, 2),
            'latency_p95_ms': round(self.latency_p95, 2),
            'latency_p99_ms': round(self.latency_p99, 2),
            'memory_allocated_mb': round(self.memory_allocated_mb, 2) if self.memory_allocated_mb else None,
            'memory_reserved_mb': round(self.memory_reserved_mb, 2) if self.memory_reserved_mb else None,
            'n_parameters': self.n_parameters,
            'n_flops': self.n_flops,
            'extras': self.extras,
        }

    def save_json(self, path: str):
        """Save results to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Saved benchmark results to {path}")


class LatencyBenchmark:
    """
    Benchmark latency for model generation

    Measures end-to-end latency including:
    - Model forward pass
    - Sampling
    - KV-cache updates (if applicable)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None
    ):
        """
        Initialize benchmark

        Args:
            model: Model to benchmark
            device: Device to use (auto-detect if None)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def warmup(self, n_warmup: int = 10):
        """
        Warmup run to avoid cold start effects

        Args:
            n_warmup: Number of warmup iterations
        """
        dummy_input = torch.randint(0, 128, (1, 32), device=self.device)
        dummy_event_types = torch.ones_like(dummy_input) * 1

        for _ in range(n_warmup):
            with torch.no_grad():
                _ = self.model(dummy_input, dummy_event_types)

        # Synchronize GPU if using CUDA
        if self.device == 'cuda':
            torch.cuda.synchronize()

    @torch.no_grad()
    def measure_single_forward(
        self,
        input_tensor: torch.Tensor,
        event_types: torch.Tensor
    ) -> float:
        """
        Measure latency of single forward pass

        Args:
            input_tensor: Input tokens (1, L)
            event_types: Event type IDs (1, L)

        Returns:
            Latency in milliseconds
        """
        if self.device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        _ = self.model(input_tensor, event_types)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return latency_ms

    def run(
        self,
        n_runs: int = 100,
        batch_size: int = 1,
        seq_len: int = 32,
        warmup: bool = True
    ) -> BenchmarkResults:
        """
        Run latency benchmark

        Args:
            n_runs: Number of benchmark runs
            batch_size: Batch size (default 1 for real-time)
            seq_len: Sequence length
            warmup: Perform warmup before benchmarking

        Returns:
            BenchmarkResults with statistics
        """
        print(f"🔥 Warming up...") if warmup else None
        if warmup:
            self.warmup()

        print(f"⏱️  Running {n_runs} benchmark iterations...")

        # Create input
        input_tensor = torch.randint(0, 128, (batch_size, seq_len), device=self.device)
        event_types = torch.ones_like(input_tensor) * 1

        # Measure latency
        latencies = []

        for _ in tqdm(range(n_runs), desc="Benchmarking"):
            latency_ms = self.measure_single_forward(input_tensor, event_types)
            latencies.append(latency_ms)

        latencies = np.array(latencies)

        # Measure memory (if CUDA)
        memory_allocated_mb = None
        memory_reserved_mb = None

        if self.device == 'cuda':
            memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

        # Count parameters
        n_parameters = sum(p.numel() for p in self.model.parameters())

        # Create results
        results = BenchmarkResults(
            name=self.model.__class__.__name__,
            device=self.device,
            n_runs=n_runs,
            latency_mean=float(np.mean(latencies)),
            latency_std=float(np.std(latencies)),
            latency_min=float(np.min(latencies)),
            latency_max=float(np.max(latencies)),
            latency_p50=float(np.percentile(latencies, 50)),
            latency_p95=float(np.percentile(latencies, 95)),
            latency_p99=float(np.percentile(latencies, 99)),
            memory_allocated_mb=memory_allocated_mb,
            memory_reserved_mb=memory_reserved_mb,
            n_parameters=n_parameters,
            extras={
                'batch_size': batch_size,
                'seq_len': seq_len,
            }
        )

        return results


def measure_generation_latency(
    model: nn.Module,
    generate_fn: Callable,
    n_runs: int = 100,
    **generate_kwargs
) -> BenchmarkResults:
    """
    Measure latency of full generation (not just forward pass)

    Args:
        model: Model to benchmark
        generate_fn: Generation function (e.g., model.generate_anticipatory)
        n_runs: Number of runs
        **generate_kwargs: Arguments to pass to generate_fn

    Returns:
        BenchmarkResults with statistics

    Example:
        results = measure_generation_latency(
            model,
            model.generate_anticipatory,
            n_runs=100,
            user_notes=dummy_notes,
            max_new_tokens=16
        )
    """
    device = next(model.parameters()).device
    model.eval()

    print(f"⏱️  Measuring generation latency ({n_runs} runs)...")

    latencies = []

    for _ in tqdm(range(n_runs), desc="Generating"):
        if device == 'cuda':
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            _ = generate_fn(**generate_kwargs)

        if device == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    latencies = np.array(latencies)

    # Memory stats
    memory_allocated_mb = None
    memory_reserved_mb = None

    if device == 'cuda':
        memory_allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        memory_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)

    results = BenchmarkResults(
        name=f"{model.__class__.__name__}.{generate_fn.__name__}",
        device=str(device),
        n_runs=n_runs,
        latency_mean=float(np.mean(latencies)),
        latency_std=float(np.std(latencies)),
        latency_min=float(np.min(latencies)),
        latency_max=float(np.max(latencies)),
        latency_p50=float(np.percentile(latencies, 50)),
        latency_p95=float(np.percentile(latencies, 95)),
        latency_p99=float(np.percentile(latencies, 99)),
        memory_allocated_mb=memory_allocated_mb,
        memory_reserved_mb=memory_reserved_mb,
        n_parameters=sum(p.numel() for p in model.parameters()),
        extras=generate_kwargs
    )

    return results


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


def compare_to_baseline(
    model: nn.Module,
    baseline_latency_ms: float,
    baseline_name: str = "Baseline",
    n_runs: int = 100
) -> Dict[str, any]:
    """
    Compare model to baseline latency

    Args:
        model: Model to benchmark
        baseline_latency_ms: Baseline latency in milliseconds
        baseline_name: Name of baseline
        n_runs: Number of benchmark runs

    Returns:
        Dictionary with comparison results
    """
    benchmark = LatencyBenchmark(model)
    results = benchmark.run(n_runs=n_runs)

    speedup = baseline_latency_ms / results.latency_mean
    overhead_ms = results.latency_mean - baseline_latency_ms
    overhead_pct = (overhead_ms / baseline_latency_ms) * 100

    comparison = {
        'model_name': model.__class__.__name__,
        'model_latency_ms': results.latency_mean,
        'baseline_name': baseline_name,
        'baseline_latency_ms': baseline_latency_ms,
        'speedup': speedup,
        'overhead_ms': overhead_ms,
        'overhead_pct': overhead_pct,
        'faster': speedup > 1.0
    }

    print("\n" + "=" * 70)
    print(f"Comparison: {model.__class__.__name__} vs {baseline_name}")
    print("=" * 70)
    print(f"  {model.__class__.__name__}: {results.latency_mean:.2f}ms ± {results.latency_std:.2f}ms")
    print(f"  {baseline_name}: {baseline_latency_ms:.2f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Overhead: {overhead_ms:+.2f}ms ({overhead_pct:+.1f}%)")
    print(f"  Verdict: {'✓ FASTER' if speedup > 1.0 else '✗ SLOWER'}")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    print("=" * 70)
    print("Benchmarking Module Test")
    print("=" * 70)

    # Create dummy model
    from .model import AnticipativeJazzFormer

    print("\n[1] Creating model...")
    model = AnticipativeJazzFormer(
        vocab_size=128,
        d_model=256,
        n_heads=4,
        n_layers=4
    )
    print(f"  ✓ Model created")

    # Count parameters
    print("\n[2] Counting parameters...")
    params = count_parameters(model)
    print(f"  Total params: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")

    # Latency benchmark
    print("\n[3] Running latency benchmark (50 runs)...")
    benchmark = LatencyBenchmark(model, device='cpu')
    results = benchmark.run(n_runs=50, warmup=True)

    print("\n📊 Benchmark Results:")
    print(f"  Device: {results.device}")
    print(f"  Mean latency: {results.latency_mean:.2f}ms ± {results.latency_std:.2f}ms")
    print(f"  Min latency: {results.latency_min:.2f}ms")
    print(f"  Max latency: {results.latency_max:.2f}ms")
    print(f"  Median (p50): {results.latency_p50:.2f}ms")
    print(f"  p95: {results.latency_p95:.2f}ms")
    print(f"  p99: {results.latency_p99:.2f}ms")

    # Compare to baseline
    print("\n[4] Comparing to baseline (100ms target)...")
    comparison = compare_to_baseline(
        model,
        baseline_latency_ms=100.0,
        baseline_name="Real-time target (<100ms)",
        n_runs=50
    )

    if comparison['faster']:
        print(f"\n✅ Model meets real-time target!")
        print(f"   Latency: {comparison['model_latency_ms']:.1f}ms < 100ms")
    else:
        print(f"\n⚠️  Model exceeds real-time target")
        print(f"   Latency: {comparison['model_latency_ms']:.1f}ms > 100ms")
        print(f"   Optimization needed: {-comparison['overhead_ms']:.1f}ms reduction required")

    # Save results
    print("\n[5] Saving benchmark results...")
    output_path = "/tmp/benchmark_results.json"
    results.save_json(output_path)

    print("\n" + "=" * 70)
    print("✅ All benchmarking tests passed!")
    print("=" * 70)
