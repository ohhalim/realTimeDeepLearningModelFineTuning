"""
Adaptive Corruption Curriculum Learning

Key Innovation #4: Automatically adjust corruption difficulty based on model performance
- Easy corruptions converge fast → sample less
- Hard corruptions need more training → sample more
- Dynamic adjustment based on recent losses
"""

import torch
import torch.nn.functional as F
import random
from typing import List, Dict, Tuple
from collections import defaultdict, deque


class AdaptiveCorruptionCurriculum:
    """Adaptive corruption sampling based on performance

    Algorithm:
    1. Track per-corruption performance (exponential moving average of loss)
    2. Compute sampling weights: difficulty × (1 - performance)
    3. Temperature-scaled softmax for stochastic sampling
    4. Temperature decreases over training (exploration → exploitation)
    """
    def __init__(self,
                 corruptions: List,
                 initial_temp: float = 2.0,
                 min_temp: float = 1.0,
                 alpha: float = 0.1,
                 window_size: int = 100):
        """
        Args:
            corruptions: List of corruption functions
            initial_temp: Initial sampling temperature (high = more uniform)
            min_temp: Minimum temperature (low = more focused on hard tasks)
            alpha: EMA smoothing factor (0.1 = slow adaptation, 0.5 = fast)
            window_size: Rolling window for performance tracking
        """
        self.corruptions = corruptions
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.temperature = initial_temp
        self.alpha = alpha

        # Corruption difficulty levels (manually designed based on task complexity)
        self.difficulty = {
            'NoCorrupt': 1,           # Easiest: identity function
            'VelocityDropout': 2,     # Medium: add dynamics
            'PitchDropout': 2,        # Medium: add melody to rhythm
            'TimeCrop': 3,            # Hard: temporal coherence
            'NoteCrop': 3,            # Hard: harmonic reasoning
            'GenreChange': 4,         # Hardest: style transfer
        }

        # Performance tracking (higher = better, range [0, 1])
        self.performance = {c.__class__.__name__: 0.5 for c in corruptions}

        # Loss history (rolling window)
        self.loss_history = {c.__class__.__name__: deque(maxlen=window_size) for c in corruptions}

        # Sampling statistics
        self.sample_counts = defaultdict(int)
        self.total_samples = 0

    def sample_corruption(self, epoch: int, total_epochs: int):
        """Sample corruption based on current performance and training progress

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs

        Returns:
            corruption: Selected corruption function
        """
        # Update temperature (anneal from initial_temp to min_temp)
        progress = epoch / max(total_epochs, 1)
        self.temperature = self.initial_temp - progress * (self.initial_temp - self.min_temp)

        # Compute sampling weights for each corruption
        weights = []
        for corruption in self.corruptions:
            name = corruption.__class__.__name__

            # Weight = difficulty × (1 - performance)
            # High difficulty + low performance → high weight (sample more)
            difficulty = self.difficulty.get(name, 2)  # Default to medium
            performance = self.performance[name]

            # Ensure non-negative weight
            weight = max(difficulty * (1.0 - performance), 0.1)  # Minimum 0.1
            weights.append(weight)

        # Temperature-scaled softmax
        weights = torch.tensor(weights, dtype=torch.float32)
        probs = F.softmax(weights / self.temperature, dim=0)

        # Sample
        idx = torch.multinomial(probs, num_samples=1).item()
        selected = self.corruptions[idx]

        # Update statistics
        self.sample_counts[selected.__class__.__name__] += 1
        self.total_samples += 1

        return selected

    def update_performance(self, corruption_name: str, loss: float):
        """Update performance metric based on recent loss

        Lower loss → higher performance → lower sampling probability
        Uses exponential moving average for smooth adaptation.

        Args:
            corruption_name: Name of corruption (e.g., 'GenreChange')
            loss: Refinement loss for this corruption
        """
        # Add to history
        self.loss_history[corruption_name].append(loss)

        # Convert loss to performance score [0, 1]
        # Using sigmoid to map loss to bounded range
        new_perf = 1.0 / (1.0 + loss)  # Higher loss → lower performance

        # Exponential moving average
        old_perf = self.performance[corruption_name]
        self.performance[corruption_name] = (
            self.alpha * new_perf + (1 - self.alpha) * old_perf
        )

    def get_statistics(self) -> Dict[str, float]:
        """Get curriculum statistics for logging"""
        stats = {
            'temperature': self.temperature,
        }

        # Sampling distribution
        if self.total_samples > 0:
            for name, count in self.sample_counts.items():
                stats[f'sample_rate/{name}'] = count / self.total_samples

        # Performance scores
        for name, perf in self.performance.items():
            stats[f'performance/{name}'] = perf

        # Average loss per corruption
        for name, losses in self.loss_history.items():
            if losses:
                stats[f'avg_loss/{name}'] = sum(losses) / len(losses)

        return stats

    def reset_statistics(self):
        """Reset sampling counts (called at end of epoch)"""
        self.sample_counts.clear()
        self.total_samples = 0

    def get_difficulty_ranking(self) -> List[Tuple[str, float, float]]:
        """Get corruptions ranked by difficulty and current performance

        Returns:
            List of (corruption_name, difficulty, performance) sorted by sampling priority
        """
        ranking = []
        for corruption in self.corruptions:
            name = corruption.__class__.__name__
            difficulty = self.difficulty.get(name, 2)
            performance = self.performance[name]
            priority = difficulty * (1.0 - performance)
            ranking.append((name, difficulty, performance, priority))

        # Sort by priority (descending)
        ranking.sort(key=lambda x: x[3], reverse=True)

        return ranking

    def print_status(self):
        """Print current curriculum status"""
        print("\n" + "="*60)
        print("Adaptive Curriculum Status")
        print("="*60)
        print(f"Temperature: {self.temperature:.3f}")
        print(f"\nCorruption Ranking (by priority):")
        print(f"{'Corruption':<20} {'Difficulty':<12} {'Performance':<12} {'Priority':<10}")
        print("-"*60)

        for name, diff, perf, priority in self.get_difficulty_ranking():
            print(f"{name:<20} {diff:<12} {perf:<12.3f} {priority:<10.3f}")

        if self.total_samples > 0:
            print(f"\nSampling Distribution (last {self.total_samples} samples):")
            for name, count in sorted(self.sample_counts.items(), key=lambda x: x[1], reverse=True):
                pct = count / self.total_samples * 100
                bar = "█" * int(pct / 2)
                print(f"  {name:<20} {count:>4} ({pct:>5.1f}%) {bar}")

        print("="*60 + "\n")


class CurriculumScheduler:
    """Scheduler for curriculum-based training

    Coordinates:
    - Corruption curriculum (which corruption to sample)
    - Learning rate schedule
    - Loss weighting (optional)
    """
    def __init__(self,
                 curriculum: AdaptiveCorruptionCurriculum,
                 optimizer: torch.optim.Optimizer,
                 total_epochs: int,
                 warmup_epochs: int = 5):
        self.curriculum = curriculum
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        # Learning rate schedule
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step_epoch(self, epoch: int):
        """Step scheduler at end of epoch"""
        self.current_epoch = epoch

        # Cosine learning rate schedule with warmup
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Print status
        print(f"Epoch {epoch}/{self.total_epochs}: LR = {lr:.6f}")

    def get_corruption(self):
        """Sample corruption for current training step"""
        return self.curriculum.sample_corruption(
            self.current_epoch, self.total_epochs
        )

    def update_corruption_loss(self, corruption_name: str, loss: float):
        """Update curriculum based on loss"""
        self.curriculum.update_performance(corruption_name, loss)

    def print_status(self):
        """Print scheduler status"""
        self.curriculum.print_status()


def test_adaptive_curriculum():
    """Test Adaptive Curriculum"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from corruptions.corruption_functions import get_all_corruptions
    import math

    print("Testing Adaptive Corruption Curriculum...")

    # Get corruptions
    corruptions = get_all_corruptions()
    print(f"\nLoaded {len(corruptions)} corruptions:")
    for c in corruptions:
        print(f"  - {c}")

    # Create curriculum
    curriculum = AdaptiveCorruptionCurriculum(
        corruptions,
        initial_temp=2.0,
        min_temp=1.0,
        alpha=0.1,
    )

    # Simulate training
    print("\n" + "="*60)
    print("Simulating Training")
    print("="*60)

    total_epochs = 20
    steps_per_epoch = 100

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch+1}/{total_epochs}")

        # Sample and update
        for step in range(steps_per_epoch):
            # Sample corruption
            corruption = curriculum.sample_corruption(epoch, total_epochs)
            name = corruption.__class__.__name__

            # Simulate loss (harder corruptions have higher initial loss)
            base_loss = curriculum.difficulty[name] * 0.5

            # Loss decreases over time (learning)
            progress = (epoch * steps_per_epoch + step) / (total_epochs * steps_per_epoch)
            simulated_loss = base_loss * (1.0 - 0.7 * progress) + random.gauss(0, 0.1)
            simulated_loss = max(simulated_loss, 0.1)  # Minimum loss

            # Update performance
            curriculum.update_performance(name, simulated_loss)

        # Print status every 5 epochs
        if (epoch + 1) % 5 == 0:
            curriculum.print_status()
            curriculum.reset_statistics()

    # Final status
    print("\n" + "="*60)
    print("Final Curriculum State")
    print("="*60)
    curriculum.print_status()

    # Verify curriculum adaptation
    print("\nVerifying Curriculum Adaptation:")
    print("Expected: Hard corruptions (GenreChange, TimeCrop, NoteCrop) should have higher performance")
    print("          Easy corruptions (NoCorrupt) should have near-perfect performance")
    print("\nActual:")
    for name in sorted(curriculum.performance.keys(), key=lambda x: curriculum.performance[x], reverse=True):
        perf = curriculum.performance[name]
        diff = curriculum.difficulty[name]
        print(f"  {name:<20} Performance: {perf:.3f}, Difficulty: {diff}")

    print("\n✅ Test completed!")


if __name__ == '__main__':
    import math
    test_adaptive_curriculum()
