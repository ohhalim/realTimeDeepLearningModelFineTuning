"""
Adaptive Loss Weighting for Multi-Task Learning

Improvement: Replace fixed loss weights with learned uncertainty-based weighting
Reference: Kendall et al. 2018, "Multi-Task Learning Using Uncertainty to Weigh Losses"

OLD PROBLEM: total_loss = refinement_loss + 0.1 * style_loss (fixed 0.1)
NEW SOLUTION: Learn optimal weights based on task uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import math


class AdaptiveMultiTaskLoss(nn.Module):
    """Learnable multi-task loss weighting via uncertainty

    Based on homoscedastic uncertainty (task-dependent, not data-dependent).

    The key idea:
    - Each task has inherent uncertainty (noise level)
    - High uncertainty tasks should be weighted less
    - Learn log variance for each task
    - Loss = weighted_sum / (2 * sigma²) + log(sigma²)

    This automatically balances tasks with different scales and difficulties.
    """
    def __init__(self, task_names: List[str], init_log_var: float = 0.0):
        """
        Args:
            task_names: Names of tasks (e.g., ['refinement', 'style', 'rhythm'])
            init_log_var: Initial log variance (0.0 → variance=1.0)
        """
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)

        # Learnable log variances (one per task)
        # Using log variance for numerical stability
        self.log_vars = nn.Parameter(torch.full((self.num_tasks,), init_log_var))

        # Task name to index mapping
        self.task_to_idx = {name: i for i, name in enumerate(task_names)}

    def forward(self, losses: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted multi-task loss

        Args:
            losses: Dictionary of task losses {task_name: loss_value}

        Returns:
            total_loss: Weighted sum of losses
            weights: Dictionary of effective weights for logging
        """
        total_loss = 0.0
        weights = {}

        for task_name, loss in losses.items():
            if task_name not in self.task_to_idx:
                # Unknown task, add with weight 1.0
                total_loss = total_loss + loss
                weights[task_name] = 1.0
                continue

            # Get learnable log variance
            idx = self.task_to_idx[task_name]
            log_var = self.log_vars[idx]

            # Compute precision (inverse variance)
            precision = torch.exp(-log_var)

            # Weighted loss: precision * loss + regularization
            # Formula: loss / (2 * sigma²) + 0.5 * log(sigma²)
            #        = 0.5 * precision * loss + 0.5 * log_var
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var

            total_loss = total_loss + weighted_loss

            # Store effective weight for logging
            weights[task_name] = precision.item()

        return total_loss, weights

    def get_weights(self) -> Dict[str, float]:
        """Get current task weights (for logging)"""
        weights = {}
        for name, idx in self.task_to_idx.items():
            precision = torch.exp(-self.log_vars[idx]).item()
            weights[name] = precision
        return weights

    def get_log_vars(self) -> Dict[str, float]:
        """Get current log variances (for analysis)"""
        log_vars = {}
        for name, idx in self.task_to_idx.items():
            log_vars[name] = self.log_vars[idx].item()
        return log_vars


class MultiTaskTrainer:
    """Trainer with adaptive multi-task loss"""
    def __init__(self,
                 model: nn.Module,
                 task_names: List[str],
                 learning_rate: float = 3e-4):
        self.model = model
        self.task_names = task_names

        # Adaptive loss
        self.loss_fn = AdaptiveMultiTaskLoss(task_names)

        # Optimizer (include loss_fn parameters!)
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=learning_rate
        )

        # Loss history
        self.loss_history = {name: [] for name in task_names}
        self.weight_history = {name: [] for name in task_names}

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step with adaptive weighting

        Args:
            batch: Training batch

        Returns:
            Dictionary of losses and weights
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass (model should return dict of losses)
        outputs = self.model(batch)

        # Extract task losses
        task_losses = {
            name: outputs[f'{name}_loss']
            for name in self.task_names
            if f'{name}_loss' in outputs
        }

        # Adaptive weighting
        total_loss, current_weights = self.loss_fn(task_losses)

        # Backward
        total_loss.backward()
        self.optimizer.step()

        # Log
        log_dict = {
            'total_loss': total_loss.item(),
        }

        for name, loss in task_losses.items():
            log_dict[f'{name}_loss'] = loss.item()
            log_dict[f'{name}_weight'] = current_weights.get(name, 1.0)
            log_dict[f'{name}_log_var'] = self.loss_fn.get_log_vars().get(name, 0.0)

            # Update history
            self.loss_history[name].append(loss.item())
            self.weight_history[name].append(current_weights.get(name, 1.0))

        return log_dict

    def print_weight_status(self):
        """Print current adaptive weights"""
        print("\n" + "="*60)
        print("Adaptive Loss Weights Status")
        print("="*60)
        print(f"{'Task':<20} {'Weight':<12} {'Log Var':<12} {'Recent Loss':<12}")
        print("-"*60)

        weights = self.loss_fn.get_weights()
        log_vars = self.loss_fn.get_log_vars()

        for name in self.task_names:
            weight = weights.get(name, 1.0)
            log_var = log_vars.get(name, 0.0)

            # Recent average loss
            if self.loss_history[name]:
                recent_loss = sum(self.loss_history[name][-100:]) / len(self.loss_history[name][-100:])
            else:
                recent_loss = 0.0

            print(f"{name:<20} {weight:<12.4f} {log_var:<12.4f} {recent_loss:<12.4f}")

        print("="*60)
        print("\nInterpretation:")
        print("  - Weight = 1/sigma² (higher = more important)")
        print("  - Log Var = log(sigma²) (higher = more uncertain)")
        print("  - High weight → task is precise, should be emphasized")
        print("  - Low weight → task is noisy, should be de-emphasized")
        print("="*60 + "\n")


def test_adaptive_loss_weighting():
    """Test adaptive loss weighting"""
    print("Testing Adaptive Multi-Task Loss Weighting...")

    # Create simple multi-task losses
    task_names = ['refinement', 'style', 'rhythm']
    loss_fn = AdaptiveMultiTaskLoss(task_names)

    print(f"\n1. Initial state:")
    print(f"   Log vars: {loss_fn.get_log_vars()}")
    print(f"   Weights: {loss_fn.get_weights()}")

    # Simulate training
    print(f"\n2. Simulating training with different task difficulties...")

    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.01)

    for step in range(100):
        # Simulate task losses with different scales
        losses = {
            'refinement': torch.tensor(2.0 + 0.1 * torch.randn(1)).requires_grad_(True),  # Low variance
            'style': torch.tensor(5.0 + 2.0 * torch.randn(1)).requires_grad_(True),      # High variance
            'rhythm': torch.tensor(1.0 + 0.5 * torch.randn(1)).requires_grad_(True),     # Medium variance
        }

        # Compute weighted loss
        total_loss, weights = loss_fn(losses)

        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Print every 20 steps
        if (step + 1) % 20 == 0:
            print(f"\n   Step {step+1}:")
            print(f"   Total loss: {total_loss.item():.4f}")
            for name, weight in weights.items():
                print(f"   {name}: loss={losses[name].item():.4f}, weight={weight:.4f}")

    print(f"\n3. Final weights (after adaptation):")
    final_weights = loss_fn.get_weights()
    final_log_vars = loss_fn.get_log_vars()

    for name in task_names:
        print(f"   {name}:")
        print(f"      Weight: {final_weights[name]:.4f}")
        print(f"      Log var: {final_log_vars[name]:.4f}")
        print(f"      Interpretation: ", end="")

        if final_weights[name] > 2.0:
            print("HIGH precision → emphasize this task")
        elif final_weights[name] < 0.5:
            print("LOW precision → de-emphasize this task")
        else:
            print("MEDIUM precision → balanced importance")

    print(f"\n4. Verify adaptation:")
    print(f"   Expected: 'style' should have lowest weight (highest variance)")
    print(f"   Expected: 'refinement' should have highest weight (lowest variance)")
    print(f"   Actual: refinement={final_weights['refinement']:.2f}, "
          f"style={final_weights['style']:.2f}, rhythm={final_weights['rhythm']:.2f}")

    if final_weights['refinement'] > final_weights['style']:
        print(f"   ✅ Correct adaptation!")
    else:
        print(f"   ❌ Weights didn't adapt properly")

    print("\n✅ Test completed!")


if __name__ == '__main__':
    test_adaptive_loss_weighting()
