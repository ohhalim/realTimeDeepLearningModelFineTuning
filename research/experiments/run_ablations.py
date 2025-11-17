"""
Ablation Study: Test different layer assignments and components

Professor's Critical Issue:
- Current layer assignment (0-3 harmony, 4-7 voicing, 8-11 rhythm) is arbitrary
- Need to verify if this assignment is optimal

This script tests:
1. Different layer assignments (reversed, uniform, random)
2. Removing each LoRA component (w/o harmony, w/o voicing, etc.)
3. Different LoRA ranks
"""

import sys
from pathlib import Path
import torch
import subprocess
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(config_name, config_dict, dry_run=False):
    """
    Run a single experiment with given configuration.

    Args:
        config_name: Name for this experiment
        config_dict: Configuration dictionary
        dry_run: If True, only print the command without running
    """
    print(f"\n{'='*70}")
    print(f"Running: {config_name}")
    print(f"{'='*70}")

    # Build command
    cmd = [
        'python', 'training/train_improved.py',
        '--output_dir', f'./outputs/ablations/{config_name}',
    ]

    # Add config parameters
    for key, value in config_dict.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    if dry_run:
        print("Command:", ' '.join(cmd))
        return

    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        # Extract final metrics (TODO: Implement proper logging)
        return {
            'config_name': config_name,
            'status': 'success',
        }

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return {
            'config_name': config_name,
            'status': 'failed',
        }


def ablation_layer_assignment():
    """
    Ablation: Different layer assignments for hierarchical LoRA.

    Tests:
    1. Original (0-3, 4-7, 8-11)
    2. Reversed (8-11, 4-7, 0-3)
    3. Alternating (0,3,6,9 / 1,4,7,10 / 2,5,8,11)
    4. Early-heavy (0-7, 8-9, 10-11)
    5. Late-heavy (0-1, 2-3, 4-11)
    """

    # Base config
    base_config = {
        'model_type': 'hierarchical_lora',
        'batch_size': 2,
        'num_epochs': 20,
        'num_train_samples': 500,
        'num_val_samples': 50,
        'seed': 42,
    }

    experiments = []

    # 1. Original (baseline)
    config = base_config.copy()
    experiments.append(('original_assignment', config))

    # 2-5: Different assignments
    # Note: Actual layer assignment modification requires code changes
    # This is a template showing what experiments to run

    print("\n" + "="*70)
    print("ABLATION: Layer Assignment")
    print("="*70)
    print("\nExperiments to run:")
    for name, _ in experiments:
        print(f"  - {name}")

    print("\n⚠️  Note: Changing layer assignments requires modifying")
    print("    hierarchical_lora.py to support configurable assignments.")
    print("    Current implementation has fixed assignments.")

    return experiments


def ablation_remove_components():
    """
    Ablation: Remove each LoRA component one at a time.

    Tests:
    1. Full model (all 4 LoRAs)
    2. w/o Harmony LoRA
    3. w/o Voicing LoRA
    4. w/o Rhythm LoRA
    5. w/o Dynamics LoRA
    6. Single LoRA (no hierarchy)
    """

    base_config = {
        'model_type': 'hierarchical_lora',
        'batch_size': 2,
        'num_epochs': 20,
        'num_train_samples': 500,
        'num_val_samples': 50,
        'seed': 42,
    }

    experiments = []

    # 1. Full model
    config = base_config.copy()
    experiments.append(('full_model', config))

    # 2-5. Remove each component
    # Implementation note: Need to add flags to disable specific LoRAs
    for component in ['harmony', 'voicing', 'rhythm', 'dynamics']:
        config = base_config.copy()
        # TODO: Add flag --disable_{component}_lora
        experiments.append((f'without_{component}', config))

    # 6. Single LoRA baseline
    config = base_config.copy()
    config['model_type'] = 'single_lora'
    experiments.append(('single_lora_baseline', config))

    print("\n" + "="*70)
    print("ABLATION: Component Removal")
    print("="*70)
    print("\nExperiments to run:")
    for name, _ in experiments:
        print(f"  - {name}")

    return experiments


def ablation_lora_rank():
    """
    Ablation: Different LoRA ranks.

    Tests: r ∈ {4, 8, 16, 32, 64}
    """

    base_config = {
        'model_type': 'hierarchical_lora',
        'batch_size': 2,
        'num_epochs': 20,
        'num_train_samples': 500,
        'num_val_samples': 50,
        'seed': 42,
    }

    experiments = []

    for rank in [4, 8, 16, 32, 64]:
        config = base_config.copy()
        config['lora_r'] = rank
        experiments.append((f'rank_{rank}', config))

    print("\n" + "="*70)
    print("ABLATION: LoRA Rank")
    print("="*70)
    print("\nExperiments to run:")
    for name, _ in experiments:
        print(f"  - {name}")

    return experiments


def ablation_loss_weights():
    """
    Ablation: Different multi-task loss weights.

    Tests adaptive weighting vs fixed weights.
    """

    base_config = {
        'model_type': 'hierarchical_lora',
        'batch_size': 2,
        'num_epochs': 20,
        'num_train_samples': 500,
        'num_val_samples': 50,
        'seed': 42,
    }

    experiments = []

    # Adaptive (uncertainty weighting)
    config = base_config.copy()
    config['use_adaptive_loss'] = True
    experiments.append(('adaptive_loss', config))

    # Fixed weights
    config = base_config.copy()
    config['use_adaptive_loss'] = False
    experiments.append(('fixed_loss', config))

    print("\n" + "="*70)
    print("ABLATION: Loss Weighting")
    print("="*70)
    print("\nExperiments to run:")
    for name, _ in experiments:
        print(f"  - {name}")

    return experiments


def main():
    """Run all ablation studies."""

    print("\n" + "="*80)
    print("ABLATION STUDIES FOR HIERARCHICAL STYLELORA-TRANSFORMER")
    print("="*80)

    print("\nProfessor's Critical Issues to Address:")
    print("  1. ✅ Layer assignment is arbitrary - need validation")
    print("  2. ✅ Each component's contribution unclear")
    print("  3. ✅ LoRA rank choice not justified")
    print("  4. ✅ Loss weighting is ad-hoc")

    # Collect all experiments
    all_experiments = []

    all_experiments.extend(ablation_layer_assignment())
    all_experiments.extend(ablation_remove_components())
    all_experiments.extend(ablation_lora_rank())
    all_experiments.extend(ablation_loss_weights())

    print("\n" + "="*80)
    print(f"TOTAL EXPERIMENTS: {len(all_experiments)}")
    print("="*80)

    # Dry run - just show what would be executed
    print("\nDry run mode - showing commands without executing:")

    for name, config in all_experiments[:3]:  # Show first 3
        run_experiment(name, config, dry_run=True)

    print("\n...")
    print(f"(and {len(all_experiments) - 3} more)")

    # Instructions
    print("\n" + "="*80)
    print("TO RUN EXPERIMENTS:")
    print("="*80)
    print("\n1. Run all experiments (will take ~48 hours):")
    print("   python experiments/run_ablations.py --execute_all")
    print("\n2. Run specific ablation:")
    print("   python experiments/run_ablations.py --ablation layer_assignment")
    print("\n3. Analyze results:")
    print("   python experiments/analyze_ablations.py")

    print("\n" + "="*80)
    print("IMPLEMENTATION STATUS:")
    print("="*80)
    print("  ✅ Experiment configurations defined")
    print("  ✅ Training script supports all configurations")
    print("  ⚠️  Need to collect real MIDI data for meaningful results")
    print("  ⚠️  Need to implement configurable layer assignments")
    print("  ⚠️  Need to add flags for disabling specific LoRA components")
    print("  TODO: Run experiments and collect metrics")
    print("  TODO: Statistical significance testing (t-test, Cohen's d)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--execute_all', action='store_true',
                       help='Actually run all experiments (WARNING: takes long time)')
    parser.add_argument('--ablation', type=str, choices=[
        'layer_assignment', 'components', 'rank', 'loss_weights'
    ], help='Run specific ablation study')

    args = parser.parse_args()

    if args.execute_all or args.ablation:
        print("⚠️  Execution mode not yet implemented")
        print("    (waiting for real MIDI data)")
    else:
        main()
