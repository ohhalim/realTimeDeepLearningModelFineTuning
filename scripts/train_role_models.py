#!/usr/bin/env python3
"""
Train role-specific Music Transformer models sequentially.

Example:
  python scripts/train_role_models.py \
    --base_config configs/pytorch_transformer_config.yaml \
    --roles lead,accompaniment,call_response \
    --data_root data/roles \
    --output_root models/finetuned/pytorch_transformer_roles
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_roles(raw_roles: str):
    roles = [item.strip() for item in raw_roles.split(",") if item.strip()]
    if not roles:
        raise ValueError("At least one role must be provided.")
    return roles


def main():
    parser = argparse.ArgumentParser(description="Train role-specific models")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/pytorch_transformer_config.yaml",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="lead,accompaniment,call_response",
        help="Comma-separated role names",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/roles",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="models/finetuned/pytorch_transformer_roles",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for all roles",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands only",
    )
    args = parser.parse_args()

    roles = parse_roles(args.roles)
    root = Path(__file__).resolve().parent.parent
    train_script = root / "scripts" / "train_pytorch_transformer.py"

    for role in roles:
        role_data_dir = Path(args.data_root) / role
        if not role_data_dir.exists():
            print(f"[SKIP] role data not found: {role_data_dir}")
            continue

        cmd = [
            sys.executable,
            str(train_script),
            "--config",
            args.base_config,
            "--use_role_dataset",
            "--role",
            role,
            "--midi_dir",
            str(role_data_dir),
            "--output_dir",
            args.output_root,
        ]
        if args.epochs is not None:
            cmd.extend(["--epochs", str(args.epochs)])

        print(f"[RUN] role={role}")
        print(" ".join(cmd))

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Training failed for role={role} (code={result.returncode})")

    print("All requested role trainings completed.")


if __name__ == "__main__":
    main()
