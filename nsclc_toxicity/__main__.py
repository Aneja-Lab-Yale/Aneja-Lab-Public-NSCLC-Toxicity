"""Small command-line entry point. Run with --help to see all commands."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="NSCLC SBRT pulmonary toxicity research models")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-weights", help="Download and verify the final model weights")
    download.add_argument("--output", type=Path, default=Path("weights"))
    download.add_argument("--imaging", action="store_true", help="Also download the three CT checkpoints")
    predict = sub.add_parser("predict", help="Run the released ensemble on a prepared CSV")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--weights", type=Path, default=Path("weights/final_model.json"))
    predict.add_argument("--output", type=Path, default=Path("results/predictions.csv"))
    experiment = sub.add_parser("experiment", help="Fit and evaluate the selected branches on your data")
    experiment.add_argument("--input", type=Path, required=True)
    experiment.add_argument("--output", type=Path, default=Path("results/experiment"))
    experiment.add_argument("--seed", type=int, default=20260720)
    prepare = sub.add_parser("prepare-ct", help="Calculate Q7 and DINOv3 inputs from CTs and lobe masks")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--weights", type=Path, default=Path("weights"))
    prepare.add_argument("--output", type=Path, default=Path("results/prepared.csv"))
    prepare.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train = sub.add_parser("train-ct", help="Retrain DINOv3 within each patient fold on your CT dataset")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("results/ct_training"))
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--seed", type=int, default=20260714)
    train.add_argument("--initial-backbone", type=Path, help="Optional local timm DINOv3 pretrained state")
    args = parser.parse_args()
    try:
        if args.command == "download-weights":
            from .download import download_weights
            download_weights(args.output, imaging=args.imaging)
        elif args.command == "predict":
            from .model import load_model, predict
            result = predict(pd.read_csv(args.input), load_model(args.weights))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"Saved {len(result)} scores to {args.output}")
        elif args.command == "experiment":
            from .training import run_experiment
            report = run_experiment(pd.read_csv(args.input), args.output, args.seed)
            print(f"Completed: {report['patients']} patients, {report['scans']} scans. Results: {args.output}")
        elif args.command == "prepare-ct":
            from .imaging import prepare_ct
            prepare_ct(args.manifest, args.weights, args.output, args.device)
        elif args.command == "train-ct":
            from .imaging import train_ct
            train_ct(args.manifest, args.output, args.device, args.seed, args.initial_backbone)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
