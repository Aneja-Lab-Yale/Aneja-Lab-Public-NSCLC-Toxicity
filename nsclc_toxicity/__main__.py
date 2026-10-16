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
    demo = sub.add_parser("download-demo-ct", help="Download the real public 3D Slicer chest CT and an example clinical manifest")
    demo.add_argument("--output", type=Path, default=Path("data/public_ct"))
    raw = sub.add_parser("infer-ct", help="Run raw CT plus clinical data, including automatic lung-lobe segmentation")
    raw.add_argument("--manifest", type=Path, required=True)
    raw.add_argument("--weights", type=Path, default=Path("weights"))
    raw.add_argument("--output", type=Path, default=Path("results/ct_inference"))
    raw.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    raw.add_argument("--full-resolution", action="store_true", help="Use 1.5 mm segmentation instead of the faster 3 mm default")
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
    prepare.add_argument("--auto-masks", action="store_true", help="Generate lung lobes from each baseline CT")
    prepare.add_argument("--full-resolution", action="store_true", help="Use 1.5 mm instead of 3 mm automatic segmentation")
    train = sub.add_parser("train-ct", help="Retrain DINOv3 within each patient fold on your CT dataset")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--output", type=Path, default=Path("results/ct_training"))
    train.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    train.add_argument("--seed", type=int, default=20260714)
    train.add_argument("--initial-backbone", type=Path, help="Optional local timm DINOv3 pretrained state")
    train.add_argument("--auto-masks", action="store_true", help="Generate baseline lung lobes before retraining")
    train.add_argument("--full-resolution", action="store_true", help="Use 1.5 mm instead of 3 mm automatic segmentation")
    args = parser.parse_args()
    try:
        if args.command == "download-weights":
            from .download import download_weights
            download_weights(args.output, imaging=args.imaging)
        elif args.command == "download-demo-ct":
            from .public_ct import download_demo
            download_demo(args.output)
        elif args.command == "infer-ct":
            from .imaging import infer_ct
            infer_ct(args.manifest, args.weights, args.output, args.device, fast=not args.full_resolution)
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
            prepare_ct(args.manifest, args.weights, args.output, args.device, auto_masks=args.auto_masks, fast=not args.full_resolution)
        elif args.command == "train-ct":
            from .imaging import train_ct
            train_ct(args.manifest, args.output, args.device, args.seed, args.initial_backbone,
                     auto_masks=args.auto_masks, fast=not args.full_resolution)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
