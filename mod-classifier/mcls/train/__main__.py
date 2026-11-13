"""python -m mcls.train --config configs/X.toml [--seed N] [--device D] [--resume] [--out-dir DIR] [--name NAME]

Validates the config, creates runs/<name>/ with config.toml and meta.json, then
hands over by model name:

- softmax_regression, lightgbm, knn, majority -> mcls.baselines.train(config, run_dir)  (Engineer 6)
- cnn1d, cnn2d, fused                         -> mcls.train.loop.fit(config, run_dir, resume=...)

Neither target is imported until it is needed, so this module never imports
torch. Exit codes: 0 done, 2 bad config or run folder.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from mcls import config as config_module
from mcls.models import NEURAL_MODELS
from mcls.train import runs

BASELINE_MODELS = ("softmax_regression", "lightgbm", "knn", "majority")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m mcls.train", description="Train one model from a TOML config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, help="override run.seed; the run name gets a _s<seed> suffix")
    parser.add_argument("--device", choices=config_module.DEVICES, help="override run.device")
    parser.add_argument("--out-dir", help="override run.out_dir")
    parser.add_argument("--name", help="override run.name")
    parser.add_argument("--resume", action="store_true", help="continue the run in its existing folder")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = config_module.load(args.config)
        config = runs.apply_overrides(config, seed=args.seed, device=args.device, out_dir=args.out_dir, name=args.name)
        config_module.validate(config, args.config.name)
        if args.resume and config["model"]["name"] in BASELINE_MODELS:
            raise ValueError("--resume applies to neural models only")
        classes = runs.class_list(config)
        run_dir = runs.run_dir_for(config)
        runs.prepare(config, run_dir, classes, resume=args.resume)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"mcls.train: {exc}", file=sys.stderr)
        return 2
    model = config["model"]["name"]
    print(f"mcls.train: {model} -> {run_dir}")
    if model in BASELINE_MODELS:
        importlib.import_module("mcls.baselines").train(config, run_dir)
    elif model in NEURAL_MODELS:
        importlib.import_module("mcls.train.loop").fit(config, run_dir, resume=args.resume)
    else:  # config.validate already rejects this; kept so a new MODELS entry cannot pass silently
        print(f"mcls.train: no trainer for model {model!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
