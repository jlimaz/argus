"""python -m mcls.baselines --config configs/baseline.toml --manifest PATH --out runs/<name>

Fits the config's baseline on the train split, selects on val and writes the run
folder. Runs on CRC with real data; on the laptop only on tiny test fixtures.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mcls import config as config_module
from mcls.baselines.models import MODELS
from mcls.baselines.run import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mcls.baselines", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="default: data.manifest from the config")
    parser.add_argument("--out", type=Path, help="run folder (default: run.out_dir/run.name)")
    parser.add_argument("--model", choices=sorted(MODELS), help="override model.name")
    parser.add_argument("--seed", type=int, help="override run.seed")
    parser.add_argument("--overwrite", action="store_true", help="empty the run folder first if it holds another run")
    args = parser.parse_args(argv)
    try:
        config = config_module.load(args.config)
        if args.model:
            config["model"]["name"] = args.model
        if args.seed is not None:
            config["run"]["seed"] = args.seed
        if config["model"]["name"] not in MODELS:
            raise ValueError(f"model.name {config['model']['name']!r} is not a baseline (choose from {sorted(MODELS)})")
        manifest = args.manifest or Path(config["data"]["manifest"])
        if not manifest.is_file():
            raise ValueError(f"manifest not found: {manifest}")
        out = args.out or Path(config["run"]["out_dir"]) / config["run"]["name"]
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        train(config, out, manifest=manifest, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"run: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
