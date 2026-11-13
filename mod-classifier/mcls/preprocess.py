"""Raw recordings of one dataset in, 20 MS/s window shards and rows out.

    python -m mcls.preprocess --dataset tampere --raw data/raw/tampere --out data/shards/tampere
        [--max-windows-per-burst 64] [--max-windows-per-recording N] [--limit-seconds S]

Processes every recording `mcls.sources.get(dataset, root=raw)` finds. Writes,
into --out: per recording `<stem>.sNNN.npy` + `<stem>.sNNN.rows.jsonl` and
`<stem>.stats.json`, then `preprocess_summary.json`. Formats: mcls/frontend/shards.py.
Exit codes: 0 done, 2 no recordings found, 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from mcls.frontend.process import FrontendParams, process_recording, stem_for


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m mcls.preprocess", description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, help="source name, as mcls.sources.get knows it")
    p.add_argument("--raw", required=True, type=Path, help="folder holding the dataset's raw files")
    p.add_argument("--out", required=True, type=Path, help="output folder (created)")
    p.add_argument("--max-windows-per-burst", type=int, default=FrontendParams.max_windows_per_burst)
    p.add_argument("--max-windows-per-recording", type=int, default=None)
    p.add_argument("--limit-seconds", type=float, default=None,
                   help="process only the first S seconds of each recording (smoke runs)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from mcls.sources import get

    params = FrontendParams(
        max_windows_per_burst=args.max_windows_per_burst,
        max_windows_per_recording=args.max_windows_per_recording,
        limit_seconds=args.limit_seconds,
    )
    source = get(args.dataset, root=args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stats, seen = [], Counter()
    for rec in source.recordings():
        base = stem_for(rec.source_file)
        seen[base] += 1
        stem = base if seen[base] == 1 else f"{base}.r{seen[base] - 1}"
        result = process_recording(rec, args.out, stem, params)
        (args.out / f"{stem}.stats.json").write_text(json.dumps(result, indent=2) + "\n")
        stats.append(result)
        print(f"{rec.source_file}: {result['seconds_processed']:.3f} s, offset {result['slice_offset_hz'] / 1e6:+.2f} MHz, "
              f"{result['bursts']} bursts, {result['windows']} windows, {result['wall_s']:.1f} s wall", flush=True)
    summary = {
        "command": ["python", "-m", "mcls.preprocess", *(argv if argv is not None else sys.argv[1:])],
        "dataset": args.dataset,
        "recordings": len(stats),
        "windows": sum(s["windows"] for s in stats),
        "bursts": sum(s["bursts"] for s in stats),
        "wall_s": round(time.perf_counter() - started, 3),
        "stats": stats,
    }
    (args.out / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not stats:
        print(f"no {args.dataset} recordings found in {args.raw}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
