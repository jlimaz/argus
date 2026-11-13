"""Build data/manifest.jsonl from the front end's per-shard rows.

    python -m mcls.dataset.build_manifest --shards DIR [--shards DIR ...] \
        --out data/manifest.jsonl --seed S [--cross-test DATASET ...] [--cap-per-class N] \
        [--holdout-emitter ID ...] [--holdout-split generalization] [--classes v1|v2|<subset>|all]

Collects rows, keeps only labels in the class set (default v1) and reports
the dropped ones, caps each class per dataset (PLAN 3.6), holds out whole
emitters into `generalization` (PLAN 3.5), splits by recording stratified by
class (`mcls.manifest.assign_splits_stratified`), checks id leakage and
identical-sample leakage, writes the manifest sorted by window_id with shard
paths relative to its own directory, and prints its sha256
(`mcls.manifest.digest`).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from mcls import manifest, taxonomy
from mcls.dataset.rows import DEFAULT_CAP_PER_CLASS, HOLDOUT_SPLIT, build_with_report, relative_to
from mcls.dataset.store import WindowStore, audit_window_hashes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m mcls.dataset.build_manifest", description=__doc__.split("\n\n")[0])
    parser.add_argument("--shards", type=Path, action="append", required=True, help="front-end output directory (repeatable)")
    parser.add_argument("--out", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cross-test", nargs="*", default=[], metavar="DATASET", help="datasets held out whole as cross_test")
    parser.add_argument("--cap-per-class", type=int, default=DEFAULT_CAP_PER_CLASS, help="per class per dataset; 0 = no cap")
    parser.add_argument("--holdout-emitter", nargs="*", default=[], metavar="ID", help="emitter_id(s) held out whole")
    parser.add_argument("--holdout-split", default=HOLDOUT_SPLIT, choices=("generalization", "test", "cross_test"),
                        help="split that held-out emitters go to (default generalization)")
    parser.add_argument("--holdout-match", choices=("emitter", "recording"), default="emitter",
                        help="match emitter_id exactly, or a substring of recording_id")
    parser.add_argument("--classes", default="v1",
                        help="keep only labels in this class set from configs/taxonomy.toml: v1, v2, a named "
                             "subset such as v2a, or all (keep every label)")
    parser.add_argument("--no-hash-audit", action="store_true", help="skip hashing window samples")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        classes = None if args.classes == "all" else taxonomy.load().classes(args.classes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        rows, report = build_with_report(args.shards, args.seed, args.cross_test, args.cap_per_class,
                                         args.holdout_emitter, args.holdout_split, args.holdout_match, classes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for label, count in report.dropped_labels.items():
        print(f"dropped label {label!r} (not in {args.classes}): {count} windows", file=sys.stderr)
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not rows:
        print("error: no rows left to write", file=sys.stderr)
        return 1
    if not args.no_hash_audit:
        leaks = audit_window_hashes(rows, WindowStore())
        if leaks:
            for digest, members in leaks[:10]:
                print(f"identical window across splits: {digest[:16]} {members}", file=sys.stderr)
            print(f"error: {len(leaks)} identical windows appear in more than one split", file=sys.stderr)
            return 1
    out_rows = relative_to(rows, args.out.resolve().parent)
    try:
        count = manifest.write(args.out, out_rows, classes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for split, n in sorted(Counter(row.split for row in rows).items()):
        print(f"  {split}: {n}")
    print(f"manifest: {args.out} rows={count} sha256={manifest.digest(args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
