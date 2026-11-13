"""Merge the per-task JSONL files a CRC array job wrote into one training file.

    python finetune/crc/merge.py nist_radar      # finetune/out/nist_radar/*.jsonl -> data/train/nist_radar.jsonl
"""

import sys
from pathlib import Path

from finetune.sources.common import ROOT, TRAIN, read_jsonl, write_jsonl


def main(name: str) -> None:
    parts = sorted((ROOT / "finetune" / "out" / name).glob("*.jsonl"))
    if not parts:
        raise SystemExit(f"nothing to merge in finetune/out/{name}/")
    rows = [row for part in parts for row in read_jsonl(part)]
    write_jsonl(TRAIN / f"{name}.jsonl", rows)
    print(f"merged {len(parts)} files, {len(rows)} rows -> data/train/{name}.jsonl")


if __name__ == "__main__":
    main(sys.argv[1])
