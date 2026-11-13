"""Gather the per-file outputs of crc/prepare_array.job into one shards dir for build_manifest.

    python scripts/merge_shards.py --prepared ~/mcls_data/prepared/tampere \
        --file-list runs/file_lists/files_tampere.txt --dest ~/mcls_data/shards
    python -m mcls.dataset.build_manifest --shards ~/mcls_data/shards --out data/manifest.jsonl --seed 1

prepare_array.job writes one folder per raw file, <prepared>/<key>/, with a DONE marker
when the task finished (key = the catalog name's base name, unique per dataset; mcls.acquire enforces that). This script:

- takes only folders with DONE; ignores the hidden .<key>.partial.* staging folders of
  running or killed tasks, which build_manifest's recursive *.jsonl search would otherwise read;
- places each one, unchanged, at <dest>/<prepared folder name>/<key>/, so relative `shard`
  paths inside its .jsonl files stay valid (build_manifest resolves them against the
  .jsonl's folder). Files are hard-linked (no extra space) or copied across filesystems;
  --move renames the folders instead;
- with --file-list, lists every file that has no finished output, and exits 1 unless
  --allow-missing (resubmit just those tasks: a rerun of the array job skips DONE files).

Rerunning is safe: a folder already merged with the same files is skipped; one that
differs is an error. Standard library only.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path

DONE = "DONE"


def key_for(name: str) -> str:
    """Output folder name for a catalog file name: its base name (must match prepare_array.job)."""
    return name.rsplit("/", 1)[-1]


def read_list(path: Path) -> list[str]:
    lines = (line.strip("\r\n") for line in path.read_text().splitlines())
    return [line for line in lines if line.strip() and not line.startswith("#")]


def finished(prepared: Path) -> tuple[list[Path], list[Path]]:
    """(folders with DONE, other non-hidden folders) directly under prepared."""
    done, unfinished = [], []
    for child in sorted(prepared.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        (done if (child / DONE).is_file() else unfinished).append(child)
    return done, unfinished


def same_tree(a: Path, b: Path) -> bool:
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    if files_a != files_b:
        return False
    return all(filecmp.cmp(a / f, b / f, shallow=False) for f in files_a if f.name != DONE)


def link_tree(src: Path, dest: Path) -> None:
    """Hard-link every file of src into dest (copy when linking is not possible)."""
    stage = dest.with_name(f".{dest.name}.merging")
    if stage.exists():
        shutil.rmtree(stage)
    for path in sorted(src.rglob("*")):
        target = stage / path.relative_to(src)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
    stage.mkdir(parents=True, exist_ok=True)
    stage.rename(dest)


def merge(prepared_dirs: list[Path], dest: Path, move: bool = False) -> dict:
    """Merge finished folders. Returns counts and the keys seen, for reporting."""
    report = {"merged": 0, "skipped": 0, "unfinished": [], "no_jsonl": [], "keys": set()}
    dest.mkdir(parents=True, exist_ok=True)
    for prepared in prepared_dirs:
        if not prepared.is_dir():
            raise FileNotFoundError(f"no prepared folder {prepared}")
        group = dest / prepared.name
        group.mkdir(exist_ok=True)
        done, unfinished = finished(prepared)
        report["unfinished"] += unfinished
        for folder in done:
            report["keys"].add(folder.name)
            if not any(folder.rglob("*.jsonl")):
                report["no_jsonl"].append(folder)
            target = group / folder.name
            if target.exists():
                if not same_tree(folder, target):
                    raise FileExistsError(f"{target} exists and differs from {folder}; remove one by hand")
                report["skipped"] += 1
                continue
            if move:
                folder.rename(target)
            else:
                link_tree(folder, target)
            report["merged"] += 1
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared", type=Path, action="append", required=True,
                        help="a prepare_array.job MCLS_OUT folder; repeatable")
    parser.add_argument("--dest", type=Path, required=True, help="the shards dir for build_manifest --shards")
    parser.add_argument("--file-list", type=Path, action="append", default=[],
                        help="the MCLS_FILE_LIST used; repeatable (one per --prepared)")
    parser.add_argument("--move", action="store_true", help="rename folders instead of hard-linking")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = merge(args.prepared, args.dest, args.move)
    except (FileNotFoundError, FileExistsError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"merged {report['merged']}, already there {report['skipped']}, into {args.dest}")
    for folder in report["unfinished"]:
        print(f"unfinished (no {DONE}, not merged): {folder}")
    for folder in report["no_jsonl"]:
        print(f"WARNING: no .jsonl rows in {folder}; build_manifest will find no windows there")

    missing = []
    for path in args.file_list:
        missing += [n for n in read_list(path) if key_for(n) not in report["keys"]]
    for name in missing:
        print(f"missing: {name}")
    if missing:
        print(f"{len(missing)} file(s) have no finished output; rerun the array job (DONE files are skipped)")
        if not args.allow_missing:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
