"""Fetch dataset files listed in configs/catalog.toml.

    python -m mcls.acquire --dataset tampere --list
    python -m mcls.acquire --dataset tampere --only DJI_mavic_mini_2G.bin --dry-run
    python -m mcls.acquire --dataset tampere --only DJI_mavic_mini_2G.bin --connections 4
    python -m mcls.acquire --dataset nist_wifi_bt --max-gb 25

Where files go: `$MCLS_RAW/<file>` when MCLS_RAW is set (a CRC task's node
scratch, see crc/prepare_array.job), else `data/raw/<dataset>/<file>`. Only the
file's base name is used, so a catalog name with folders
(`SimulatedRadarWaveforms/Group1/x.mat`) lands as `x.mat`.

Connections: default 1, never more than 8, and never more than the host's
`max_connections` in the catalog. Fairdata ignores Range and its download
tokens are single-use, so it always gets one stream and a fresh token per
attempt.

Exit codes: 0 done, 1 download or checksum failed, 2 refused (unknown dataset or
file, access not direct, file list unknown, over --max-gb).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath

from mcls import fetch

CATALOG = Path(__file__).resolve().parents[1] / "configs" / "catalog.toml"
DEFAULT_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"  # same as mcls.sources.base.DEFAULT_RAW
MAX_CONNECTIONS = fetch.MAX_CONNECTIONS
FETCHABLE = {"direct", "fairdata"}
ACCESS_TYPES = FETCHABLE | {"ieee_login", "manual"}
DATASET_KEYS = ("title", "page", "host", "access", "licence", "files_status", "files_note")
FILE_KEYS = ("name", "size", "checksum", "url")
ROLES = ("data", "metadata", "doc")  # optional per-file `role`; data (a recording) is the default
FAIRDATA_AUTHORIZE = "https://etsin.fairdata.fi/api/v3/download/authorize"


class Refused(Exception):
    """A request mcls.acquire will not carry out (exit code 2)."""


def load_catalog(path: Path | None = None) -> dict:
    """Parse and check the catalog. Returns {"hosts": {...}, "datasets": {...}}."""
    with open(path or CATALOG, "rb") as source:
        data = tomllib.load(source)
    hosts, datasets = data.get("hosts", {}), data.get("datasets", {})
    problems = []
    for name, host in hosts.items():
        if not isinstance(host.get("max_connections"), int) or host["max_connections"] < 1:
            problems.append(f"host {name}: max_connections must be a positive integer")
        if not isinstance(host.get("supports_range"), bool):
            problems.append(f"host {name}: supports_range must be true or false")
    for name, ds in datasets.items():
        for key in DATASET_KEYS:
            if not ds.get(key):
                problems.append(f"{name}: missing {key}")
        if ds.get("host") not in hosts:
            problems.append(f"{name}: host {ds.get('host')!r} has no [hosts.*] entry")
        if ds.get("access") not in ACCESS_TYPES:
            problems.append(f"{name}: access {ds.get('access')!r} not one of {sorted(ACCESS_TYPES)}")
        if ds.get("access") == "fairdata" and not ds.get("record"):
            problems.append(f"{name}: a fairdata dataset needs its dataset id in `record`")
        files = ds.setdefault("files", [])
        if ds.get("files_status") == "complete" and not files:
            problems.append(f"{name}: files_status complete but no files")
        if ds.get("files_status") == "unknown" and files:
            problems.append(f"{name}: files_status unknown but files are listed")
        bases = set()
        for i, f in enumerate(files):
            missing = [k for k in FILE_KEYS if k not in f]
            if missing:
                problems.append(f"{name} file {i}: missing {missing}")
                continue
            if not isinstance(f["size"], int) or f["size"] <= 0:
                problems.append(f"{name}/{f['name']}: size must be a positive integer")
            try:
                fetch.parse_checksum(f["checksum"])
            except ValueError as exc:
                problems.append(f"{name}/{f['name']}: {exc}")
            if f.get("role", "data") not in ROLES:
                problems.append(f"{name}/{f['name']}: role {f['role']!r} not one of {list(ROLES)}")
            base = local_name(f["name"])
            if base in bases:
                problems.append(f"{name}: two files share the base name {base!r}")
            bases.add(base)
    if problems:
        raise ValueError("catalog problems:\n  " + "\n  ".join(problems))
    return data


def local_name(name: str) -> str:
    return PurePosixPath(name).name


def dest_dir(dataset: str) -> Path:
    """$MCLS_RAW itself if set (the file lands directly there), else data/raw/<dataset>."""
    raw = os.environ.get("MCLS_RAW")
    return Path(raw) if raw else DEFAULT_RAW / dataset


def select(ds: dict, only: list[str]) -> list[dict]:
    """The files named in `only` (catalog name or base name), or all files."""
    if not only:
        return list(ds["files"])
    chosen = []
    for want in only:
        hits = [f for f in ds["files"] if f["name"] == want] or [
            f for f in ds["files"] if local_name(f["name"]) == want
        ]
        if not hits:
            raise Refused(f"{want!r} is not in the catalog for this dataset (see --list)")
        if len(hits) > 1:
            raise Refused(f"{want!r} matches several files; give the full name from --list")
        if hits[0] not in chosen:
            chosen.append(hits[0])
    return chosen


def effective_connections(requested: int, host: dict) -> int:
    if requested < 1:
        raise Refused("--connections must be at least 1")
    cap = MAX_CONNECTIONS if host.get("supports_range") else 1
    return min(requested, cap, host["max_connections"])


def fairdata_url(dataset_id: str, path: str, timeout: float = 60.0) -> str:
    """A single-use download URL for one Fairdata file, from Etsin's public authorize endpoint."""
    body = json.dumps({"cr_id": dataset_id, "file": path}).encode()
    request = urllib.request.Request(
        FAIRDATA_AUTHORIZE,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "User-Agent": fetch.USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        url = json.load(response).get("url")
    if not url:
        raise IOError(f"Fairdata gave no download URL for {path}")
    return url


def fetch_file(ds: dict, f: dict, out: Path, connections: int) -> Path:
    algorithm, value = fetch.parse_checksum(f["checksum"])
    digest = {algorithm: value}
    if ds["access"] == "fairdata":
        path = f["url"].removeprefix("fairdata:")
        return fetch.stream_download(lambda: fairdata_url(ds["record"], path), out, f["size"], **digest)
    return fetch.download(f["url"], out, f["size"], connections=connections, **digest)


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def run(args, out=None) -> int:
    out = out or sys.stdout
    catalog = load_catalog(args.catalog)
    datasets = catalog["datasets"]
    if args.dataset not in datasets:
        raise Refused(f"no dataset {args.dataset!r} in the catalog; known: {', '.join(sorted(datasets))}")
    ds = datasets[args.dataset]
    host = catalog["hosts"][ds["host"]]

    if args.list:
        for f in ds["files"]:
            print(f"{f['size'] / 1e6:12.1f} MB  {f.get('role', 'data'):8s}  {f['name']}", file=out)
        print(f"{gb(sum(f['size'] for f in ds['files']))} in {len(ds['files'])} files "
              f"(access: {ds['access']}, files: {ds['files_status']})", file=out)
        if not ds["files"]:
            print(f"file list unknown: {ds['files_note']}", file=out)
        return 0

    if ds["access"] not in FETCHABLE:
        raise Refused(
            f"{args.dataset}: access is {ds['access']!r}, not direct; mcls.acquire cannot fetch it.\n"
            f"  page: {ds['page']}\n  {ds['files_note']}"
        )
    if not ds["files"]:
        raise Refused(f"{args.dataset}: file list unknown. {ds['files_note']}")

    chosen = select(ds, args.only)
    total = sum(f["size"] for f in chosen)
    if args.max_gb is not None and total > args.max_gb * 1e9:
        raise Refused(f"{len(chosen)} file(s) total {gb(total)}, over --max-gb {args.max_gb}; nothing fetched")

    connections = effective_connections(args.connections, host)
    if connections != args.connections:
        print(f"note: {ds['host']} allows {connections} connection(s); using {connections}, not {args.connections}",
              file=out)
    folder = dest_dir(args.dataset)
    for f in chosen:
        target = folder / local_name(f["name"])
        if args.dry_run:
            print(f"would fetch {f['name']} ({f['size'] / 1e6:.1f} MB, {f['checksum'].split(':')[0]}) "
                  f"-> {target} over {connections} connection(s) from {ds['host']}", file=out)
            continue
        fetch_file(ds, f, target, connections)
    if args.dry_run:
        print(f"dry run: {gb(total)} in {len(chosen)} file(s); nothing downloaded", file=out)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mcls.acquire", description="Fetch dataset files from the catalog.")
    p.add_argument("--dataset", required=True, help="dataset name in configs/catalog.toml")
    p.add_argument("--list", action="store_true", help="list the dataset's files and sizes; fetch nothing")
    p.add_argument("--only", action="append", default=[], metavar="NAME",
                   help="fetch only this file (name or base name from --list); repeatable")
    p.add_argument("--connections", type=int, default=1, help=f"parallel byte ranges per file (default 1, max {MAX_CONNECTIONS})")
    p.add_argument("--max-gb", type=float, default=None, help="refuse if the selected files total more than this (1 GB = 1e9 bytes)")
    p.add_argument("--dry-run", action="store_true", help="print what would be fetched and where; download nothing")
    p.add_argument("--catalog", type=Path, default=None, help="catalog file (default configs/catalog.toml)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    except IOError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
