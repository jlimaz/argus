"""Download large files from Zenodo (or any HTTP server that honours Range).

Why this exists: Zenodo caps each connection at about 0.6 MB/s (measured
Nov 11, 2026: 0.6 MB/s on one connection, 2.4 MB/s on four, against 18 MB/s
from a CDN on the same line). At one connection a 50 GB archive takes about a
day. This module splits a file into byte ranges and fetches several at once.

Rules it follows:

- At most MAX_CONNECTIONS ranges at once per file, so a cluster does not get
  rate-limited or blocked.
- Resumable: each range is written to its own part file; a rerun skips
  finished parts and continues partial ones.
- Verified: the assembled file must match the expected size and checksum
  (Zenodo publishes md5; NIST, Mendeley and Fairdata publish sha256) or it is
  not kept.
- Hosts that ignore Range (Fairdata) use `stream_download`: one connection,
  one GET from byte 0, a fresh URL per attempt (Fairdata tokens are single-use).
- No external packages: runs on CRC with the system Python 3.12.

Command line:

    python -m mcls.fetch URL OUTPUT --size BYTES --md5 HEX [--connections 4]
    python -m mcls.fetch URL OUTPUT --size BYTES --sha256 HEX
    python -m mcls.fetch --zenodo 4264467 --only DJI_mavic_mini_2G.bin --out data/raw/tampere
    python -m mcls.fetch --zenodo 4264467 --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MAX_CONNECTIONS = 8
DEFAULT_CONNECTIONS = 4
CHUNK = 1 << 20
USER_AGENT = "argus-mcls-fetch/0.1"
RETRIES = 5


def ranges(size: int, parts: int) -> list[tuple[int, int]]:
    """Split [0, size) into `parts` contiguous inclusive byte ranges."""
    parts = max(1, min(parts, size))
    step = -(-size // parts)
    return [(start, min(start + step, size) - 1) for start in range(0, size, step)]


def _fetch_range(url: str, part: Path, start: int, end: int, timeout: float) -> int:
    """Fetch bytes start..end into `part`, resuming from what it already holds."""
    want = end - start + 1
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return want
        if have > want:
            part.unlink()
            have = 0
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": USER_AGENT}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise IOError(f"server ignored the Range header (status {response.status})")
                with open(part, "ab") as out:
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
        except (OSError, IOError) as exc:
            if attempt == RETRIES:
                raise IOError(f"{part.name}: gave up after {RETRIES} attempts: {exc}") from exc
            time.sleep(min(30, 2**attempt))
    have = part.stat().st_size
    if have != want:
        raise IOError(f"{part.name}: got {have} bytes, expected {want}")
    return want


def checksum(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as source:
        while chunk := source.read(CHUNK * 8):
            digest.update(chunk)
    return digest.hexdigest()


def expected_digest(md5: str | None = None, sha256: str | None = None) -> tuple[str, str] | None:
    """(algorithm, lowercase hex) to verify against, or None. sha256 wins if both are given."""
    if sha256:
        return "sha256", sha256.lower()
    if md5:
        return "md5", md5.lower()
    return None


def parse_checksum(text: str) -> tuple[str, str]:
    """Split "sha256:abc..." into ("sha256", "abc..."); refuse unknown algorithms and bad lengths."""
    algorithm, sep, value = text.partition(":")
    algorithm = algorithm.lower()
    lengths = {"md5": 32, "sha256": 64}
    if not sep or algorithm not in lengths:
        raise ValueError(f"checksum {text!r}: expected 'md5:<hex>' or 'sha256:<hex>'")
    if len(value) != lengths[algorithm] or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"checksum {text!r}: {algorithm} needs {lengths[algorithm]} hex digits")
    return algorithm, value.lower()


def _have(output: Path, size: int, digest: tuple[str, str] | None) -> bool:
    return output.exists() and output.stat().st_size == size and (digest is None or checksum(output, digest[0]) == digest[1])


def download(
    url: str,
    output: Path,
    size: int,
    md5: str | None = None,
    connections: int = DEFAULT_CONNECTIONS,
    timeout: float = 120.0,
    progress: bool = True,
    sha256: str | None = None,
) -> Path:
    """Download `url` to `output` over parallel byte ranges; verify size and md5 or sha256.

    Returns `output`. Raises IOError and leaves the part files in place (so a
    rerun resumes) when anything goes wrong.
    """
    if connections < 1 or connections > MAX_CONNECTIONS:
        raise ValueError(f"connections must be 1..{MAX_CONNECTIONS}")
    digest = expected_digest(md5, sha256)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if _have(output, size, digest):
        if progress:
            print(f"have {output.name}")
        return output

    plan = ranges(size, connections)
    parts_dir = output.with_name(output.name + ".parts")
    parts_dir.mkdir(exist_ok=True)
    parts = [parts_dir / f"{i:03d}" for i in range(len(plan))]
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        futures = {pool.submit(_fetch_range, url, part, s, e, timeout): part for part, (s, e) in zip(parts, plan)}
        for future in as_completed(futures):
            future.result()
    seconds = max(time.time() - started, 1e-6)

    tmp = output.with_name(output.name + ".tmp")
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as source:
                while chunk := source.read(CHUNK * 8):
                    out.write(chunk)
    assembled = tmp.stat().st_size
    if assembled != size:
        tmp.unlink()
        raise IOError(f"{output.name}: assembled {assembled} bytes, expected {size}")
    if digest is not None:
        algorithm, want = digest
        got = checksum(tmp, algorithm)
        if got != want:
            tmp.unlink()
            for part in parts:
                part.unlink()
            raise IOError(f"{output.name}: {algorithm} {got} does not match {want}; parts removed, rerun to fetch again")
    tmp.replace(output)
    for part in parts:
        part.unlink()
    parts_dir.rmdir()
    if progress:
        print(f"got  {output.name} ({size / 1e6:.0f} MB in {seconds:.0f} s, {size / 1e6 / seconds:.2f} MB/s)")
    return output


def stream_download(
    url,
    output: Path,
    size: int,
    md5: str | None = None,
    sha256: str | None = None,
    timeout: float = 120.0,
    progress: bool = True,
    retries: int | None = None,
) -> Path:
    """Download over ONE connection with no Range header, for hosts that ignore ranges.

    `url` is a string or a zero-argument callable returning a fresh URL; it is
    called once per attempt, so single-use download tokens (Fairdata) work.
    A broken transfer cannot resume: the next attempt starts from byte 0.
    A checksum mismatch keeps the file as `<name>.bad` (not deleted), since
    fetching it again costs the whole transfer.
    """
    digest = expected_digest(md5, sha256)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if _have(output, size, digest):
        if progress:
            print(f"have {output.name}")
        return output
    tmp = output.with_name(output.name + ".tmp")
    retries = RETRIES if retries is None else retries
    started = time.time()
    for attempt in range(1, retries + 1):
        try:
            target = url() if callable(url) else url
            request = urllib.request.Request(target, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response, open(tmp, "wb") as out:
                if response.status != 200:
                    raise IOError(f"unexpected status {response.status}")
                while chunk := response.read(CHUNK):
                    out.write(chunk)
            got = tmp.stat().st_size
            if got != size:
                raise IOError(f"got {got} bytes, expected {size}")
            break
        except (OSError, IOError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise IOError(f"{output.name}: gave up after {retries} attempts: {exc}") from exc
            time.sleep(min(30, 2**attempt))
    seconds = max(time.time() - started, 1e-6)
    if digest is not None:
        algorithm, want = digest
        got = checksum(tmp, algorithm)
        if got != want:
            # One stream, no resume: refetching costs the whole file, so keep it for inspection.
            bad = output.with_name(output.name + ".bad")
            tmp.replace(bad)
            raise IOError(
                f"{output.name}: {algorithm} {got} does not match {want}; kept as {bad} "
                f"({size / 1e9:.1f} GB, not deleted). Check it, then delete it and rerun to fetch again"
            )
    tmp.replace(output)
    if progress:
        print(f"got  {output.name} ({size / 1e6:.0f} MB in {seconds:.0f} s, {size / 1e6 / seconds:.2f} MB/s, one stream)")
    return output


def zenodo_files(record: str, timeout: float = 60.0) -> list[dict]:
    """Name, size, md5 and download URL of every file in a Zenodo record."""
    request = urllib.request.Request(
        f"https://zenodo.org/api/records/{record}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    files = []
    for entry in data["files"]:
        algorithm, _, value = entry["checksum"].partition(":")
        files.append(
            {
                "name": entry["key"],
                "size": int(entry["size"]),
                "md5": value if algorithm == "md5" else None,
                "url": entry["links"]["self"],
            }
        )
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parallel, resumable, verified downloads.")
    parser.add_argument("url", nargs="?", help="file URL (or use --zenodo)")
    parser.add_argument("output", nargs="?", type=Path, help="where to write the file")
    parser.add_argument("--size", type=int, help="expected size in bytes (required with a URL)")
    parser.add_argument("--md5", help="expected md5")
    parser.add_argument("--sha256", help="expected sha256 (used instead of --md5 if both are given)")
    parser.add_argument("--zenodo", help="Zenodo record id, e.g. 4264467")
    parser.add_argument("--only", action="append", default=[], help="with --zenodo: file name(s) to fetch")
    parser.add_argument("--out", type=Path, default=Path("."), help="with --zenodo: output folder")
    parser.add_argument("--list", action="store_true", help="with --zenodo: list files and sizes, fetch nothing")
    parser.add_argument("--connections", type=int, default=DEFAULT_CONNECTIONS)
    args = parser.parse_args(argv)

    try:
        if args.zenodo:
            files = zenodo_files(args.zenodo)
            if args.list:
                for f in files:
                    print(f"{f['size'] / 1e6:10.1f} MB  {f['name']}")
                print(f"{sum(f['size'] for f in files) / 1e9:.2f} GB in {len(files)} files")
                return 0
            chosen = [f for f in files if not args.only or f["name"] in args.only]
            missing = set(args.only) - {f["name"] for f in files}
            if missing:
                print(f"not in record {args.zenodo}: {sorted(missing)}", file=sys.stderr)
                return 1
            for f in chosen:
                download(f["url"], args.out / f["name"], f["size"], f["md5"], args.connections)
            return 0
        if not (args.url and args.output and args.size):
            parser.error("give URL OUTPUT --size, or --zenodo RECORD")
        download(args.url, args.output, args.size, args.md5, args.connections, sha256=args.sha256)
        return 0
    except (IOError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
