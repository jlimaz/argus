"""Bytes of node /tmp one prepare task needs, from the catalog; printed as one integer.

    python scripts/space_needed.py gateman rawdata.zip --unpack
    python scripts/space_needed.py gateman rawdata.zip --unpack --unpacked-gb 60

The file's catalog `size`, plus, with --unpack, its extracted size: the file's optional
`unpacked_size` (integer bytes) in configs/catalog.toml, or --unpacked-gb when the catalog
does not give it. With --unpack and neither, it refuses rather than guess. Preprocess output
is not counted: it is windows only, far smaller than the raw file (PLAN 3.7).

crc/prepare_array.job runs this before downloading and compares it with `df` of node /tmp.
Exit 0 ok, 2 refused (unknown dataset or file, unknown extracted size). No network.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mcls.acquire import load_catalog  # noqa: E402


def needed_bytes(catalog: dict, dataset: str, name: str, unpack: bool = False,
                 unpacked_gb: float | None = None) -> int:
    """Raises ValueError with the reason when the answer is unknown."""
    entry = catalog.get("datasets", {}).get(dataset)
    if entry is None:
        raise ValueError(f"unknown dataset {dataset!r}")
    matches = [f for f in entry.get("files", []) if name in (f["name"], PurePosixPath(f["name"]).name)]
    if len(matches) != 1:
        raise ValueError(f"{dataset}: no single catalog file named {name!r}")
    f = matches[0]
    need = int(f["size"])
    if unpack:
        if unpacked_gb is not None:
            need += int(unpacked_gb * 1e9)
        elif isinstance(f.get("unpacked_size"), int) and f["unpacked_size"] > 0:
            need += f["unpacked_size"]
        else:
            raise ValueError(f"{dataset}/{f['name']}: the catalog gives no unpacked_size, so the room needed "
                             "to extract it is unknown; set MCLS_UNPACKED_GB to the extracted size in GB")
    return need


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset")
    parser.add_argument("name")
    parser.add_argument("--unpack", action="store_true")
    parser.add_argument("--unpacked-gb", type=float)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args(argv)
    try:
        print(needed_bytes(load_catalog(args.catalog), args.dataset, args.name, args.unpack, args.unpacked_gb))
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
