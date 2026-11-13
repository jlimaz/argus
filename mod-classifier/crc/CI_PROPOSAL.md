# Proposal: run `mod-classifier/tests` in the repo CI

**Status: draft, not applied.** `.github/workflows/ci.yml` belongs to Gabriel; nothing changes without his agreement.

## Today

The `python` job installs `fastapi uvicorn pytest httpx2 numpy h5py unrar-cffi` (no torch) and runs `python -m pytest tests` from the repo root. `mod-classifier/tests` is never collected, so a PR to `main` can break the classifier pipeline with CI green.

`mod-classifier/tests` needs only numpy and pytest on a laptop; torch tests use `pytest.importorskip("torch")`, and the CRC script checks need only `bash`. `mod-classifier/conftest.py` puts `mod-classifier/` on `sys.path`, so the tests must run with `mod-classifier/` as the root directory (`import mcls` would fail from the repo root otherwise).

## Proposed change (one step, no new installs)

Add one step to the existing `python` job, after the root tests:

```yaml
      - run: python -m pytest tests
      # mod-classifier pipeline tests: no torch; torch tests skip themselves.
      - run: python -m pytest tests -q
        working-directory: mod-classifier
```

- Same job, same cached pip install: about one extra second today (58 tests, 1.7 s on a laptop).
- `numpy` and `h5py` are already installed. If a later loader needs `scipy`, add it to the install line then (a small wheel, no torch).
- Running from `mod-classifier/` keeps its `conftest.py` and `tests/` separate from the root `tests/`, so the two suites cannot shadow each other's modules.

## Alternatives considered

| Option | Why not first choice |
| --- | --- |
| A separate `mod-classifier` job | Cleaner failure reports, but repeats checkout and pip install (about 30 s more per run). Worth it only if the suite grows slow |
| `python -m pytest tests mod-classifier/tests` from the root | Two `conftest.py` roots and two `tests` packages in one session; fragile import paths |
| Mirror the tests under root `tests/` | Duplicated files drift apart |
| Run only on changes under `mod-classifier/**` (`paths:` filter) | Saves little; a path filter on a required check can block PRs that do not touch the folder |

## What stays out of CI

- Anything needing torch, a GPU, or real datasets (they skip or live on CRC).
- `shellcheck` on `mod-classifier/crc/*`: optional extra step (`shellcheck` is preinstalled on `ubuntu-latest`), proposed only once the scripts have been run on CRC and their warnings cleaned up.

## To decide with Gabriel

1. Accept the one-step change above, or prefer a separate job.
2. Whether the mod-classifier step is required for merging to `main` from day one.
