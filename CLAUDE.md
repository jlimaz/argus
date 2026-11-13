# Argus

Argus classifies a list of observed radio signals with a fine-tuned [Laya](https://github.com/NandhaKishorM/laya) model and lets a local chat model (via Ollama) answer plain-language questions about the result. Everything runs on one machine; there is no cloud service. Built at SprintHack@ND by Gabriel Marques and José Lima.

[README.md](README.md) is the source of truth for setup, the HTTP API, configuration and limits. [documentation/document.md](documentation/document.md) is the hackathon write-up (a snapshot; do not extend it for new work).

**Current focus (since 2026-11-13):** `mod-classifier/`, a separate classifier that reads raw IQ. Work is improving its data and its routing table. Start from [mod-classifier/STATUS.md](mod-classifier/STATUS.md).

## Layout

| Path | What |
|---|---|
| `pipeline/` | Python 3.12. File checks (`inputs.py`), Laya wrapper (`laya.py`, `checkpoint.py`), chat context (`context.py`, `analysis.py`), FastAPI service (`server.py`, port 8000), CLI (`__main__.py`). Capture ingest: sweep CSVs (`sweep.py`), IQ files (`iq.py`), burst measurement (`features.py`), CLI (`ingest.py`) |
| `frontend/` | Next.js app (port 3000), LangGraph chat agent in `backend/agent.ts` (port 2024), shared helpers in `lib/` |
| `mod-classifier/` | IQ signal classifier, self-contained: package `mcls/`, `configs/`, CRC jobs in `crc/`, trained runs in `runs/crc/` (committed on purpose; the rest of `runs/` and `data/` are ignored), Burst Router in `ui/` (port 8765), its own tests. Entry point: [mod-classifier/README.md](mod-classifier/README.md). `PLAN.md` and `docs/history/` are historical |
| `documentation/` | The write-up, its figures, and `local-setup.md` |
| `finetune/` | Data converters (`sources/`), dataset builder, training and evaluation |
| `data/` | Knowledge files, the 774 held-out test observations, training observations |
| `tests/` | pytest for the pipeline; does not load Laya or call Ollama |
| `models/` | Fine-tuned checkpoint (`laya-rf-modes`, ~800 MB), not committed |

## Commands

```bash
.venv/bin/python -m pytest tests          # Python tests (what CI runs)
cd mod-classifier && ../.venv/bin/python -m pytest tests -q   # classifier tests; must run from mod-classifier/, not in CI
cd frontend && npm run dev                 # app + agent + Laya service together
cd frontend && npx oxlint                  # lint (what CI runs; `npm run lint` also checks formatting, which fails today)
python -m pipeline "question"              # classify from the terminal
```

## Constraints worth remembering

- Laya and the chat model share a small GPU and take turns; the service unloads Ollama before a run and reloads it after.
- The Laya service and the chat agent must name the same `OLLAMA_MODEL`.
- One result is held in memory at a time; one run at a time.
- The chat model never sees raw rows, only the summary built in `pipeline/context.py`.
- `mod-classifier` trains on CRC only. Its kNN and LightGBM models are pickles, so the UI environment (`.venv-ui`) pins the library versions used in training.
- `mod-classifier` splits data by recording, never by window, and builds the routing table from validation predictions only.
- The classifier names a signal type; it never decodes content. Do not call it demodulation.

## Git workflow

- **NEVER run `git push` (or anything that pushes, such as `gh pr create` on an unpushed branch) without explicit authorization from the user in that moment.** Approval for one push does not carry over to the next.
- Every major feature starts on its own branch off `main`. Do not commit feature work directly to `main`.
- Branch names: `feat/<short-name>` for features, `fix/<short-name>` for bug fixes, `docs/<short-name>` for documentation, `chore/<short-name>` for tooling and cleanup. Use kebab-case, e.g. `feat/live-sdr-input`.
- Commit often: small, focused commits with a short imperative subject (`feat: add reclassify count`, `docs: update setup steps`).
- Each major feature ends in one PR from its branch to `main`, opened only after the user authorizes the push.
- Fill in [.github/pull_request_template.md](.github/pull_request_template.md) when opening a PR; its review checkboxes are optional.
- A PR must pass the `CI` workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml): pytest and `npx oxlint`) before merging. Run the same commands locally before asking to push.

## Documentation

Keep documentation good and brief, in plain language (readers are not RF experts).

- **Technical docs** (`documentation/local-setup.md`, `frontend/README.md`, `finetune/README.md`, `mod-classifier/README.md` and `STATUS.md`, and any doc next to the code) are updated often: in the same branch as every change that affects them.
- **Main `README.md`** changes only for major developments (a new capability, a new API, changed setup), not on every PR.
- PR descriptions: what changed, why, and how it was tested. A few lines, no filler.
