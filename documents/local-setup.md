# Run locally

This walks a new machine from a fresh clone to a working dashboard: radio observations go to Laya, which assigns each one a signal type, then a local Ollama model answers questions about that result.

When it is up you have three processes:

| What | Address |
|---|---|
| Dashboard (Next.js) | http://localhost:3000 |
| Chat agent (LangGraph) | http://localhost:2024 |
| Laya service | http://127.0.0.1:8000 |

You can skip the dashboard and run the same classification from the command line. That path is at the end.

Run every command from the repository root unless a step says otherwise. On Windows, use `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

## 1. Prerequisites

Install these before the project-specific steps:

- **Git**
- **Python 3.12.** The virtual environment below is created with `python3.12`. Check with `python3.12 --version`.
- **Node.js 22** and **npm.** The LangGraph dev server is pinned to Node 22 in `frontend/langgraph.json`. Check with `node --version`.
- **Ollama.** Install it from [ollama.com](https://ollama.com): the macOS app, `brew install ollama`, or the Linux install script on that site. The Windows installer from the same site works too.

Laya's checkpoint is about 1.6 GiB of weights, plus working memory. It uses CUDA when at least 2 GiB of GPU memory is free, otherwise Apple MPS, otherwise CPU. Laya and the chat model do not both stay on a small GPU: the app unloads one before loading the other.

The chat model the app expects is `qwen2.5:14b`. The chat uses it to call a tool, so a model without tool calling will not be able to change how many rows Laya classifies. That pull is several gigabytes and the model needs enough RAM or VRAM to load. If this machine cannot hold it, pick one smaller model that supports tool calling and does not spend the reply on a long hidden reasoning trace, pull that instead, and set the same name in both places in the steps below (the shell for the pipeline, and `frontend/.env.local` for the chat).

## 2. Clone and Python environment

```bash
git clone https://github.com/jlimaz/undefined-sprint-hack.git
cd undefined-sprint-hack
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs three packages: `laya` (loads the checkpoint), `fastapi`, and `uvicorn`. Leave the virtual environment active for the Laya and command-line steps. The dashboard starts the Python service with `.venv/bin/python` itself, so that environment has to exist at the repository root even if your shell is not activated.

## 3. Ollama

Start Ollama. On macOS, opening the app is enough. Otherwise:

```bash
ollama serve
```

Leave that process running. In another terminal, pull the model:

```bash
ollama pull qwen2.5:14b
ollama list
```

`ollama list` should show `qwen2.5:14b`. A quick check that the server is up:

```bash
curl http://127.0.0.1:11434/api/tags
```

The pipeline and the dashboard must name the same model. They read it from different places:

| Setting | Where the pipeline reads it | Where the dashboard reads it | Default |
|---|---|---|---|
| Model | `OLLAMA_MODEL` in the shell | `OLLAMA_MODEL` in `frontend/.env.local` | `qwen2.5:14b` |
| Address | `OLLAMA_HOST` in the shell | `OLLAMA_HOST` in `frontend/.env.local` | `http://127.0.0.1:11434` |

Leave both unset to use the defaults. If you pulled a different model, export it before any `python -m pipeline` command, and put the same value in `frontend/.env.local`:

```bash
export OLLAMA_MODEL=your-model-name
```

`qwen3:4b` without `-instruct` reasons at length and runs out of tokens before it answers. Do not use that tag.

## 4. Laya checkpoint

Classification looks for `models/laya-rf-modes/model.safetensors`. That folder is not in git. Without it, a run stops with `FileNotFoundError` and a message naming that path.

Use one of the two options below. Copying is the short path when someone already has the folder. Training builds it on this machine.

### Option A. Copy an existing checkpoint

Copy the whole `laya-rf-modes` folder (about 800 MB) so this file exists:

```text
models/laya-rf-modes/model.safetensors
```

The folder sits at the repository root, next to `pipeline/` and `frontend/`. To use a checkpoint that lives somewhere else, or a Hugging Face id, set `LAYA_CHECKPOINT` in the same shell that starts the pipeline (and in the environment of `npm run dev` if you use the dashboard):

```bash
export LAYA_CHECKPOINT=/path/to/laya-rf-modes
```

### Option B. Train the checkpoint

Stay at the repository root with the virtual environment active. Do not run `git checkout laya`. That step is in `finetune/README.md` and is out of date; this tree already contains the pipeline and the fine-tuning code.

Build the training items. This takes a few seconds and should report 6,964 training rows and 774 test rows:

```bash
python -m finetune.build_dataset
```

Train. The first run downloads the base checkpoint (about 800 MB) into `models/laya-base`, then writes `models/laya-rf-modes`.

```bash
python finetune/train.py \
  --items finetune/out/train_items.pt \
  --model-dir models/laya-base \
  --output-dir models/laya-rf-modes \
  --epochs 3 --micro-batch 8 --grad-accum 1 --no-checkpointing
```

It should print `Device: mps` on an Apple Silicon Mac (or `cuda` when a NVIDIA GPU has enough free memory) and `Training items: 6564; calibration items: 400`, then a loss line every 100 batches. Expect hours, not minutes. The guide this command comes from is [finetune/README.md](../finetune/README.md), including what the accuracy numbers mean.

If training runs out of memory, rerun with `--micro-batch 4 --grad-accum 2`. If that still fails, use `--micro-batch 2 --grad-accum 4` and leave out `--no-checkpointing`. If the log says `Device: cpu` on a Mac, MPS is not available in this Python, and training will take many more hours. An interrupted run saves `checkpoint_latest/` inside the output folder after every epoch. That folder can be evaluated, but the trainer cannot resume from it.

## 5. Dashboard

From `frontend/`:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

`npm run dev` starts all three processes. Leave `.env.local` on `OLLAMA_MODEL=qwen2.5:14b` unless you pulled a different model in step 3. The other values in that file already point at local Ollama, the Laya service on port 8000, and the LangGraph server on port 2024. LangSmith keys can stay blank.

Open http://localhost:3000.

To start one process at a time, from `frontend/`: `npm run dev:frontend`, `npm run dev:backend`, and `npm run dev:pipeline`.

## 6. Run a classification

In the dashboard Library:

1. Open **Knowledge** and add `data/questions_20.json` from the repository root (the file picker path is the file on disk).
2. Open **Data** and add `data/test_observations.jsonl`. A JSON list of observations works too. Each row needs `center_frequency_hz`, `bandwidth_hz`, and `modulation`.
3. A file that is not valid, or that was added on the wrong tab, shows an error under its name and cannot be selected.
4. Press **Run Laya**. Only the first 100 rows are classified. Chat is paused while Laya has the GPU. On a small GPU this is often under a minute for 100 rows, plus the time to load Ollama again afterward (several seconds to half a minute). Asking the chat to analyze the remaining rows, or a different number, runs Laya on that slice and summarizes it.
5. When the Library shows **Active**, ask the chat about the result. It answers from Laya's output. If the result does not contain the answer, it says so.

**Clear**, or removing a file the result came from, drops the result. Otherwise the result stays until the Laya service restarts, including across a page reload.

## 7. Command line only

Skip section 5 if you do not want the dashboard. Ollama must be running, the model must be pulled, and `models/laya-rf-modes` (or `LAYA_CHECKPOINT`) must be in place. From the repository root, with the virtual environment active:

```bash
python -m pipeline
```

Each observation is printed as one JSON object with the signal type Laya chose and its confidence. Then the Ollama answer, then a line showing how many of Laya's labels match `label_mode`. Timings go to stderr.

The default question is "Which observations are Morse code, and how many are not?". Pass another one as arguments:

```bash
python -m pipeline "How many weather fax signals were seen?"
```

To see only the answer and the reference line:

```bash
python -m pipeline | tail -n 2
```

To serve the same HTTP API the dashboard uses, without Next.js:

```bash
python -m uvicorn pipeline.server:app --host 127.0.0.1 --port 8000
```

## 8. If something fails

**Ollama is not running.** `curl http://127.0.0.1:11434/api/tags` fails, or a run errors while calling the model. Start `ollama serve` or the Ollama app, then `ollama pull qwen2.5:14b`.

**Missing Laya checkpoint.** The error names `models/laya-rf-modes`. Copy that folder in, set `LAYA_CHECKPOINT`, or train it (section 4). The base model id `convaiinnovations/laya` loads if you set `LAYA_CHECKPOINT` to it, but it has not been trained on this question, so the labels are not meaningful.

**Chat and pipeline use different models.** A small GPU holds one model. `OLLAMA_MODEL` in the shell (pipeline) and in `frontend/.env.local` (chat) must match, and that model must appear in `ollama list`. Restart `npm run dev` after changing `.env.local`.

**Port already in use.** The dashboard needs 3000, the agent needs 2024, and the Laya service needs 8000. Stop the other process using that port, or the second `npm run dev` will fail to bind.

**`Device: cpu` during training on a Mac.** MPS is not available in this Python environment. Use the Python 3.12 virtual environment from section 2.

**Run Laya says the service is unreachable.** The pipeline process is not up. From `frontend/`, `npm run dev` should show a `pipeline` line. By itself: `npm run dev:pipeline`, which runs `.venv/bin/python` from the repository root. If that binary is missing, repeat section 2.

**A second classification is refused.** One result is kept in memory, and a run started while one is going returns an error. Wait until the Library leaves the running state.

Tests do not load Laya or call Ollama, so a green test run does not mean the stack is up. They only check file validation, the text built for the chat model, and the HTTP service:

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest tests
```
