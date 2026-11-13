This is the [assistant-ui](https://github.com/assistant-ui/assistant-ui) starter project for LangGraph. It ships a minimal Ollama-backed agent (`backend/agent.ts`) plus a Next.js chat UI that streams from it.

## Getting Started

1. Set up the pipeline's Python environment, from the repository root (see the [root README](../README.md)):

   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

   The Laya service loads the fine-tuned checkpoint from `models/laya-rf-modes` in the repository root. **Run Laya** fails with a message naming that path when the folder is missing.

2. Copy the env template:

   ```bash
   cp .env.example .env.local
   ```

   Required:
   - Ollama running locally (`ollama serve`), with the model from `OLLAMA_MODEL` already pulled. Defaults to `qwen2.5:14b` at `http://127.0.0.1:11434`. Use the same model the pipeline loads: it keeps that one in GPU memory, and a small GPU cannot hold a second.

   Optional:
   - `OLLAMA_HOST` / `OLLAMA_MODEL` — override the local server or model id
   - `OLLAMA_NUM_CTX` — a larger context window for the chat
   - `PIPELINE_API_URL` — where the Laya service listens, `http://127.0.0.1:8000` by default
   - `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` — tracing
   - `LANGCHAIN_API_KEY` — only needed when pointing `LANGGRAPH_API_URL` at LangGraph Platform (cloud)

3. Install deps and run the web app, the LangGraph backend and the Laya service together:

   ```bash
   npm install
   npm run dev
   ```

   - `localhost:3000` — Next.js app
   - `localhost:2024` — LangGraph dev server (serves the `agent` graph)
   - `localhost:8000` — Laya service (`pipeline/server.py`)

   Run them individually with `npm run dev:frontend`, `npm run dev:backend` and `npm run dev:pipeline`.

## Using it

1. In the Library, open **Knowledge** and add a Laya questions file, such as `data/questions_20.json`.
2. Open **Data** and add observations, such as `data/test_observations.jsonl`. JSON lists and one-object-per-line files both work.
3. Each file is checked as it is added. One that is not valid JSON, or sits in the wrong tab, says so under its name and cannot be picked.
4. Press **Run Laya**. Chat is paused while Laya has the GPU. Only the first 100 rows of the data file are classified. Asking the chat to analyze the remaining rows, or a different number, runs Laya on that slice and summarizes it.
5. When the Library shows **Active**, ask the chat about the result. It answers from Laya's output only, and says so when the answer is not there.

**Clear**, or removing a file the result came from, drops the result. It otherwise survives a page reload, until the Laya service restarts.

## Project layout

```
app/                    Next.js App Router pages
app/api/[..._path]/     proxy to the LangGraph server
app/api/pipeline/       proxy to the Laya service
backend/agent.ts        LangGraph graph exported as `graph`
components/library.tsx  file upload, checks and the Run control
hooks/use-analysis.ts   run state shared by the Library and the chat
langgraph.json          LangGraph CLI config (graph id, node version, env file)
```

`app/assistant.tsx` builds the runtime with `useStreamRuntime({ assistantId, apiUrl })` from `@assistant-ui/react-langchain`, which wraps `useStream` from `@langchain/react`. It uses the websocket transport: over SSE, replies never arrive in Firefox.

On every turn `backend/agent.ts` reads the current result from the Laya service and puts it in the system prompt, so a new run is picked up by the next message.

## Deployment security

The bundled proxy rejects browser requests marked same-site or cross-site so the local starter works without exposing `LANGCHAIN_API_KEY` to the client. For clients without Fetch Metadata, reverse proxies must preserve the public scheme and host in the request URL for the `Origin` fallback. Request-context checks are not user authentication. Before deploying with a cloud API key, require your application session in `app/api/[..._path]/route.ts` and apply a durable rate limit.
