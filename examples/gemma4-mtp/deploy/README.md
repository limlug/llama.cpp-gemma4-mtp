# Production deployment with Docker + OpenWebUI

This directory contains a `docker-compose.yml` that builds and runs the
gemma4-mtp llama-server in a CUDA-enabled container, gated by an API
key, ready to be pointed at from OpenWebUI.

## Prerequisites

- A Linux host with one or more NVIDIA GPUs that has enough memory for
  the 31B base GGUF (~62 GB at fp16/bf16) plus KV cache. An H100 80 GB
  works at `CONTEXT_SIZE=8192`. Smaller GPUs require quantizing the
  base GGUF first.
- Docker Engine + [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).
- The two GGUFs already converted and reachable from the host:
  - `gemma-4-31B-it.gguf` (base, ~62 GB)
  - `gemma-4-31B-it-assistant.gguf` (MTP overlay, ~940 MB)
  See the top-level README for conversion instructions.

## Setup

```bash
cd examples/gemma4-mtp/deploy
cp .env.example .env
$EDITOR .env                                 # set MODELS_DIR and API_KEY

# docker-compose v1 (Ubuntu 22.04 default) or `docker compose` v2 plugin:
docker-compose build                         # ~5 min on a modern box
docker-compose up -d
docker-compose logs -f gemma4-mtp-server     # wait for `model loaded`
```

> The compose file uses `runtime: nvidia` for compatibility with both
> docker-compose v1 (non-swarm) and v2. Make sure
> `nvidia-container-runtime` is installed and `/etc/docker/daemon.json`
> registers it as a runtime.

The server is now reachable on `http://<host>:${HOST_PORT}/v1` (default
`http://localhost:8090/v1`).

## Smoke test

```bash
source .env
curl -s http://localhost:${HOST_PORT}/v1/models \
  -H "Authorization: Bearer ${API_KEY}"
curl -s http://localhost:${HOST_PORT}/v1/chat/completions \
  -H "Authorization: Bearer ${API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":200}'
```

Expected: `choices[0].message.content` contains the answer.
`choices[0].message.reasoning_content` contains the model's chain of
thought (Gemma 4 31B-it is a reasoning model — `max_tokens` must leave
headroom for both reasoning and answer; ~200 tokens is enough for
typical short responses).

## OpenWebUI integration

In OpenWebUI: **Admin Panel → Settings → Connections → OpenAI API**.

- **Base URL:** `http://<host>:${HOST_PORT}/v1`
- **API key:** value of `API_KEY` from `.env`
- **Models:** discovered automatically via `/v1/models`.

OpenWebUI ≥ 0.6 renders `reasoning_content` as a collapsible "thinking"
panel, so the reasoning-vs-content split is handled correctly.

## Sizing notes

- The whole 31B fp16 fits comfortably on one H100 80 GB. At
  `CONTEXT_SIZE=8192` the container sits around 65 GB used.
- KV cache scales roughly linearly with `CONTEXT_SIZE`. For `-ctk f16
  -ctv f16` (default): ~0.9 MB per token of context per slot.
- `PARALLEL_SLOTS>1` divides `CONTEXT_SIZE` across slots. Single-user
  setups should stay at `PARALLEL_SLOTS=1` so the entire budget goes
  to one session.
- Single-token decode TPS depends heavily on context length — expect
  ~10–15 TPS at 4K and ~2–4 TPS at 16K with the 31B fp16 model.

## Operational notes

- Restart policy is `unless-stopped`. A crash auto-restarts; `docker
  compose stop` stops cleanly.
- Health check polls `/health` every 30 s. Initial start is given 120 s
  because loading the 62 GB base GGUF from disk dominates startup.
- Models are mounted read-only — swap GGUFs by stopping the container,
  changing the `BASE_GGUF`/`DRAFT_GGUF` env vars, and starting again.
  No rebuild needed.
- Logs go to the standard Docker logging driver; consider attaching a
  log driver suitable for your infrastructure.

## Performance (measured on llm01, single H100 80 GB)

`-c 8192 --reasoning off -fa on`, prompt battery:

| Metric                         | value      |
| ------------------------------ | ---------- |
| Time-to-first-token (short Q)  | ~0.8 s     |
| Time-to-first-token (code)     | ~1.7 s     |
| Sustained decode TPS           | 3–6 tok/s  |
| Draft acceptance               | 3–5 %      |
| VRAM @ `-c 8192 -np 1`         | ~65 GB     |
| VRAM @ `-c 16384 -np 4`        | ~73 GB     |

Concurrency: `-np 2` gives the best aggregate throughput (~3.8 tok/s
across requests). `-np 4` doesn't scale further because the model is
memory-bandwidth bound on a single H100.

### Reasoning mode (thinking model — important!)

Gemma 4 31B-it is a thinking model. With reasoning ON (`REASONING_MODE=on`
or `auto`), the model emits 60+ seconds of `<|think|>` output before the
first real answer token — time-to-first-token jumps from <2 s to ~65 s
and long answers can hit a 600 s wall-clock cap. We therefore default
`REASONING_MODE=off`, which strips the `<|think|>` block and answers
directly.

If you want the reasoning panel back (OpenWebUI ≥ 0.6 renders it
nicely), set `REASONING_MODE=on` and increase `max_tokens` to 500+ so
both reasoning and answer fit.

## Known limitations

- Custom `stop` sequences set by the client can cut off reasoning
  before the answer arrives (when `REASONING_MODE` is not `off`). OpenWebUI
  doesn't set custom stops by default.
- Draft acceptance with the MTP overlay is currently 3–5 % on
  short answers and up to 10 % on longer continuations — useful but not
  transformative. Drafts that miss cost minimal extra compute, so it's
  a strict win vs no spec decoding.
- The drafter is not enabled if `DRAFT_GGUF` is removed; the server
  still works, just without speculative decoding.
