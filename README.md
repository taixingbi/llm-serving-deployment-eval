# qwen-deployment-eval

Phase 1 harness: **Performance Evaluation of Qwen2.5-7B Across Self-Hosted, ECS, and Amazon Bedrock Deployments**.

Identical OpenAI-compatible workloads via [LLMPerf](https://github.com/ray-project/llmperf) against:

| Backend | Hardware / service | Endpoint |
| --- | --- | --- |
| `selfhost` | 1× RTX 3090, Ubuntu 24.04, vLLM | `SELFHOST_URL` in `.env` |
| `ecs` | 1× g5.xlarge (A10G), vLLM | CloudFormation `ecs-inference-mvp` → `ServiceUrl` |
| `bedrock` | Imported Qwen2.5-7B | CloudFormation `bedrock-inference-mvp` → Function URL |

## Fixed configuration

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Temperature | `0` |
| Top-p | `1.0` |
| Input/output token stddev | `0` |
| API | OpenAI `/v1/chat/completions` (streaming for TTFT) |
| Tool | LLMPerf `token_benchmark_ray.py` |

## Experiments (Phase 1)

1. **Concurrency** — 1, 2, 4, 8, 16, 32, 64 (mean_in≈550, mean_out≈150)
2. **Prompt length** — 512, 1024, 2048, 4096 (conc=1; **8192 deferred** until ECS `MAX_MODEL_LEN` is raised)
3. **Output length** — 128, 256, 512, 1024 (conc=1, mean_in=512)
4. **RAG TopK** — deferred (no retrieval stack yet)

## Prerequisites

```bash
cd qwen-deployment-eval
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Clone LLMPerf once (no requirements.txt — use PYTHONPATH or editable install)
mkdir -p vendor
git clone https://github.com/ray-project/llmperf.git vendor/llmperf
# Optional if your Python is 3.8–3.10:
# pip install -e vendor/llmperf

# Local secrets / endpoints (gitignored)
cp .env.example .env
# edit .env: INFERENCE_API_KEY, SELFHOST_URL, AWS_REGION, ...
```

Scripts auto-load the repo-root `.env` (`scripts/load_env.sh` / `scripts/envutil.py`).

Smoke-test endpoints with [`example.md`](example.md).

## Run one cell

From the repo root, with the venv activated (`source .venv/bin/activate`):

```bash
source scripts/resolve_endpoints.sh selfhost   # or ecs / bedrock
BACKEND=selfhost EXP_ID=smoke_c1_in550_out150 \
  MEAN_IN=550 MEAN_OUT=150 CONC=1 N_REQ=10 \
  ./scripts/run_one.sh
# optional GPU/CPU sampling on the machine that has the GPU:
# SAMPLE_RESOURCES=1 ./scripts/run_one.sh
```

## Run a full experiment matrix

With the venv activated (`source .venv/bin/activate`), so `python` points at `.venv`:

```bash
source .venv/bin/activate
# Preview
python scripts/run_matrix.py --experiment concurrency --backends selfhost --dry-run

# Execute (add --sample-resources on the GPU host for self-host)
python scripts/run_matrix.py --experiment concurrency --backends selfhost,ecs,bedrock
python scripts/run_matrix.py --experiment prompt_length --backends selfhost,ecs,bedrock
python scripts/run_matrix.py --experiment output_length --backends selfhost,ecs,bedrock
```

### ECS GPU sampling

`sample_resources.py` must run on a host with the GPU (and `nvidia-smi` / NVML). On the ECS GPU instance via SSM:

```bash
# after resolve + while a load test is running from your laptop, on the instance:
python3 sample_resources.py --out /tmp/resources.jsonl --interval 1.0
# copy JSONL into results/raw/ecs/<exp_id>/resources.jsonl before aggregate.py
```

## Aggregate, cost, figures

```bash
python scripts/aggregate.py

# Optional but recommended for Bedrock: attach CloudWatch ModelCopy per cell
# python scripts/fetch_bedrock_metrics.py --exp-id concurrency_c64_in550_out150 \
#   --imported-model-arn 'arn:aws:bedrock:us-east-1:ACCOUNT:imported-model/ID'
# then re-run aggregate.py

python scripts/estimate_cost.py   # edit configs/cost.yaml first
python scripts/plot_figures.py
```

`estimate_cost.py` writes **three** cost views (paper primary = normalized):

| Column | Use |
| --- | --- |
| `normalized_compute_cost_usd` | Busy capacity (`$/hr × active duration`) — **main efficiency table** |
| `standalone_billed_cost_usd` | If the cell ran alone (Bedrock ≥1×5-min window) — cold/bursty |
| `session_allocated_cost_usd` | Share of the matrix **wall-clock** session bill — invoice attribution |
| `cost_per_request_normalized_usd` | Frontier / cost-vs-latency plots |
| `cost_per_request_billed_usd` | Per-request session allocation |
| `model_copies_observed` / `model_copy_source` | `cloudwatch` or `configured_assumption` |

Also writes `results/session_costs.csv` (one row per backend×experiment session).

Self-host prints electricity-only vs amortized TCO $/hour. ECS is labeled **compute-only** (g5.xlarge instance price; no ALB/EBS/NAT).

Outputs:

- `results/aggregated.csv`
- `results/with_cost.csv`
- `paper/figures/*.png`

## Layout

```
.env.example
configs/backends.yaml
configs/cost.yaml
configs/experiments/{concurrency,prompt_length,output_length}.yaml
scripts/{load_env,resolve_endpoints,run_one,run_matrix,sample_resources,aggregate,estimate_cost,fetch_bedrock_metrics,plot_figures,envutil}
paper/METHODS.md
example.md
```

## Paper notes

See [`paper/METHODS.md`](paper/METHODS.md). Capacity-fair framing: 1×3090 vs 1×A10G vs Bedrock imported copies — not hardware-identical. Cost figures must distinguish **normalized busy capacity** from **CMU 5-minute billing floors / session invoices**.
