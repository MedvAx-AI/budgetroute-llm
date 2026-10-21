# BudgetRoute-LLM

BudgetRoute-LLM is a typed, quality-aware language-model routing system. It combines small and large model backends, retrieval, confidence cascades, abstention, and durable human review with reproducible evaluation and a tenant-aware service boundary.

> **Status: v0.7 research and multi-host deployment-validation implementation with two published real GPU studies.** PostgreSQL integration tests prove control-plane transaction semantics, not target-fleet capacity or availability. The latest MMLU-500/held-out result remains a negative learned-routing result, and fake timings remain instrumentation-only.

**Research implementation (2026-10-21):** the free-Colab [pilot specification](docs/superpowers/specs/2026-10-21-colab-routing-study-design.md) and [Native implementation plan](docs/superpowers/plans/2026-10-21-colab-routing-study.md) are approved. New router/calibrator artifacts now fail closed when no threshold is feasible; legacy numeric artifacts remain readable and uncertified. The [assessment](docs/research-reassessment.md) records the motivation and remaining gaps. Remote CI currently proves the old source only, and in-app-browser Colab access is verified; no new remote GPU result, quality guarantee, or speedup is claimed. Historical benchmark profiles/reports are unchanged.

## What it answers

The central question is: can a routing policy reduce average inference cost or latency while maintaining a stated quality target? The project makes that question measurable by preserving route decisions, model outcomes, confidence, retrieval, latency, usage, uncertainty, configuration, environment, and provenance.

## Architecture

```mermaid
flowchart LR
    A["Authenticated tenant request"] --> B["Shared quota and admission lease"]
    B --> C["Bounded deadline-aware scheduler"]
    C --> D["Features and optional retrieval"]
    D --> E["Routing policy"]
    E --> F["Small / retrieved / large / cascade"]
    E --> G["Abstain / human review"]
    F --> H["Structured response and prediction record"]
    G --> H
    H --> I["Metrics, drift, feedback, audit"]
    I --> J["Gated calibration registry"]
    H --> K["Benchmark artifacts and reports"]
```

Protocols separate generation backends, retrieval indexes, routing policies, operational storage, evaluators, and reports. Pydantic schemas form the boundaries, and composable YAML selects implementations. See [architecture](docs/architecture.md).

## System capabilities

- Multi-host replica coordination through pooled PostgreSQL, with ordered checksum-verified migrations, database-clock quotas/leases, atomic admission, idempotent records, optimistic review transitions, and serialized hash-chain audit writes. SQLite WAL remains the zero-infrastructure same-host option.
- Database-aware readiness, bounded connection pools, PostgreSQL TLS fail-closed validation, a real PostgreSQL CI contract, and reference Compose/Kubernetes topologies.
- Environment-only tenant credentials with constant-time comparison and `inference`, `feedback`, `review`, and `admin` scopes. Credentials are not serialized to config responses or artifacts.
- Privacy-minimal durable review workflow with claim/resolve transitions, optimistic versions, tenant isolation, idempotent feedback, and a verifiable SHA-256 audit chain.
- Chronological online recalibration from delayed labels with a held-out tail, Brier/ECE promotion gates, Page-Hinkley change points, immutable candidates, atomic activation, and rollback.
- A separate learned retrieval-benefit classifier trained from paired `always_small` and `retrieval_first` evidence with group-safe train/calibration/test partitions.
- Portable exact cosine retrieval, optional FAISS flat search, and optional FAISS HNSW approximate search. Transformers embedding profiles require an exact model revision.
- Runtime compatibility artifacts and real multi-endpoint HTTP load generation with p50/p95/p99, measured wall-clock throughput, overload/error rates, and clearly labeled heuristic capacity guidance.
- Bootstrap confidence intervals, minimum-sample claim warnings, phrase-safe keyword matching, improved final numeric/fraction/percentage extraction, and numeric/category drift monitoring.
- Built-in TLS certificate/key support or an explicit external TLS-termination declaration for non-loopback serving, plus bounded metadata and configurable prompt/metadata/output substring policy.

Earlier milestones also provide revision-pinned public dataset adapters, content-addressed generation replay, calibrated learned routing, padded Transformers batching, load/cost-aware policies, local OpenAI-compatible backends, Prometheus metrics, Docker, reporting, and security CI.

## Execution profiles

| Profile | Purpose | Network/GPU | Configuration |
|---|---|---:|---|
| Fake | Tests, CI, architecture demos | No | `configs/serving/fake.yaml` |
| Distributed fake | Tenant/review/audit/replica workflow | No model network | `configs/serving/distributed.yaml` |
| Multi-host PostgreSQL | Independent API replicas and shared control-plane state | PostgreSQL network | `configs/serving/postgres.yaml` |
| CPU smoke | Pinned public-data/model integration | Download, CPU | `configs/benchmarks/real-cpu-gsm8k-smoke.yaml` |
| Local GPU | Published Qwen2.5 / MMLU-100 and held-out MMLU-500 studies | Download, CUDA | `configs/benchmarks/real-gpu-mmlu-500-learned-live.yaml` |
| Local servers | vLLM/llama.cpp/Ollama-compatible serving | Local endpoints | `configs/serving/local-openai-compatible.yaml` |

Fake output, confidence, and artificial latency are marked `fake: true` and are never performance evidence.

## Installation

Python 3.11–3.13 is supported.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e '.[dev]'
```

Use `.[postgres]` for multi-host operational state, `.[transformers,datasets]` for real local models and public datasets, `.[retrieval]` for supported FAISS platforms, or `.[all]` for the complete optional stack. CUDA wheels and quantization support can require platform-specific installation.

## Quick offline validation

```bash
python -m budgetroute demo --config configs/serving/fake.yaml
python -m budgetroute benchmark --config configs/benchmarks/fake-smoke.yaml
python -m budgetroute generate-report --latest
python -m budgetroute compatibility-matrix --config configs/serving/fake.yaml
```

The benchmark creates an ignored, unique `outputs/<UTC>_fake-smoke/` directory with resolved config, environment/Git metadata, predictions, routes, timings, errors, metrics, and reports. A fake warning is embedded in its artifacts.

## Published real benchmark

The latest study collected 500 paired Qwen2.5-0.5B/1.5B outcomes, trained and calibrated on disjoint 300/100 partitions, and evaluated once on 100 persisted untouched examples using an RTX 3060 Laptop GPU. The learned-router live matrix completed 400/400 requests with zero failures.

| Policy | Accuracy (95% CI) | p50 | p95 |
|---|---:|---:|---:|
| Always small | 41% (31%-51%) | 353.9 ms | 429.3 ms |
| Always large | 53% (43%-63%) | 423.3 ms | 528.6 ms |
| Learned (8 small / 92 large) | 51% (41%-61%) | 445.9 ms | 503.0 ms |
| Calibrated cascade (83 escalations) | 51% (41%-61%) | 689.7 ms | 895.9 ms |

The learned router made no quality wins over always-large, lost two questions, routed only 8% to small, and had a 5.3% higher observed p50. The calibrated cascade produced the same two losses and much worse latency. This is a useful negative result: the current cheap prompt features and token-confidence signal are insufficient. See the [held-out technical card](reports/benchmarks/qwen25-mmlu-500-learned-rtx3060.md), [machine-readable evidence](reports/benchmarks/qwen25-mmlu-500-learned-rtx3060.json), and the earlier [MMLU-100 study](reports/benchmarks/qwen25-mmlu-100-rtx3060.md).

## Distributed service setup

`configs/serving/distributed.yaml` uses SQLite for multiple processes on one reliable host. `configs/serving/postgres.yaml` uses PostgreSQL for replicas on independent hosts. Both expect tenant credentials from `BUDGETROUTE_TENANT_KEYS_JSON`; the PostgreSQL profile also reads its secret DSN from `BUDGETROUTE_POSTGRES_DSN`. Supply both through a secret manager or orchestrator, never a committed file:

```powershell
$env:BUDGETROUTE_TENANT_KEYS_JSON = '{"tenants":[{"tenant_id":"portfolio","subject":"operator","api_key":"replace-with-a-long-random-secret","scopes":["inference","feedback","review","admin"]}]}'
$env:BUDGETROUTE_POSTGRES_DSN = 'postgresql://budgetroute:replace-me@db.example.com/budgetroute?sslmode=verify-full'
$env:BUDGETROUTE_REPLICA_ID = 'budgetroute-1'
python -m budgetroute migrate-store --config configs/serving/postgres.yaml
# Replace BUDGETROUTE_POSTGRES_DSN with the narrower runtime-role DSN after migration.
python -m budgetroute validate-config --config configs/serving/postgres.yaml
python -m budgetroute security-check --config configs/serving/postgres.yaml
python -m budgetroute serve --config configs/serving/postgres.yaml
```

Change `api.trusted_hosts` to actual DNS names. The example declares `external_tls_termination: true`, so a trusted reverse proxy must terminate HTTPS and must not permit direct public access to the application port. Alternatively configure `api.tls_certfile` and `api.tls_keyfile` for Uvicorn-managed TLS. Non-loopback startup fails without authentication and one of these explicit TLS boundaries.

`migrate-store` applies packaged PostgreSQL migrations in filename order while holding a database advisory lock; normal production startup verifies their names and SHA-256 checksums with a narrower credential. Quotas use atomic upserts; lease admission and audit-head updates use short transaction-scoped advisory locks; review transitions use row locks and versions. Runtime secrets and prompt/answer bodies are not stored. `/readyz` returns HTTP 503 when inference or operational storage is unavailable.

For a local proof, use `docker-compose.postgres.yml`. For a multi-node starting point, review `deploy/kubernetes/` and the [PostgreSQL operations runbook](docs/postgres-operations.md). These examples do not provision managed-database HA, backups/PITR, an ingress/WAF, an identity provider, or target-fleet capacity evidence. SQLite remains appropriate only on one reliable host/filesystem and must not be placed on an arbitrary network filesystem.

### API and scopes

Health probes are public. Other endpoints require a Bearer or `X-API-Key` credential when authentication is configured.

| Scope | Endpoints |
|---|---|
| `inference` | `POST /v1/route`, `POST /v1/generate`, and `GET /v1/config` |
| `feedback` | `POST /v1/feedback` |
| `review` | list, claim, and resolve `/v1/reviews` cases for the caller's tenant |
| `admin` | monitoring/metrics, audit reads/verification, and cross-tenant administrative access |

`admin` implies every scope. Review records retain correlation IDs, state, reason, actor, and outcome—not prompts or generated text. Audit events are tamper-evident within the retained chain, not externally notarized or immutable against a database administrator.

## Adaptation workflow

Generation stores raw/calibrated confidence and numeric features under the authenticated tenant and request ID. Delayed feedback joins to that prediction without retaining feedback notes.

```bash
python -m budgetroute adapt-confidence --config configs/serving/distributed.yaml
python -m budgetroute rollback-calibration --config configs/serving/distributed.yaml
```

Adaptation is operator-invoked and fail-closed below the configured label count. It sorts by label time, trains on the earlier portion, evaluates on the held-out tail, and promotes only when Brier improvement and ECE regression gates pass. It never mutates a version in place. Promotion does not prove future performance; monitor after deployment and keep rollback operational.

Train the retrieval-benefit policy from a compatible benchmark artifact:

```bash
python -m budgetroute train-retrieval-router --artifacts outputs/RUN --output outputs/router/retrieval.joblib
```

Configure the resulting artifact with `configs/routing/learned-retrieval.yaml`. The label asks whether retrieval improved paired answer quality by the configured margin; it is intentionally separate from the small-model-success router.

## Semantic retrieval

`configs/retrieval/semantic-hnsw.yaml` demonstrates a revision-pinned `sentence-transformers/all-MiniLM-L6-v2` embedding profile with FAISS HNSW. Build with:

```bash
python -m budgetroute build-index --config configs/serving/YOUR-CONFIG.yaml
```

The persisted NumPy vectors/chunks remain portable, while HNSW is rebuilt with configured graph and search parameters on load. Approximate retrieval must be validated for recall and latency on the target corpus; the repository does not claim universal HNSW settings.

## Compatibility and load evidence

Probe exact service/backend configurations without comparing their speed:

```bash
python -m budgetroute compatibility-matrix --config configs/serving/fake.yaml --config configs/serving/local-openai-compatible.yaml
```

Run load only against endpoints you own or are authorized to test:

```bash
python -m budgetroute load-test --target https://budgetroute.example --requests 1000 --concurrency 32 --api-key-env BUDGETROUTE_LOAD_KEY
```

Targets, environment, request mix, actual measurement wall time, per-HTTP-request latency, samples, errors, overload, and the claim boundary are saved. The generator is a fixed-request closed-loop concurrency test, not an open-loop arrival model. The autoscaling multiplier is a heuristic starting point, not a deployment command or capacity guarantee.

## CLI

`budgetroute` and `python -m budgetroute` expose:

```text
doctor                 inspect dependencies, hardware, config, and output access
validate-config        resolve and validate composed YAML
security-check         audit deployment-sensitive settings without printing secrets
inspect-data           validate benchmark data and manifests
materialize-dataset    download a pinned public split and hash it
materialize-router-test-split
                       export persisted router test IDs with a derived manifest
build-index            build the configured exact/FAISS/HNSW index
benchmark              execute policies and persist experiment artifacts
collect-baselines      collect content-addressed model outcomes
replay-benchmark       compare policies without loading model weights
verify-replay          compare live and cached semantic responses
train-router           train calibrated small-success or paired-quality routing
evaluate-router        evaluate a saved router on artifacts
calibrate-confidence   fit an offline backend-confidence calibrator
train-retrieval-router train paired retrieval-benefit routing
adapt-confidence       create and gate a delayed-label calibration candidate
rollback-calibration   restore a prior promoted calibrator
compatibility-matrix   probe exact runtime configurations
load-test              generate authorized multi-endpoint HTTP load evidence
audit-check            verify the operational audit hash chain
generate-report        render reports from saved artifacts only
serve                  start FastAPI
demo                   run the five-case deterministic demo
```

Use `python -m budgetroute COMMAND --help` for exact options.

## Evaluation and reproducibility

Quality metrics include exact match, token F1, numeric/classification/keyword correctness, abstention, selective accuracy, routing and escalation diagnostics, Brier score, ECE, and reliability bins. Systems metrics include separated queue/routing/retrieval/generation/escalation/total latency, actual batch size, measured wall throughput, tokens, RSS/CUDA memory, failures, overload, and route shares.

Grouped deterministic bootstrap intervals quantify sampling uncertainty, and results below `benchmark.minimum_samples_for_claims` carry a claim warning. These controls cannot make an unrepresentative sample representative. See [evaluation](docs/evaluation.md), [benchmarking](docs/benchmarking.md), and [reproducibility](docs/reproducibility.md).

## Development

```bash
python -m compileall src tests
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest --cov=budgetroute --cov-branch --cov-report=term-missing --cov-fail-under=75
python -m build
python -m twine check dist/*
```

Default tests are deterministic and offline. Do not commit credentials, model weights, downloaded datasets, built indexes, operational databases, calibration registries, or ordinary outputs.

## Security boundary

The repository provides scoped authentication, input bounds, configurable literal-content rules, API/database TLS configuration checks, trusted hosts, quota/admission controls, sanitized errors/config, tenant-scoped operational records, and security automation. It does not replace an identity provider, secret manager, managed-database HA/backup program, WAF, malware scanner, model guardrail, service mesh, external audit archive, or privacy/compliance program. Literal blocklists are narrow defense-in-depth controls and do not solve prompt injection.

See [deployment](docs/deployment.md), [SECURITY.md](SECURITY.md), and the detailed [limitation matrix](docs/limitations.md).

## Repository map

```text
src/budgetroute/      backends, routing, retrieval, operations, adaptation, API, evaluation
configs/              model, dataset, routing, retrieval, benchmark, and serving profiles
deploy/               reference Kubernetes topology and operator notes
data/                 small original development fixtures; generated state is ignored
tests/                offline unit, integration, and timing tests
docs/                 architecture, operations, methods, decisions, and limitations
scripts/              PowerShell and Bash setup/check/demo helpers
.github/workflows/    CI, packaging, fake smoke, CodeQL, dependency review/audit
```

## Resume use

The architecture and test evidence can be described today. Replace every quality, latency, throughput, savings, recall, and scale placeholder with a link to a real immutable run. Never present fake-mode values as model evidence. See [portfolio notes](docs/portfolio-notes.md).
