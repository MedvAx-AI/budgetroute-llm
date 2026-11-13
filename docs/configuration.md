# Configuration

Backend identity fields include `runtime_identity` (required for cached HTTP execution), `model_loader` (`causal_lm` by default, or `image_text_to_text`), and optional `max_input_tokens`. `generation.enable_thinking` is an explicit optional mode flag; it belongs to cache identity and modern execution profiles, not an inferred checkpoint default. The local HTTP example identities are illustrative and must be replaced with exact deployment identities before claiming reproducible results. Uncached legacy HTTP configurations remain accepted. Modern loader/mode execution support is validated separately from configuration parsing.

Runtime behavior is defined by strict Pydantic models loaded from YAML. `extends` accepts one path or an ordered list; mappings are deep-merged left to right and the current file wins. Missing files, cycles, unknown fields, invalid types, and unsafe combinations fail with actionable errors.

## Main sections

- `models`: fake, Transformers, or OpenAI-compatible backend profiles. Real model/tokenizer revisions can be required and are recorded.
- `retrieval`: corpus/index paths, fake or Transformers embedder, exact/FAISS/HNSW index, exact embedding revision, chunking, score threshold, and HNSW graph/search parameters.
- `routing`: policy, difficulty/cascade/retrieval thresholds, learned artifacts, load/cost weights, and review behavior.
- `batching`: local queue size, maximum batch, first-item wait, admission timeout, and end-to-end deadline.
- `api`: bind address, trusted hosts, request limits, authentication sources, review/feedback switches, and built-in or external TLS boundary.
- `operations`: memory/SQLite/PostgreSQL backend, SQLite path, environment-named PostgreSQL DSN, TLS/migration/pool controls, replica ID source, shared quota, global inflight leases, and audit retention.
- `adaptation`: delayed-label registry, minimum sample count, chronological holdout, Brier/ECE gates, selective target, and Page-Hinkley settings.
- `content_policy`: optional literal input/output rules plus always-enforced metadata byte/depth/key bounds.
- `monitoring`: bounded window, baseline size/file, and drift threshold.
- `benchmark`: dataset, policies, warm-up/measured runs, concurrency/batch size, cache, bootstrap resamples, and minimum sample count for claims.

## Safety validation

Configuration rejects, among other cases:

- fake/real backend mismatches, missing model IDs, unpinned revisions when pinning is required, and invalid device/precision/quantization combinations;
- credential-bearing or malformed model-server URLs; remote servers require explicit opt-in and HTTPS;
- learned policies without artifacts and retrieval policies without retrieval;
- invalid chunk/HNSW settings and Transformers embeddings without model ID and exact revision;
- non-loopback API binding without an API credential source and either a certificate/key pair or `external_tls_termination: true`;
- partial TLS configuration, missing certificate/key files, missing environment credentials at runtime, and unavailable optional dependencies.

`external_tls_termination` is an operator assertion, not automatic TLS discovery. It is appropriate only when a trusted proxy/load balancer terminates HTTPS and the application port is network-restricted.

## Credential formats

Legacy single-key mode names `api.api_key_env`, normally `BUDGETROUTE_API_KEY`. Tenant mode names `api.tenant_keys_env`, normally `BUDGETROUTE_TENANT_KEYS_JSON`. The JSON value has this schema:

```json
{
  "tenants": [
    {
      "tenant_id": "acme",
      "subject": "reviewer-1",
      "api_key": "a long random secret",
      "scopes": ["inference", "feedback", "review"]
    }
  ]
}
```

Tenant and subject identifiers are bounded strings. Keys and subject/key pairs must be unique; multiple subjects may belong to one tenant. `admin` implies all scopes. Tenant mode is exclusive: when `tenant_keys_env` is configured, an inherited legacy-key variable is not accepted as a global bypass. Credential values are loaded at process construction and compared in constant time. Rotate by changing the secret source and restarting replicas; support overlapping old/new subjects when a no-downtime rotation is needed.

## Environment overrides

```text
BUDGETROUTE_OUTPUT_DIR
BUDGETROUTE_HOST
BUDGETROUTE_PORT
BUDGETROUTE_LOG_LEVEL
BUDGETROUTE_API_KEY
BUDGETROUTE_TENANT_KEYS_JSON
BUDGETROUTE_POSTGRES_DSN
BUDGETROUTE_REPLICA_ID
```

The first four are direct overrides. The remaining variables are read only when named by the selected config. The project does not implicitly load `.env`; use a shell, secret manager, container runtime, or orchestrator. Sanitized config responses expose credential source names and TLS state, never secret values or arbitrary environment contents.

## Recommended profiles

- `configs/serving/fake.yaml`: local no-network development.
- `configs/serving/secure.yaml`: authenticated service behind explicit external TLS termination.
- `configs/serving/distributed.yaml`: tenant scopes, SQLite coordination, durable feedback/review/audit, adaptation registry, and content rules.
- `configs/serving/postgres.yaml`: TLS-required multi-host PostgreSQL coordination with runtime schema verification.
- `configs/serving/postgres-local.yaml`: explicitly unencrypted loopback-only Compose development override.
- `configs/serving/postgres-adaptation.yaml`: operator-side delayed-label adaptation over PostgreSQL observations.
- `configs/retrieval/semantic-hnsw.yaml`: revision-pinned semantic embedding and HNSW fragment; compose it into a serving/benchmark profile.
- `configs/routing/learned-retrieval.yaml`: learned retrieval-benefit fragment; set its trained artifact path.
- `configs/benchmarks/real-gpu-mmlu-500-live.yaml`: paired small/large outcome collection for router development.
- `configs/benchmarks/real-gpu-mmlu-500-learned-live.yaml`: live learned/cascade comparison restricted to the persisted router test split.
- `configs/benchmarks/real-gpu-mmlu-500-learned-replay.yaml`: model-free replay of that held-out comparison from a complete cache.

Leaving `routing.retrieval_benefit_threshold` unset uses the calibration-selected value stored in the artifact; set it only as an explicit reviewed override.

Run both `validate-config` and `security-check` against the exact deployment config and environment before serving.

PostgreSQL DSNs are read only from `operations.postgres_dsn_env` and never appear in sanitized config. Production defaults require `sslmode=require`, `verify-ca`, or `verify-full`; prefer `verify-full`. Pool limits are per process, so total possible connections grow with replica count. Set `postgres_auto_migrate: false` for production, run `budgetroute migrate-store` with a DDL-capable credential, and give serving replicas a narrower runtime credential. See the [PostgreSQL operations runbook](postgres-operations.md).
# Semantic feature identity

Pilot semantic training pins `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, 256 input tokens. `train-router --help` exposes explicit split, prompt dataset and encoder settings; semantic `evaluate-router` also requires `--dataset`. Never load router joblib files from untrusted sources: they are executable Python artifacts.

Modern pilot profiles: `configs/models/colab-qwen35.yaml` pins Qwen3.5-0.8B/4B, selecting the built-in image-text-to-text loader for text-only requests; `colab-qwen3-fallback.yaml` separately pins Qwen3-0.6B/4B-Instruct-2507. Official Qwen Hub metadata resolved the full model/tokenizer revisions and Apache-2.0 licenses on 2026-10-21 ([primary small](https://huggingface.co/Qwen/Qwen3.5-0.8B), [primary large](https://huggingface.co/Qwen/Qwen3.5-4B), [fallback small](https://huggingface.co/Qwen/Qwen3-0.6B), [fallback large](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)). Both profiles use FP16, no quantization/compilation, explicit non-thinking greedy generation, 2,048 input tokens and 64 output tokens. Over-limit input fails before generation rather than silently truncating. These are feasibility candidates, not validated speedups; unsupported loaders fail explicitly without remote code execution.

`load_direct_to_device` defaults to false for backward compatibility. The Colab profiles explicitly set it true: `from_pretrained(device_map='cuda')` loads onto the requested single device rather than materializing the complete checkpoint in host RAM and moving it afterward. This is not automatic CPU/disk offload and does not change precision, model class or decoding. Use the `all`/`gpu` installation with Accelerate. The setting is recorded in backend metadata and cache identity. See the [official device-placement documentation](https://huggingface.co/docs/transformers/v5.20.2/hardware). A retained real Colab host-RAM crash motivated this loading-path change; its new configuration must pass fresh component gates before collection.

