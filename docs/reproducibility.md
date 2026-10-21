# Reproducibility

Generation caches now use schema v2, incorporating a hashed endpoint namespace, operator-declared runtime/deployment identity, logprob settings, loader, context limit, and explicit generation/template settings. Credentials and request IDs are not identity inputs. Old v1 entries are never silently reused: recollect rather than copy old files into a new cache layout. HTTP caches require a nonempty `runtime_identity`; operators must identify an immutable server build and deployed weights, not reuse an alias across redeployments. This declaration is provenance, not cryptographic attestation. A returned model mismatch is rejected; absent returned model/system identity stays unavailable. Historical result cards remain unmodified evidence for their original source/cache contract.

Each run directory is named with a UTC timestamp and workload slug, with a numeric suffix on collision. It contains:

```text
config.resolved.yaml  exact effective configuration
environment.json     Python, OS, CPU/memory, optional dependencies, PyTorch/CUDA/GPU
run.json             identity, hashes, Git, seed, workload, backend metadata, status
metrics.json          aggregate and per-policy metrics
predictions.jsonl     answer, reference, evaluator, quality, route, usage, error
routes.jsonl          policy, route, confidence, feature/load snapshot, estimates, escalation/review
timings.jsonl         queue/batch/routing/retrieval/generation/escalation/total timing
errors.jsonl          failed request records
report/               generated Markdown, CSV, and figures
```

Configuration and dataset SHA-256 hashes identify inputs. Git commit and dirty state are recorded when a commit exists. Backend metadata records fake/Transformers/OpenAI-compatible type, local endpoint (never its credential), device request/resolution, precision, model/tokenizer IDs and exact requested/resolved revisions, model license/card, generation settings, concurrency slots, configured cost rates, load telemetry, artificial delay, initialization/warm-up time, and optimization request/success/first compiled execution as applicable. When configured, `run.json` embeds the validated dataset manifest and cache/replay mode.

Generation cache entries contain their complete backend and request fingerprints plus a typed generation envelope. Entry paths derive from SHA-256 keys, and reads revalidate both key and backend fingerprint. Replay artifacts keep the original generation measurements and separately record replay overhead.

Atomic temporary-file replacement reduces partial final JSON/YAML/JSONL files. A run status reports completion with or without request errors. Reports read the artifact directory and never fill missing metrics with invented values.

Load-aware choices depend on the observed concurrent schedule, so exact route reproduction additionally requires the same batch/concurrency/load pattern; every observed snapshot is stored in `routes.jsonl`. Cost units are configuration inputs. Drift/feedback service state is intentionally not part of benchmark artifacts unless a deployment exports it separately.

Compatibility runs create an immutable `compatibility.json` containing the exact sanitized configurations, environment, single/batch observations, metadata, source hash, and explicit claim boundary. Load runs create ordered samples and a summary with sanitized targets, request mix, concurrency, warm-up, measured wall interval, latency/error/overload results, credential *source name* only, and advisory capacity math. Neither artifact contains credential values.

Operational SQLite/PostgreSQL state is not a benchmark artifact. PostgreSQL migration names/checksums prove schema identity, not benchmark provenance or capacity. Calibration candidates carry a source hash, chronological split counts, calibrator and threshold, candidate/active holdout metrics, promotion gates, change points, and immutable version ID. The registry manifest records promotion/rollback history and atomically materializes `active-calibrator.json`. Retrieval-router artifacts similarly retain paired label definition, group partitions, source hash, feature order, calibration, threshold, and versions.

For a reproducible real comparison, use committed revision-pinned model, tokenizer, embedding, dataset, and runtime profiles; retain environment lock information; use consistent hardware and scheduler state; run `verify-replay`; state independent group/sample counts and uncertainty; and publish immutable experiment artifacts separately from the source repository.
# Pilot identity and restart boundaries

The development pilot freezes source SHA/clean state, resolved configuration, dataset/split/exclusion hashes, generation fingerprints, hardware identity and the complete installed-version inventory before measured stages. `BUDGETROUTE_REVIEWED_SHA` must explicitly match the clean checkout. Changing any identity requires a new workspace. This strict rule permits resumption within a compatible retained VM, not automatic mixing across replacement Colab GPUs or changed packages. Hashes provide integrity and traceability, not third-party execution attestation.

The source notebook is intentionally unexecuted in Git. Its structure and adapters are checked offline; only an exported executed Colab notebook plus validated real artifacts can establish remote completion. Cached baseline quality is distinct from cache-disabled online latency. Repeated timings preserve original question groups, and descriptive intervals do not certify non-inferiority.

