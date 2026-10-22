# Changelog

All notable changes follow Keep a Changelog conventions. The project uses semantic versioning.

## [Unreleased]

### Fixed

- The Colab loopback API probe now gives its injected service the same temporary configuration as its app, without mutating the caller. A real T4 probe exposed the previously disabled feedback flag; an in-process ASGI regression reproduces it without claiming GPU or network evidence.

- Per-call usage/cost accounting includes both cascade executions without double-counting accepted small answers. Missing/partial reported usage and undeclared rates yield unknown totals, with known subtotals preserved. Default policy forecast rates are not treated as measured cost.

- Generation-cache schema v2 separates endpoint/runtime/logprob/template/loader/context identity and rejects old-schema entries. Cached HTTP backends require a deployment identity; returned model mismatches and non-object JSON responses fail explicitly, while absent identity remains unavailable.

- Router probability/target records and calibration are separate from final-answer confidence. Missing probability evidence yields unavailable calibration rather than complemented large-answer confidence or binary route scores. Prediction records preserve explicit fake/real identity when known.

- New router and backend-confidence artifacts explicitly disable cheap acceptance when calibration is infeasible or certification failed/is insufficient. Numeric overrides cannot bypass this state; single and batch cascades always escalate. Legacy numeric artifacts remain readable and uncertified.

### Added

- Research reassessment covering current routing literature/model candidates, confirmed methodological defects, a proposed free-Colab replication-and-extension direction, and fresh October 21 remote CI evidence. The scientific redesign is pending approval; no runtime fixes, model migration, or Colab GPU result are claimed.
- A written measurement-repair and free-Colab pilot specification for review, following acceptance of the research direction. In-app-browser Colab access is verified; GPU execution and implementation remain pending.
- An approved test-first implementation plan, now executing natively; remote GPU execution remains pending.

### Changed

- GitHub checkout, Python setup, and artifact upload workflows now use current Node 24-based major releases, removing deprecated Node 20 action execution.

## [0.7.0] - 2026-08-28

### Added

- A pooled PostgreSQL operational store for multi-host quotas, admission leases, metrics, predictions, delayed labels, review workflows, and serialized hash-chain audit writes.
- Ordered checksum-verified PostgreSQL migrations, database-clock lease/quota decisions, startup TLS validation, and database-aware readiness responses.
- Secret-suppressed PostgreSQL DSN/startup failures and an installed-wheel CI assertion for packaged migration resources.
- Hugging Face Datasets 5.0.1+ to exclude vulnerable archive-extraction and path-traversal releases from optional evaluation installs.
- PostgreSQL parity and high-contention integration tests backed by a real PostgreSQL 17 GitHub Actions service.
- Production and local PostgreSQL profiles, a two-service Compose topology, and a security-hardened three-replica Kubernetes reference with Service, PDB, HPA, probes, and ingress policy.
- An operations runbook covering least privilege, TLS, migrations, pool budgets, backup/PITR, rollout, rollback, monitoring, and failure drills.
- Deterministic subject-stratified MMLU materialization, a cache-isolated warmup marker, pinned live/replay RTX 3060 benchmark profiles, and a curated Qwen2.5/MMLU-100 result card with machine-readable evidence.
- Paired-quality learned-router labels, held-out end-to-end routing outcome metrics, deterministic persisted-test export with a derived integrity manifest, and MMLU-500 development/held-out benchmark profiles.
- A curated clean-commit Qwen2.5/MMLU-500 study with 500 paired baselines, disjoint 300/100/100 router partitions, an untouched 100-record live comparison, replay verification, and an explicitly negative learned-routing conclusion.

### Fixed

- Shared tenant quota enforcement and delayed-label adaptation now treat PostgreSQL as a durable backend instead of falling back to process-local behavior.
- Transformers cleanup now releases runtime references and forces cyclic collection between repeated replay-policy model loads, preventing the observed Windows `torch_cpu.dll` verifier crash.

## [0.6.0] - 2026-08-07

### Added

- Transactional SQLite coordination for replica-shared quotas, global admission leases, aggregate metrics, delayed labels, durable human-review cases, and hash-chained audit events.
- Environment-backed multi-tenant credentials with scoped inference, feedback, review, and administrative access; tenant data remains isolated at API queries.
- Chronological delayed-label confidence adaptation with Page-Hinkley change points, immutable candidates, explicit promotion gates, atomic activation, and rollback.
- Learned retrieval-benefit routing trained from paired baseline/retrieval evidence with group-safe partitions and calibrated thresholds.
- Revision-pinned Transformers embeddings, optional FAISS HNSW search, runtime compatibility probes, and multi-endpoint HTTP load artifacts.
- Bootstrap uncertainty intervals, minimum-sample claim warnings, improved numeric/fraction/percentage extraction, phrase-safe keyword evaluation, and measured wall-clock throughput.
- Configurable prompt/output substring policy and bounded metadata validation, built-in TLS options, explicit external TLS termination, audit verification, and a distributed serving profile.

### Changed

- Runtime drift combines numeric mean/quantile shifts with category-distribution total variation.
- Security validation now fails non-loopback configurations without scoped authentication and an explicit TLS boundary.
- Documentation distinguishes mitigated repository limitations from deployment-specific evidence and infrastructure requirements.

## [0.4.0] - 2026-08-07

### Added

- Bounded deadline-aware inference scheduler with queue telemetry, overload rejection, and orderly shutdown.
- Ordered small/large backend-wave batch execution and padded Transformers generation.
- OpenAI-compatible local-runtime backend with safe endpoint and credential configuration.
- Thread-safe backend load telemetry plus load-aware and quality/cost/latency-aware policies.
- Explicit human-review outcomes, configured cost estimates, Prometheus metrics, aggregate feedback, and rolling feature-shift detection.
- API-key authentication, trusted Host validation, body/prompt/rate limits, secure response headers, and a deployment security audit command.
- Dependabot configuration, CodeQL extended scanning, dependency review, `pip-audit`, and private vulnerability reporting guidance.

### Changed

- Fake smoke benchmarking now exercises ordered backend batching and the adaptive policies without presenting fake measurements as performance evidence.
- Reports include human-review counts and configured cost units.
- GitHub Actions use current Node 24-compatible checkout/setup actions.

## [0.2.0] - 2026-08-06

### Added

- Revision-pinned GSM8K, MMLU, and HotpotQA adapters with deterministic JSONL/corpus materialization and integrity manifests.
- Revision/license-pinned SmolLM2 CPU and Qwen2.5 GPU model profiles.
- Content-addressed generation collection, model-free replay, cache integrity checks, and live/replay agreement verification.
- Model-derived token likelihood/log-probability/entropy signals with separate raw and calibrated confidence.
- Group-aware train/calibration/test router partitions, scalar temperature calibration, calibration-only threshold selection, and confidence-calibrated cascade support.
- CLI workflows for dataset materialization, baseline collection, replay, replay verification, and backend-confidence calibration.

### Changed

- Learned-router evaluation now defaults to the persisted untouched test partition.
- Real reports explicitly distinguish live execution from cache replay.

## [0.1.0] - 2026-08-06

### Added

- Initial quality-aware routing, retrieval, inference, evaluation, reporting, CLI, API, and offline fake-mode implementation.
# Research measurement updates

- Align paired quality denominators, failure handling and observation-weighted cluster uncertainty; block unverified live latency ratios.
- Freeze historical exclusions and 120/40/40 development roles; add strict MCQ/truncation scoring, prompt-only semantic gates, pinned primary/fallback modern profiles, and a resumable free-Colab notebook with fail-closed real-evidence export.
- Repair current-kernel editable-source import in Colab, quote the cache-disabled YAML literal, and retain environment/configuration evidence on component initialization failure.

