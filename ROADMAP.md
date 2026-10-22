# Roadmap

Completed milestones are marked explicitly. Completion means the repository capability and its offline validation exist; it does not imply universal model quality, hardware performance, or production certification.

## v0.1 — functional benchmark and API (complete)

- [x] Fake/Transformers benchmark path, typed artifacts, API, packaging, and a non-comparative CPU acceptance run.

## v0.2 — datasets, replay, and calibration (complete)

- [x] Licensed revision-pinned datasets, integrity manifests, content-addressed generation replay, group-safe router training, and confidence calibration.

## v0.3 — scheduling and backends (complete)

- [x] Bounded deadline-aware batching, padded Transformers batches, load-aware routing, and local OpenAI-compatible runtimes.

## v0.4 — operational and research guardrails (complete)

- [x] Multi-objective routing, overload handling, API security controls, Prometheus metrics, feature drift, and GitHub security automation.

## v0.5 — distributed operational control plane (Phase 6, complete)

- [x] Transactional SQLite quotas and expiring global admission leases shared by same-host replicas.
- [x] Environment-backed tenant credentials, scoped authorization, and tenant-isolated review/feedback queries.
- [x] Durable delayed labels, human-review cases, shared counters, and hash-chained audit events with verification.
- [x] Explicit built-in TLS or trusted external TLS termination for non-loopback serving.
- [x] Configurable prompt/output substring rules and bounded metadata validation without logging rejected content.

## v0.6 — adaptation and deployment validation (Phase 7, complete)

- [x] Chronological delayed-label recalibration, Page-Hinkley change points, gated promotion, immutable versions, and rollback.
- [x] Learned retrieval-benefit routing from paired outcomes with group-safe partitions and safe fallbacks.
- [x] Revision-pinned semantic embeddings and optional FAISS HNSW approximate search.
- [x] Reproducible runtime compatibility probes and multi-endpoint HTTP load artifacts with measured wall throughput.
- [x] Bootstrap uncertainty, claim-size warnings, stronger deterministic answer extraction, and richer feature/category drift.
- [x] A documented mitigation matrix for repository, deployment, evidence, and irreducible model-safety limitations.

## v0.7 — multi-host operational control plane (complete)

- [x] Pooled PostgreSQL implementation of the complete `OperationalStore` protocol.
- [x] Atomic cross-host quotas, database-clock expiring leases, optimistic review transitions, idempotent observations, and serialized audit-chain appends.
- [x] Ordered transactional migrations with advisory locking, checksum verification, and separate migration/runtime deployment modes.
- [x] PostgreSQL dependency/readiness/TLS validation plus real database parity and contention CI.
- [x] Local Compose and three-replica Kubernetes topology references with probes, PDB, HPA, security contexts, and secret placeholders only.
- [x] A production operations runbook covering roles, pool budgets, rollout, monitoring, backup/PITR, restore, and rollback boundaries.

## Next evidence milestones

### Research-first direction (2026-10-21; approved, Native implementation in progress)

See the [relevance/methods audit and proposed study](docs/research-reassessment.md). Use free Colab only. Freeze additional infrastructure expansion while reviewing this direction; existing completed engineering milestones are not scientific-success claims.

- [x] Reassess recent routing literature, modern small-model candidates, the historical negative result, and remote validation gaps.
- [x] Refresh existing-source remote CPU/API/storage contracts and real PostgreSQL CI on October 21.
- [x] Accept the narrow research direction and write the [pilot specification](docs/superpowers/specs/2026-10-21-colab-routing-study-design.md); verify signed-in Colab access through the in-app browser only.
- [x] Approve the written specification and write the [implementation plan](docs/superpowers/plans/2026-10-21-colab-routing-study.md).
- [x] Approve the implementation plan and choose Native execution with one independent final review.
- [x] Repair infeasible-threshold fallback in newly trained artifacts and single/batch serving; preserve uncertified legacy readability.
- [x] Separate router-target probabilities from answer confidence and remove fabricated routing/answer calibration.
- [x] Version generation-cache identity and record/validate remote model/runtime provenance before remote collection.
- [x] Record per-backend calls and preserve unknown/partial usage and cost instead of reporting missing cost as zero.
- [x] Repair infeasible-threshold fallback, router/answer confidence metrics, cache identity, missing-cost semantics, and grouped/paired uncertainty before confirmatory claims.
- [x] Implement the independent 120/40/40 development splits, shared semantic features, modern pinned adapters and resumable thin Colab notebook.
- [x] Allocate free T4 and preserve the first real component attempt (generation, semantic, retrieval and scheduling passed; API-probe failure retained). Full gate rerun remains required.
- [ ] Execute a development-only modern-model feasibility pilot on free Colab and determine whether an adequately powered study fits the available budget.
- [ ] Freeze fresh splits, primary policy, scoring, stopping rules, and the complete non-inferiority/latency decision rule before examining final outcomes.
- [ ] Execute and preserve real remote component and study evidence; publish positive, negative, infeasible, or inconclusive outcomes without post-hoc changes to success criteria.

### Existing evidence and deployment backlog

- [x] Publish an initial license-compliant comparison on named models, data, hardware, runtime revisions, and a clean source commit: Qwen2.5-0.5B/1.5B on stratified MMLU-100 using an RTX 3060 Laptop GPU.
- [x] Expand paired baseline collection to 500 records and publish a trained/calibrated router on a persisted 100-record untouched split; the negative result rules out a quality-retention claim for the current feature set.
- Add a new external/later-seeded holdout, paired-delta uncertainty, randomized repeated timing trials, and batch/concurrency matrices before making quality-retention or capacity claims.
- Validate PostgreSQL pool/lock/query behavior, failover, recovery, and autoscaling under an authorized target-fleet load matrix; code and CI do not establish fleet capacity or availability.
- Design multi-region topology only against an explicit consistency/RPO/RTO requirement; the current adapter intentionally targets one PostgreSQL primary/consistency domain.
- Integrate an external identity provider, secret manager, WAF/service mesh, centralized telemetry, and ticketing system for sensitive production use.
- Add domain-specific semantic/human evaluation and red-team evidence; no generic filter can prove factuality, usefulness, or prompt-injection safety.
- Validate HNSW recall/latency and autoscaling behavior on the target corpus and fleet before setting production thresholds.
# Active research implementation

Completed software foundations: fail-closed thresholds, typed probability targets, cache/runtime identity, per-call cost provenance, failure-aware paired evaluation, independent pilot data, shared semantic gates, modern adapters and resumable Colab orchestration. Actual remote component/pilot execution and exported evidence remain the next gates; confirmation remains separately approval-gated.

