# Limitations and mitigation matrix

The v0.7 multi-host control plane reduces a major architectural gap, but software cannot erase missing evidence, external infrastructure, or fundamental model uncertainty. The table distinguishes what is implemented from what must still be supplied.

| Area | Repository mitigation | Remaining boundary / required evidence |
|---|---|---|
| Limited development evidence | Revision-pinned public adapters, content hashes, group-safe splits, grouped bootstrap intervals, minimum-sample claim warnings, a 500-pair MMLU collection, and one persisted 100-record held-out routing evaluation | The held-out sample is still drawn from one MMLU development universe; external data, larger samples, repeated studies, domain review, and external validity remain required |
| Fake backend | Every response/artifact is explicitly fake and workflows match real interfaces | Fake quality, confidence, and latency are never model or capacity evidence |
| Model/hardware variance | Exact revisions/config/environment metadata, compatibility probes, warm/cold separation, real HTTP load artifacts, and a named RTX 3060/Qwen2.5 result | Quality and speed remain specific to weights, prompt, runtime, drivers, hardware, traffic, background load, and thermals |
| Confidence calibration | Delayed-label joins, chronological holdout, Brier/ECE gates, immutable promotion, change points, and rollback | Calibration can fail under future shift; representative recent labels and post-deployment monitoring remain mandatory |
| Drift | Numeric mean/quantile and category-distribution drift plus durable feature summaries | These do not prove semantic, label, factuality, abuse, or safety drift |
| Deterministic evaluation | Final-answer numeric extraction supports decimals/scientific notation/fractions/percentages; keyword matching respects phrases; bootstrap uncertainty is recorded | Exact metrics still miss open-ended usefulness, factuality, style, and safety; use domain experts or a validated optional judge |
| Batching and throughput | Actual batching/concurrency execute, benchmark throughput uses wall time, and load tests report p50/p95/p99, errors, overload, and claim boundaries | No speedup/capacity claim exists until measured on the target deployment; the autoscaling multiplier is heuristic |
| Retrieval scalability | Exact NumPy/FAISS plus configurable HNSW and revision-pinned semantic embeddings | Measure HNSW recall/latency, corpus freshness, licensing, and retrieval-induced answer quality on the target corpus |
| Retrieval usefulness | A paired, group-safe learned retrieval-benefit policy estimates whether context improves quality | Similarity/benefit scores are not causal guarantees; changed corpora/models require retraining and holdout evaluation |
| Replica coordination | SQLite WAL provides same-host coordination; PostgreSQL provides pooled multi-host quotas, expiring leases, counters, labels, reviews, and serialized audit writes inside one database consistency domain | No multi-region active/active consensus, automated failover/fencing, or capacity evidence; validate the managed database and target topology |
| Schema lifecycle | Ordered transactional PostgreSQL migrations use a global lock and recorded SHA-256 checksums; production can separate DDL and runtime credentials | Migrations are forward-only; expand/contract design, backup/restore, compatibility releases, and rollback drills remain operator responsibilities |
| Database availability | Startup and `/readyz` verify operational storage, and protected requests never fall back silently to local quota state | The repository does not provision PostgreSQL HA, replication, connection proxying, backups/PITR, or an SRE on-call process |
| Durable scheduling | Global leases bound inflight work and recover on expiry; local queues are bounded and deadline-aware | Prompts are intentionally not persisted, so this is not a durable job queue and disconnected requests are not replayed |
| Tenant access | Environment-backed subject keys, scopes, constant-time comparison, tenant-filtered records, and separable PostgreSQL DDL/runtime roles | No OIDC/MFA/user lifecycle, PostgreSQL row-level-security policy, row encryption domain, billing, or compliance certification; use an IdP/secret manager and hardened datastore |
| Human review | Durable privacy-minimal cases, claim/resolve/versioning, idempotent labels, and audit events | No notification/SLA/ticketing UI or workforce integration; connect an external system when humans must actually review |
| Audit | Hash-chained events and continuity verification with bounded retention | A database administrator can replace the store; export to an independently controlled append-only/notarized sink for stronger assurance |
| Transport | Non-loopback config requires credentials and built-in TLS or explicit external termination | Edge certificates, network isolation, DDoS/WAF/service-mesh policy, and header trust are operator responsibilities |
| Content safety | Metadata bounds and configurable input/output literal rejection without echoing blocked content | Literal rules do not solve prompt injection, jailbreaks, malware, privacy leakage, factuality, or semantic policy enforcement |
| OpenAI-compatible runtimes | Safe URL/credential rules and exact single/batch compatibility artifacts | Vendors vary in seeds, logprobs, tokenization, batching, errors, and extensions; probe every exact runtime revision |
| Learned artifact supply chain | Metadata records source hashes, versions, splits, and label definitions | `joblib` loading is executable deserialization; accept artifacts only from trusted, integrity-verified build paths |

## Scientific interpretation

Bootstrap intervals quantify sampling variation for the observed independent groups; they do not correct selection bias or dataset mismatch. Minimum-sample warnings prevent silent overclaiming but do not make a threshold universally valid. Cache replay is excellent for comparing routing over fixed model outcomes and cannot reproduce live contention, batching, queueing, or thermal behavior.

## Operational interpretation

The supplied SQLite and PostgreSQL profiles exercise real transactional control-plane behavior, and the PostgreSQL CI job stresses concurrent quota, lease, feedback, migration, and audit semantics. This is not internet-scale capacity, availability, or disaster-recovery evidence. The stores deliberately keep prompts/answers out of operational state, which limits forensic and reviewer context but reduces retained sensitive data. Deployments must choose that trade-off explicitly and can integrate a separately governed content store if their threat model and retention policy allow it.

## Claims policy

Do not claim universal “production readiness,” solved prompt injection, model safety, semantic correctness, cost savings, HNSW scale, throughput, or latency from repository code or fake artifacts. Claims must name the exact dataset, sample size/groups, model and revision, prompt template, runtime, hardware, load shape, metric definition, uncertainty, and artifact location.
# Development pilot boundaries

The notebook/orchestrator alone is software, not a positive routing result. On October 22, free Colab supplied a T4 and the repaired `5ab58a8` component gate passed real primary Qwen3.5 generation, pinned MiniLM/FAISS retrieval, scheduling, authenticated loopback API and cleanup. Paired pilot collection and final evidence verification remain in progress. Free GPU availability/session lifetime remain external gates. Primary failures must be retained; the prespecified Qwen3 fallback uses a separate identity/workspace. No quantization, token-budget or loader changes may silently rescue a measured run.

The 40 diagnostic questions cannot support a two-point quality claim. MMLU may overlap models' pretraining and does not establish production generalization. Co-resident always-large timing is not an optimized single-model deployment. A matched-random comparator controls cheap-route count, not every runtime interaction. Unknown rates or token usage keep total estimated cost unknown; configured rates are not invoices.

Colab storage is ephemeral. Compatible retained sessions can resume collection, but changed hardware/package/source/data identities require a fresh run. Timed partial blocks restart; cached timing never counts as live. HTTP runtime compatibility, managed-database failover/restore and internet-scale deployment evidence remain separate from this free-Colab scientific pilot.

