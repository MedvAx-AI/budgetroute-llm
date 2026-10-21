# Free-Colab routing research: measurement repair and feasibility pilot

Date: 2026-10-21. Status: **written specification awaiting user review**. The user approved continuing the recommended scientific direction and restricted execution to free Colab and browser-only interaction. This is a replication-and-extension project, not a claim of a novel routing algorithm.

## 1. Intent and scope

Build a scientifically defensible, reproducible portfolio study using the existing BudgetRoute pipeline. Answer when a modern small/large model pair can save real inference time without materially degrading measured answer quality, and when calibration data or overhead makes routing infeasible.

This implementation scope is deliberately limited to necessary measurement repairs, one semantic-router comparison, a resumable Colab pilot, and honest remote component evidence. It does not authorize a bulk confirmatory experiment with an unspecified sample size. The pilot must produce a concrete, separately frozen confirmation protocol before any final-test outcomes are examined.

Constraints:

- Free Colab only: no subscription, paid GPU, commercial API, account creation, external public service, tunnels, or quota circumvention.
- Use the in-app browser for Colab; no native computer/desktop automation, local GPU inference, or local execution presented as remote testing.
- Keep normal tests offline and optional model dependencies lazy. Existing backend/policy/artifact boundaries and Windows support remain intact.
- Reuse existing dataset adapters, MiniLM embedding implementation, logistic classifier, cache/replay, manifests, and reporting rather than introduce another evaluation framework.
- Preserve historical Qwen2.5 reports/configurations unchanged. Do not publish agent instructions, local execution scratch, secrets, weights, or raw downloaded datasets.

## 2. Scientific claims and decision boundaries

Target primary hypothesis for the later confirmation: a frozen prompt-only semantic router has mean end-to-end latency at most 0.80 times always-large while its absolute accuracy difference versus always-large is greater than -0.02 on the same preregistered workload.

The feasibility pilot does **not** prove that hypothesis. Its deliverable is a measured decision about whether an adequately sized confirmation is feasible on free Colab, including an estimated sample requirement and execution budget.

Separate quantities:

- Absolute accuracy: every scheduled question counts; failures, missing responses, and unparseable/truncated answers score zero.
- Harmful substitution: cheap answer wrong and large answer correct.
- Beneficial substitution: cheap answer correct and large answer wrong.
- Relative paired quality difference: routed correctness minus always-large correctness, averaged over the same questions.
- Cheap coverage: fraction routed cheaply; not accuracy and not a resource saving by itself.
- Mean live latency: complete online request time, including feature extraction/embedding, route selection, tokenization, generation, decoding, and escalation if applicable.
- GPU inference time, wall time, tokens, memory, and unknown monetary cost: separate fields. Free compute is not zero computational cost. No billing/energy claim is made.

For eventual confirmation, reserve a one-sided error allocation of 0.025 per primary endpoint: accuracy lower bound must exceed -0.02 and latency-ratio upper bound must be below 0.80. The frozen protocol must identify and validate the paired accuracy and latency interval procedures, effective independent sample unit, repeated-trial design, power target, and fixed sample size before final evaluation. Do not substitute a tiny-sample percentile bootstrap for this requirement. A failed endpoint, insufficient precision, or resource exhaustion produces a negative/inconclusive result, not relaxed targets.

Risk certification is an additional guardrail, not equivalent to the two primary endpoints: an independently chosen threshold may be certified against a 5% harmful-substitution rate among cheap routes using a one-sided exact 95% binomial bound. This assumes independent calibration units and an unchanged, fixed model/policy. It is not a claim of 95% absolute correctness or safety under arbitrary distribution shift.

## 3. Experimental conditions

### Models and generation

First-choice pilot pair: `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B`. Feasibility fallback: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-4B-Instruct-2507`. Resolve immutable model/tokenizer revisions and exact supported loader/runtime dependencies before downloads and record them in the run manifest. No mutable `main` revision in a claimed reproducible run.

Start with explicit non-thinking generation, greedy decoding, FP16, no compilation, no quantization, batch size one, concurrency one, maximum 64 generated tokens, and a common short-context limit recorded in the resolved profile. Use a versioned direct-answer MCQ prompt requiring the final option letter. Render and retain the actual template configuration. A checkpoint's default thinking mode is not accepted as an explicit setting.

Check that a supported loader can run text-only requests for the chosen architecture without `trust_remote_code`. Do not silently repurpose a multimodal loader or discard outputs that fail the requested format. If the pair cannot initialize or operate within the allocated memory, record the failure, release it, and try the named fallback. A runtime/model switch creates a new run/cache identity; never merge performance results across pairs, quantization, hardware, or thinking modes.

Load both generators concurrently for steady-state routing measurements. If they cannot coexist with the gate, report that configuration infeasible; sequential baseline collection may still support quality analysis but not a resident-pair live latency claim. Keep cold-start/model-load measurements separate. Do not replace simultaneous residency with hidden weight swapping.

### Pilot data and policies

Use the existing revision-pinned `cais/mmlu` adapter, with a new versioned direct-answer prompt and deterministic subject-stratified selection of 200 development-only questions, seed 20261021. Exclude all original MMLU-500 IDs and normalized duplicate question/option groups. If the old ID list cannot be recovered and verified, stop data selection rather than assume a new seed guarantees separation.

Persist a parent manifest and disjoint pilot fitting/selection/diagnostic partitions of 120/40/40 independent questions before fitting. These are pilot partitions, never described as confirmatory. Their small sizes can legitimately make certification infeasible. Retain a separate untouched candidate pool for later confirmation; no reference answers/model outputs from that pool enter pilot decisions.

Collect paired always-small/always-large outcomes once with complete provenance. Compare:

1. Always-small and always-large.
2. Existing numeric lexical/shape-feature logistic router.
3. A single semantic logistic gate using the existing revision-pinned `sentence-transformers/all-MiniLM-L6-v2` representation.
4. Seeded random routing matched to the primary router's cheap-route count for quality comparison; for live latency use the same frozen count with an independently fixed assignment/order.

Baseline summaries may use all 200 pilot questions. Router fitting uses only the 120 fitting questions, threshold selection uses only the 40 selection questions, and held-out pilot policy comparisons use only the 40 diagnostic questions. Do not present routed performance on the fitting/selection questions as held-out performance, or interpret these 40 diagnostic questions as an adequately powered final test.

The gate uses prompt information only, never the expected answer, quality label, large-model output, or test outcomes. Paired labels treat both-wrong as non-harmful, not correct. Record input truncation rate and embedding time. Reuse the existing mean-pooling implementation and pin representation preprocessing; do not tune multiple embedding families on the final test.

The oracle is a diagnostic upper bound only. A confidence cascade is a secondary component validation/overhead comparison and does not become the primary policy because it looks better on diagnostic outcomes.

## 4. Required measurement and correctness changes

### Thresholds and certification

Represent threshold feasibility explicitly in router/calibrator artifacts. If selection cannot meet the requested constraint or certification is empty/insufficient/failed, route all requests large (cascade escalates all), with a machine-readable reason. Do not use `0.5`, NaN, or a non-JSON infinity sentinel as an implicit fallback. New artifacts and runtime behavior must agree; legacy artifacts remain readable but cannot be labeled certified.

For the pilot, empirical threshold choice uses the selection partition only; diagnostic risk intervals remain descriptive. For later certification, use a separate untouched certification split and a threshold fixed beforehand. A failed certificate cannot be repeatedly retried at other thresholds on that same split. Adaptive-search correction or a different certification method is out of the initial scope.

### Confidence and targets

Persist router probability, target name, calibrated/raw state, threshold, and feasibility separately from the final backend's answer confidence. Score each probability against its declared target. Never derive small-success probability by complementing large-answer confidence. Legacy absent fields mean unavailable, not fabricated probabilities. Keep absolute correctness and paired-harm targets distinct in reports.

### Cache and provenance

Version the cache contract for all generation/confidence-affecting configuration. Include an endpoint namespace without credentials, requested logprob settings, explicit template/thinking mode, and a required non-secret immutable runtime/deployment identifier for remote HTTP backends. Record model identity returned by the server where available and reject known mismatches. Changing deployments under a mutable identifier is outside the reproducibility guarantee and must be documented. Old cache entries do not silently satisfy the new identity.

### Resource and cost records

Aggregate actual backend executions, including both cascade calls, independently from policy forecasts. Preserve missing token usage, rate, or cost as unknown, never zero. A measured aggregate is unknown if any required contribution is unknown; partial known totals may be reported only with explicit completeness. No configured dollar estimate becomes an observed bill. Capture wall-clock latency and CUDA timing separately and keep replay timing visibly non-live.

### Uncertainty, grouping, and failures

Choose observation-weighted quality as the default estimand. Cluster resampling must recompute that same estimand from resampled clusters, not equally weight their means unless macro-group reporting is explicitly selected. Preserve original group IDs across repeats; repeats are not independent new questions. Add paired policy differences with question alignment, explicit missingness/failure counts, and a refusal to claim narrow effects from insufficient independent groups. Report successful-request-only metrics separately if retained for compatibility.

## 5. Minimal architecture and artifacts

Keep changes within current configuration, routing/training, schema, inference execution accounting, experiments, and evaluation modules. Introduce only the smallest reusable semantic feature path needed by both training and serving. Its model revision/preprocessing identity travels in the learned artifact and is checked on load. Old artifacts retain their numeric feature path; semantic features must not silently change an old router's inputs.

The Colab notebook is a thin orchestrator of versioned repository commands/functions, not an independent implementation of routing or scoring. It must select an exact source SHA, capture dirty state and installed versions, require real CUDA for GPU lanes, assert artifact success/zero errors, checkpoint collection by content identity, and expose explicit stop/resume steps. A default notebook run stops after the pilot and evidence export; it never automatically opens a confirmatory set.

Required persisted evidence: executed notebook, source SHA, installed package versions, resolved config/hash, exact dataset/model/embedding identities, split IDs/hashes, all predictions/errors/routes/usage, raw versus calibrated probabilities, feasibility/certification state, replay agreement, observed GPU/driver/CUDA/memory, timing order/trials, and an honest summary. Download an evidence bundle outside the ephemeral VM before claiming remote completion. Do not mount the user's entire Drive or create new credentials merely to save artifacts.

Collect paired baseline caches separately from timed online policy runs. Score reuse is legitimate for quality comparisons; runtime measurements cannot be synthesized by adding cached component timings. Keep feature caching/priming policy identical between compared live conditions and disclose whether the gate is warm. For subsequent confirmation, randomize/block policy order across repeated trials on the same hardware; repeated requests remain grouped by question.

## 6. Remote verification matrix

All required model evidence runs in a notebook-driven Colab runtime; CPU contracts run in GitHub Actions. Local static checks are allowed but are not counted as off-local proof.

| Lane | Acceptance |
| --- | --- |
| GitHub offline contracts and PostgreSQL | Existing quality matrix plus real database job pass at the exact changed source SHA; new regression cases cover every repaired behavior. |
| Colab environment | Actual remote Python/package versions and CUDA device recorded; unsupported or absent GPU fails explicitly. |
| Real generation | Both pinned models initialize, generate non-fake single responses, clean up, and run a small padded batch with correct order/token accounting. Compare single/batch answers and record any differences. |
| Confidence | Finite documented raw likelihood/logprob/entropy signals; router-target calibration remains separate from answer confidence. |
| Data and semantic gate | Real pinned dataset download/materialization and real MiniLM representation; disjoint split assertions; trained/reloaded gate produces equivalent decisions. |
| Cache/replay | Live/replay audit on a fixed sample has explicit field comparisons and no hidden downloads or generator initialization in replay. |
| Scheduling/API | Bounded batch/concurrency cases `(1,1)`, `(2,1)`, `(1,2)` plus authenticated loopback request, feedback, metrics, and graceful shutdown. No public port/tunnel or fleet-capacity claim. |
| Existing retrieval capability | Separate small real MiniLM plus FAISS/HNSW build/save/reload/search check. A fake embedder or mocked FAISS object does not satisfy this lane. Retrieval is disabled in the primary routing study. |
| HTTP-compatible runtime | Explicitly report untested unless an exact real model server is exercised. Mock-transport success is not runtime compatibility. This optional lane cannot block the core Transformers pilot. |

Compatibility commands that return exit code zero after collecting probe errors still fail acceptance when their artifacts report failure. A benchmark with `completed_with_errors` is not a successful pilot. Classify evidence per lane so one passing CI badge cannot imply all capabilities were tested.

## 7. Pilot completion and next confirmation gate

Pilot completion requires actual exported remote evidence, all core lane assertions, and a report of the 200 development pairs including discordance, absolute accuracy, coverage/harm, live mean/p50/p95 timing, memory, failures, and confidence missingness. If Colab availability prevents completion, retain checkpoints and report incomplete rather than silently switch to local hardware.

The pilot report must recommend proceed, change only development conditions, or stop. It must estimate the independent confirmation sample needed for the two-point margin from observed paired discordance, plan enough independent timed questions/repetitions for the latency endpoint, and estimate remaining free-runtime work from measured throughput. No nominal p99 claim from a tiny smoke sample.

Before a confirmation can start, a separate committed machine-readable protocol must contain final model/runtime/config hashes, exact source/data/split manifests, primary policy and threshold/certification rule, independent test size, interval algorithms/error allocation, power/precision rationale, timing design, seed list, maximum number of sessions/questions, exclusion rules, and the immutable decision/stopping rule. Approval of this specification does not authorize post-hoc tuning or mean those empirical fields are already known.

Calibration-budget and domain-shift studies are later extensions, using cached outcomes and predeclared splits where appropriate. Results on shifted data do not inherit an IID certificate. Neither extension nor infrastructure expansion is required for the first pilot to be honest and useful.

## 8. Documentation and review

Update README, roadmap, changelog, configuration/evaluation/reproducibility/real-evaluation instructions and limitations where implemented behavior actually changes. Keep speculative goals distinct from delivered capabilities. The reassessment remains the rationale; this spec defines scope; a reviewed execution plan will name files, tests, stages, and progress.

As of this specification: existing-source remote CI passed on October 21; the in-app browser successfully opened signed-in Colab. No GPU allocation, notebook creation/execution, modern-model migration, measurement repair, or new research result has been completed. Desktop automation is not used in this continuation.

Next process gate: user review of this written specification, then a written implementation plan and execution-method selection under Superpowers. Product changes start only after those reviews.
