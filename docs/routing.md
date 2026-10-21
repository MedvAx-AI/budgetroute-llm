# Routing policies

Every decision contains route, policy, confidence, optional difficulty, reason, feature snapshot, and thresholds.

## Baselines

`always_small` estimates the inexpensive quality floor. `always_large` estimates the expensive quality ceiling. `random` samples from normalized configured probabilities using a hash of seed, request ID, and prompt, making repeated ordered inputs reproducible.

## Heuristic

The heuristic difficulty score combines bounded contributions from request length, sentences, code fences, mathematical symbols, long-output indicators, code/numeric category, URLs, and digit ratio. Rules are readable and configurable. Retrieval-required requests use the small model with context when a hit survives the score threshold. Ambiguous or explicitly unanswerable requests can abstain.

Heuristics are portable but dataset-sensitive. Feature thresholds must be validated on representative data, not tuned on test labels.

## Learned

New schema-v4 router and confidence-calibration artifacts represent an infeasible threshold explicitly with `threshold_selection.status: infeasible` and a JSON-null threshold. Learned routing then selects only large; the cascade always escalates. A numerical serving override cannot reopen an infeasible selection or a failed/insufficient certificate. A feasible empirical threshold is **not** a statistical certificate: new pilot artifacts record `certification_status: not_requested`. Legacy artifacts remain readable using their historical numeric thresholds, explicitly uncertified.

The learned policy supports two explicit training targets using the same numeric feature snapshot at training and serving. `small_success` predicts whether the small backend meets a quality threshold. `paired_quality` joins successful `always_small` and `always_large` outcomes for each example and predicts whether small-model quality is at least large-model quality. The paired target directly models when selecting the inexpensive backend avoids an observed quality loss; it does not assume that the larger model is always correct.

The artifact stores the preprocessing/model pipeline, ordered features, label strategy and definition, source hash, three group-aware partitions, scalar calibrator, calibration-selected serving threshold, seed, raw/calibrated test metrics, held-out routing outcome metrics when paired outcomes exist, warnings, and package versions. Outcome metrics include selected quality, route share, deltas against both fixed baselines, and regret against a per-example oracle.

Logistic regression is used when both training classes exist; a dummy classifier provides explicit, non-crashing behavior for one-class demonstrations. Related records share `group_id` and cannot cross train/calibration/test boundaries when enough groups exist. Tiny in-sample fallbacks are labeled and are not generalization evidence.

`materialize-router-test-split` exports the persisted test IDs in artifact order and writes a derived integrity manifest linked to the parent dataset hash. This lets the final live policy benchmark use only untouched examples. The test split must not be used to select features, thresholds, or hyperparameters after its results are inspected.

## Retrieval first

Retrieval runs before final routing. A hit above the configured similarity threshold selects small-with-retrieval; otherwise backend choice falls back to difficulty. This can improve context-dependent quality but adds retrieval latency to all requests and can inject irrelevant text.

## Learned retrieval benefit

`learned_retrieval` predicts whether retrieved context will improve answer quality by a configured minimum delta. Training pairs `always_small` and `retrieval_first` rows for the same request, keeps related `group_id` values in one partition, calibrates on a separate split, and reports untouched test metrics when enough groups exist. The artifact records the label definition, partitions, feature order, source hash, threshold, calibration, warnings, and package versions.

At serving time retrieval runs first, features include similarity/margin evidence, and a positive prediction selects small-with-retrieval. Otherwise routing falls back to ordinary small/large difficulty behavior. This separates retrieval usefulness from the learned small-model-success target and prevents a high similarity score from being treated as proof of benefit.

## Cascade

The small backend runs first. Transformers confidence is the geometric mean selected-token probability, accompanied by log-probability and entropy signals. If a configured calibration artifact exists, its scalar temperature transforms this raw value before thresholding. Below the selected threshold, the large backend runs and the trace records the initial answer (internally when enabled), escalation reason, extra input/output tokens, and extra latency. High false escalation wastes compute; high false non-escalation harms quality.

## Load aware

`load_aware` combines deterministic difficulty with trusted per-backend inflight utilization and exponentially weighted observed latency. Easy requests prefer the lower-pressure/latency backend. Hard requests prefer the large backend, but may use a guarded cascade when the large backend crosses the overload threshold. When both backends are overloaded and a sufficiently hard request crosses the review threshold, the policy emits explicit human review.

Load values come from backend decorators, not client metadata. Decisions include both snapshots and active thresholds. Because live load is time-dependent, load-aware routing is not expected to choose identical routes under different concurrency schedules; benchmark artifacts preserve every decision.

## Budget aware

`budget_aware` estimates each backend's input/output cost units from configured per-1,000-token rates, combines estimated quality, load-adjusted latency, and cost with explicit weights, then selects the highest-utility eligible backend. `max_estimated_cost_units` can make a route ineligible. If no backend fits, the policy abstains or requests human review according to configuration.

Cost units are deployment supplied. They can represent money, energy, reserved GPU time, or normalized compute, but must use one consistent unit within a run. They are estimates, not observed invoices. The transparent quality estimator must be validated or replaced with deployment evidence before operational use.

## Abstention

Abstention is a first-class route, not an exception. It trades coverage for selective accuracy when required context is absent or the request is unanswerable. Production policies need domain-specific calibration and user experience design.

## Human review

Human review is distinct from abstention and records a reason in the execution trace. When review is enabled, the API creates a privacy-minimal durable case in the operational store. Scoped reviewers can tenant-list, claim, and resolve cases using optimistic versions; a resolved correctness outcome can become a delayed label and every mutation is audited. Prompts and answers are intentionally absent, and the project does not notify a person or provide ticketing/SLA management—deployments needing those functions must integrate an external workflow.
