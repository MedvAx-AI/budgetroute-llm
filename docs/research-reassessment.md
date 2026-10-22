# Research reassessment — 2026-10-21

Status updated 2026-10-22: relevance/methods audit, specification and [Native implementation plan](superpowers/plans/2026-10-21-colab-routing-study.md) approved; Tasks 1–9 implemented. The defects below describe the audited old source and are now covered by measurement repairs. Remote CPU/database CI and the repaired T4 component gate passed at `5ab58a8`; collection then exhausted host RAM during model reload, before any paired outcomes. An explicit direct-to-CUDA loading repair requires a new identity and fresh remote validation. Failed attempts are retained. No completed pilot, confirmatory protocol, or new positive routing result exists yet. Budget: free Google Colab only; no paid compute or public service deployment.

## Bottom line

LLM routing remains relevant; another generic router implementation is not a scientific contribution by itself. Keep BudgetRoute as an experimental harness and narrow the next study to the conditions under which quality-constrained routing is actually useful. Prefer a reproducible replication-and-extension study over an unsupported novelty claim.

The old small model being weaker on average is expected. Routing succeeds only if it can identify a useful subset for the cheaper model and its savings exceed routing/escalation overhead. The published MMLU-500 result does not demonstrate that: learned routing achieved 51% versus always-large's 53% on 100 test questions, with only eight cheap routes and slower observed median latency. This is evidence about that setup, not a universal impossibility result. The difference is descriptive; no paired difference interval or non-inferiority test was computed.

The 500 old questions and inspected final holdout are now exploratory history. Preserve their reports unchanged and exclude them from a new confirmatory test. A fresh routing holdout also cannot guarantee absence of foundation-model pretraining contamination.

## Relevance and prior art

Primary sources checked on 2026-10-21:

| Work | Implication for this project |
| --- | --- |
| [FrugalGPT](https://arxiv.org/abs/2305.05176), 2023 | Quality/cost cascades are established; small-first escalation is not a new method. |
| [RouterBench](https://arxiv.org/abs/2403.12031), 2024 | Public paired outcomes can support larger method checks without regenerating every answer. Cached outcomes cannot validate current models or live latency. |
| [RouteLLM](https://arxiv.org/abs/2406.18665), 2024, revised 2025 | Learned strong/weak routing and transfer between model pairs are established. |
| [CP-Router](https://arxiv.org/abs/2505.19970), 2025 | Uncertainty/conformal routing is established. Prediction-set coverage is not equivalent to routed-answer correctness. |
| [LLMRouterBench](https://arxiv.org/abs/2601.07206), 2026 | Large unified evaluation finds several elaborate routers do not reliably beat simple baselines. Strong baselines and failure-regime analysis matter more than adding complexity. |
| [Conformal LLM Routing with Distribution-Free Safety Guarantees](https://aclanthology.org/2026.acl-srw.70/), ACL SRW, July 2026 | Embedding logistic routing with statistical threshold calibration is direct prior art. Replication with calibration-size, split stability, shift, and actual runtime costs is a candidate extension, not proven novelty. |
| [Learn then Test](https://arxiv.org/abs/2110.01052) | Adaptive threshold selection must account for multiple testing or use independent selection/certification data. Scanning pointwise confidence bounds does not automatically certify the selected threshold. |

The ACL routing paper's relative-harm label counts a violation only when the cheap model is wrong and the strong model is correct. Both-wrong cases count as non-harmful substitutions, **not correct answers**. Report absolute accuracy, harmful substitutions, beneficial substitutions, and relative degradation separately. Do not describe relative risk control as general model safety.

## Model refresh: implemented profiles, validation in progress

- Historical control: retain pinned Qwen2.5-0.5B/1.5B profiles and their negative result.
- Recommended modern pilot: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The [official repository](https://github.com/QwenLM/Qwen3.8) records the small-model release on March 2, 2026. These are Apache-2.0 models, but their hybrid/multimodal architecture requires a real loader/runtime probe; changing the model name in the current causal-LM loader is not a verified migration.
- Lower integration-risk fallback: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) and [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507). Select a fallback only on development feasibility criteria, never by looking at the final test outcome.

The implemented primary/fallback profiles pin exact model revisions, built-in loader class, non-thinking greedy FP16 decoding, 2,048 input and 64 output tokens. Thinking defaults differ between checkpoints: never infer the mode from the family name. The October 22 component attempt loaded both primary generators on a 15-GiB T4 and exercised real single/batch generation, but the full component gate and measured pilot remain unfinished. This is not a latency or quality comparison. Quantization remains a separate condition, not an invisible memory workaround.

## Methodological repairs needed before a confirmatory study

These historical findings were checked against source at `5a5e359cfac0740af8652fce7d3092e62a1f4ca2`. Tasks 1–9 now implement and test the corresponding repairs: fail-closed artifact states, cache v2 identity, separate router probabilities, per-call nullable costs, observation-weighted clustered uncertainty, and failure-aware paired/blocked evaluation. Software tests do not establish the routing hypothesis.

1. **Infeasible thresholds fail open.** `routing/training.py` uses `0.5` when `choose_threshold` returns no feasible threshold, for both learned routing and confidence calibration. Replace this with an explicit infeasible/always-large or always-escalate outcome before claiming a quality guardrail.
2. **Remote cache identity is incomplete.** `experiments/cache.py:backend_fingerprint` omits endpoint, requested logprobs, and immutable remote deployment/runtime identity. Distinct configurations can collide. Repair identity and preserve provenance before using remote-server replay. This does not demonstrate contamination of the historical local Transformers study.
3. **Router and answer calibration are mixed.** The harness prefers final-answer confidence; the runner complements that confidence for large routes and evaluates a small-success target even for paired-quality routing. Store router probability/target independently from final-answer confidence. End-to-end historical accuracies are not invalidated by this metric issue.
4. **Missing cost is rendered as zero.** Most policies do not provide the pre-decision estimate used by cost aggregation. Account for every backend call with actual usage and explicitly declared rates, or retain unknown cost. For free Colab, measured GPU time is the primary resource endpoint; tokens, parameter ratios, and route share are not observed monetary savings or energy measurements.
5. **Grouped uncertainty needs a consistent estimand.** Current point means are observation-weighted, while bootstrap samples average group means equally. Fix the discrepancy for unequal groups and keep repeated observations grouped under their original IDs. Existing MMLU adapters have independent one-question groups, so this finding does not itself invalidate that report.
6. **Study design needs stronger comparisons.** Add paired difference uncertainty, failure-aware quality, matched-budget random routing, and randomized/blocked repeated live timing. Fixed-order single passes and overlapping per-policy intervals do not establish non-inferiority or speedup.

Reuse the typed interfaces, manifests, group splits, raw artifacts, deterministic evaluators, replay architecture after repair, and existing logistic baseline. Freeze expansion of deployment infrastructure, online adaptation, retrieval features, and routing abstractions during this study. Keep those capabilities supported and distinguish their tests from evidence for the routing hypothesis.

## Recommended scientific direction — accepted for specification

Three possible routes:

1. **Recommended: replication plus break-even analysis.** Test routing usefulness under limited calibration data and actual single-GPU overhead. Feasible to stage on free Colab and can yield a useful positive, negative, or inconclusive result.
2. **Model-refresh demonstration.** Easier, but a new model pair alone explains little and establishes neither novelty nor generality.
3. **Broad multi-model benchmark.** More external validity, but not a realistic initial commitment under an unguaranteed free GPU budget.

Candidate primary hypothesis: **a frozen prompt-only semantic gate reduces mean end-to-end request latency by at least 20% relative to always-large while reducing absolute accuracy by no more than two percentage points on a preregistered workload.** These are proposed practical effect targets, not discovered results or already-approved margins. Require both endpoints; reduced cheap-route risk alone is insufficient.

Candidate secondary question: how do calibration budgets of 100, 500, 1,000, and 2,000 examples affect certified cheap coverage and actual net latency? A post-generation confidence cascade is a controlled comparison, not assumed to be faster. A separate held-out-domain stress test examines sensitivity to shift; do not claim an IID guarantee survives arbitrary shift.

Suggested staged design:

- Run an engineering/pair-feasibility pilot on 100–200 development-only paired questions. Record discordance, baseline accuracy, parsing/truncation errors, memory, mean live cost, and realistic session throughput. An insufficient strong baseline is not made useful merely by preserving its accuracy.
- Choose the pair, one primary workload, prompting/scoring, margin, seeds, resource cap, and sample size from development evidence. Use an unseen, version-pinned primary pool excluding old study IDs; specify duplicate/group exclusions. Consider MMLU knowledge for continuity and a separate math/subject-shift validation, without silently extending eight-token scoring to reasoning tasks.
- Estimate the required final-test size from paired discordance and the desired non-inferiority power/precision before committing to the full run. If the free budget cannot support it, narrow the claim or report a pilot/inconclusive result; do not relax the margin after test inspection.
- Separate fitting, threshold selection, independent certification, and final test. The simplest defensible initial certificate is an independently selected fixed threshold assessed on untouched certification data. A valid multiple-testing procedure is an alternative if needed; do not implement a naive adaptive pointwise-bound sweep.
- Compare always-small, always-large, route-share-matched random, existing lexical logistic routing, and exactly one prompt-only semantic representation. Report the oracle as an unattainable diagnostic, not a deployable policy. Include feature/embedding overhead. Freeze one primary policy before opening the test; exploratory policy sweeps remain labeled exploratory.
- Use paired accuracy differences and an appropriate one-sided non-inferiority bound; count generation failures and unresolved/truncated answers rather than dropping them. Prespecify timing repetitions, blocking/randomization, and latency-ratio uncertainty. A positive claim needs the accuracy lower bound above the negative margin and a latency-ratio upper bound below the proposed target, with the joint error policy stated in the protocol.
- Publish every preregistered condition, infeasible outcomes, split IDs/hashes, source/environment/model identity, raw predictions where licensing permits, and execution evidence. Changes after test inspection require a new untouched confirmation set.

Sample-size intuition: for a **fixed** threshold, eight accepted calibration examples and zero harmful substitutions still yield a one-sided 95% binomial upper risk bound of about 31.2%. At least 59 accepted, independent, zero-failure examples are needed to bring that bound below 5%. Adaptive selection, grouping, or nonzero failures require additional care. The old 100-example calibration split is therefore not evidence of tight risk certification.

## Off-local validation: actual evidence and remaining gaps

Fresh existing-code validation was triggered on a dedicated branch pointing to unchanged source SHA `5a5e359cfac0740af8652fce7d3092e62a1f4ca2`:

[GitHub Actions CI, October 21, 2026](https://github.com/MedvAx-AI/budgetroute-llm/actions/runs/39600795502): Python 3.11 and 3.12 each passed 90 tests with one PostgreSQL-module skip and 76.09% coverage; the separate real PostgreSQL 17 job passed both contract tests. Format, lint, strict typing, and builds also passed. The skipped module is supplied with a DSN in the dedicated database job. This is remote software-contract evidence, not model-quality or deployment-capacity evidence. The old August run could not be rerun because GitHub rejected runs older than a month.

| Remote lane | Status and acceptance evidence |
| --- | --- |
| CPU/API/offline routing/storage contracts and real PostgreSQL | Repaired source `5ab58a8` passed Python 3.11/3.12 (197 tests each, 78.38% branch coverage) and PostgreSQL 17 (2 tests); [exact run](https://github.com/MedvAx-AI/budgetroute-llm/actions/runs/39811563385). Each quality job skips the database module supplied separately in the real database job. |
| Colab real GPU backend | Full component gate passed at `5ab58a8` after fixing the earlier API probe. Subsequent CPU-staged model reload exhausted host RAM; the explicit direct-device loading retry must pass the gate again at its new source/config identity. |
| Colab research pipeline | Data materialized remotely: exactly 120 fitting / 40 selection / 40 diagnostic questions, excluding the old 500. Baseline collection, fitting, diagnostic/replay, live timing and verified final export remain pending. Pilot has no certification or final confirmation split. |
| Real semantic retrieval | Pinned CPU MiniLM (2×384 vectors) and actual FAISS HNSW passed within the repaired component gate. This is not a corpus-scale retrieval study. |
| Real HTTP-compatible backend and loopback API | Repaired authenticated IPv4 loopback generation/feedback/metrics and graceful shutdown passed on Colab at `5ab58a8`. No separate provider/runtime compatibility or public capacity claim. |

CLI success is not enough: compatibility probes catch exceptions, and benchmarks can finish with errors. Acceptance must assert artifact success, zero errors, non-fake backend identity, correct batch/order fields, source/config/data hashes, and observed environment. The committed [thin notebook](../notebooks/routing_pilot_colab.ipynb) delegates all stages to versioned code. Store executed notebook output and allowlisted artifacts outside the ephemeral runtime; the unexecuted source notebook is not evidence.

The [official Colab FAQ](https://research.google.com/colaboratory/faq.html) makes GPU availability/type and session limits dynamic; free sessions may end early. Use interactive, notebook-driven execution and resumable checkpointed runs, without SSH, remote desktops, public service tunnels, or quota-bypass tricks. The [runtime-version FAQ](https://research.google.com/colaboratory/runtime-version-faq.html) documents selectable historical runtimes; record the version actually observed rather than assuming a particular default.

The requested in-app-browser-only approach allocated a free Tesla T4 on October 22 without desktop automation. The observed environment was Python 3.13.15, torch 2.11.0+cu130, CUDA 13.0 and driver 580.82.07. Data and partial component execution are real; the full experiment is not complete. Failed-probe configuration/environment/component files were retained outside Colab before retrying. No paid upgrade, public tunnel, Drive mount or credential provisioning was used.

## Next decision

The written pilot specification and Native implementation plan are approved. Follow the plan's live progress and evidence record. Browser access does not establish GPU execution, and design approval is not a scientific result.
