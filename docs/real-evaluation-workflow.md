# Real evaluation workflow

This workflow separates expensive model execution from inexpensive routing-policy experiments. It is designed for a constrained local machine while preserving traceability.

## 1. Materialize pinned datasets

Dataset specifications under `configs/datasets/` pin the Hugging Face source revision, split, license, homepage, adapter, and prompt version. Materialization writes benchmark JSONL plus a sibling manifest containing the record count and SHA-256 digest. HotpotQA also writes per-example retrieval documents and a deterministic corpus digest.

```bash
python -m budgetroute materialize-dataset --spec configs/datasets/gsm8k.yaml --output data/materialized/gsm8k-test.jsonl --limit 100
python -m budgetroute materialize-dataset --spec configs/datasets/mmlu.yaml --output data/materialized/mmlu-test.jsonl --limit 100 --sampling stratified --seed 42
python -m budgetroute materialize-dataset --spec configs/datasets/hotpotqa.yaml --output data/materialized/hotpotqa-validation.jsonl --limit 100 --corpus-dir data/corpora/hotpotqa-validation
```

`inspect-data --manifest ...` rehashes the JSONL and rejects modified or mismatched data. Downloads and materialized files are intentionally ignored by Git.

MMLU supports deterministic subject-stratified selection. It ranks rows and subjects by SHA-256 of the seed, samples in round-robin subject order, and records the method, seed, and stratification field in the manifest. This avoids the subject concentration of a source-ordered prefix while remaining exactly reproducible. It is still a sample, not the full MMLU suite.

## 2. Collect model baselines once

Model profiles pin model/tokenizer revisions, license identifiers, model-card URLs, precision, seed, and generation settings. `collect-baselines` forces live execution and fills the content-addressed generation cache. The cache key includes every request field that can influence generation plus the backend/model/generation fingerprint; request IDs are excluded so repeated policy runs reuse identical work.

```bash
python -m budgetroute collect-baselines --config configs/benchmarks/real-gsm8k.yaml
```

When retrieval is enabled, collection also runs retrieval-first and cascade paths so contextual small-model generations are available for replay.

## 3. Replay policies without models

```bash
python -m budgetroute replay-benchmark --config configs/benchmarks/real-gsm8k.yaml
```

Read-only replay does not initialize generation model weights. A cache miss is an error naming the missing backend/key rather than silently generating. Replay predictions preserve recorded generation latency, token usage, confidence, and model memory while recording replay overhead separately. Reports identify replay explicitly.

Use `verify-replay` on a small sample after backend, prompt, cache-schema, or Transformers changes:

```bash
python -m budgetroute verify-replay --config configs/benchmarks/real-gsm8k.yaml --sample-size 3
```

The verifier refreshes live entries, replays them without model initialization, and compares answer, route, execution, usage, raw confidence, uncertainty signals, and fake/real identity exactly.

The published RTX 3060 / MMLU-100 profiles use a `read_write` config for replay and a separate `refresh` config for measured live execution:

```bash
python -m budgetroute benchmark --config configs/benchmarks/real-gpu-mmlu-100-live.yaml
python -m budgetroute verify-replay --config configs/benchmarks/real-gpu-mmlu-100.yaml --sample-size 3
```

The corresponding [curated result card](../reports/benchmarks/qwen25-mmlu-100-rtx3060.md) shows how to publish a bounded claim without checking in the generated cache, dataset, or raw model output.

## 4. Train and calibrate the router

```bash
python -m budgetroute train-router --artifacts outputs/RUN --output outputs/router/router.joblib --quality-threshold 0.8 --target-selective-accuracy 0.8
python -m budgetroute calibrate-confidence --artifacts outputs/RUN --output outputs/router/small-confidence.json --policy always_small --quality-threshold 0.8 --target-selective-accuracy 0.8
python -m budgetroute evaluate-router --router outputs/router/router.joblib --artifacts outputs/RUN --quality-threshold 0.8
```

The router uses three deterministic group-aware partitions:

- training fits feature scaling and the base classifier;
- calibration fits scalar temperature and selects the maximum-coverage threshold meeting the configured selective-accuracy target;
- test is untouched until final evaluation.

The artifact records every split ID/group, class counts, source hash, package versions, calibration parameters, selected threshold, raw/calibrated metrics, and any small-sample fallback. Groups must be disjoint whenever enough groups exist. The serving policy loads the stored calibrator and threshold unless configuration explicitly overrides the threshold.

The backend-confidence artifact can be attached with `routing.cascade_calibrator_path`. It transforms raw small-model token likelihood before cascade thresholding. Raw and calibrated values remain separate in predictions.

For a cost-sensitive comparison with paired baselines, use `--label-strategy paired_quality`. After training, materialize exactly the persisted test IDs before the final live benchmark:

```bash
python -m budgetroute materialize-dataset --spec configs/datasets/mmlu.yaml --output data/materialized/mmlu-test-stratified-500.jsonl --limit 500 --sampling stratified --seed 42
python -m budgetroute benchmark --config configs/benchmarks/real-gpu-mmlu-500-live.yaml
python -m budgetroute train-router --artifacts outputs/BASELINE_RUN --output outputs/router/mmlu-500-paired.joblib --quality-threshold 1.0 --target-selective-accuracy 0.8 --label-strategy paired_quality --seed 42
python -m budgetroute calibrate-confidence --artifacts outputs/BASELINE_RUN --output outputs/router/mmlu-500-small-confidence.json --policy always_small --quality-threshold 1.0 --target-selective-accuracy 0.8 --seed 42
python -m budgetroute materialize-router-test-split --router-metadata outputs/router/mmlu-500-paired.joblib.metadata.json --dataset data/materialized/mmlu-test-stratified-500.jsonl --manifest data/materialized/mmlu-test-stratified-500.manifest.json --output data/materialized/mmlu-test-stratified-500-router-test.jsonl
python -m budgetroute benchmark --config configs/benchmarks/real-gpu-mmlu-500-learned-live.yaml
```

The 60/20/20 group split is selected once from the baseline artifact. Model fitting uses train only, probability calibration and the serving threshold use calibration only, and the final live profile contains test IDs only. Treat any later tuning informed by test results as a new development cycle requiring a new untouched test set.

## Interpretation

Length-normalized token likelihood and entropy are genuine model measurements, but neither is automatically a correctness probability. Calibration is task/model/prompt specific. Cache replay supports policy comparisons over fixed model outcomes; it does not reproduce queueing, concurrent contention, thermal drift, or live batching behavior. Confirm final systems conclusions with live runs.

## 5. Train retrieval benefit separately

Paired `always_small` and `retrieval_first` rows from the same compatible run can train the retrieval-benefit artifact:

```bash
python -m budgetroute train-retrieval-router --artifacts outputs/RUN --output outputs/router/retrieval.joblib
```

Its label is the retrieval-minus-baseline quality delta, not small-model correctness. Inspect paired coverage, group partitions, calibration warning, threshold, and untouched test metrics. Validate again after changing the corpus, chunker, embedding revision, model, or prompt template.

## 6. Validate delayed-label adaptation

In a service deployment, predictions and later correctness feedback join in the operational store. `adapt-confidence` requires the configured minimum count, uses a chronological tail holdout, compares the candidate with the active calibrator, and writes an immutable candidate whether or not it is promoted.

```bash
python -m budgetroute adapt-confidence --config configs/serving/distributed.yaml
python -m budgetroute rollback-calibration --config configs/serving/distributed.yaml
```

Review candidate Brier/ECE, promotion gates, source hash, label window, and Page-Hinkley change points. A passed gate justifies a controlled deployment test, not automatic global rollout.

## Batch, concurrency, and runtime matrix

After single-request live/replay agreement is established, create separate resolved configs for each batch size, concurrency, precision, quantization, compiler state, and runtime. Do not change several dimensions inside one unlabeled run. Use enough live requests for stable tail percentiles, retain warm-up/first-compile metadata, and compare only matching model revision, prompt template, dataset, hardware, queue deadlines, and cost-unit definitions.

Cache replay can compare policy quality/cost composition but cannot measure dynamic batching, load-aware routes, model-server contention, queueing, or thermals. Those require live artifacts. The bundled local OpenAI-compatible profile is an integration example, not a compatibility or speed claim for every vLLM, llama.cpp, or Ollama release.

Use `compatibility-matrix` for exact runtime revisions, then `load-test` against authorized deployed endpoints for wall throughput and tail latency. Preserve each artifact and re-test the tool's heuristic replica suggestion rather than treating it as an autoscaling decision.
# Independent development pilot

`prepare-routing-pilot --spec configs/datasets/mmlu-pilot.yaml --exclusions data/exclusions/mmlu-500-v1.json --output data/materialized/routing-pilot` freezes exactly 200 development questions before fitting. Historical exclusions were reconstructed from the SHA-256-verified MMLU-500 file and rehydrated pinned upstream rows; no historical prompt/answer text is published in the exclusion manifest. Canonical source indices and normalized question hashes prevent a new prefix or whitespace change from evading exclusion.

The pilot partitions are 120 fitting, 40 threshold selection and 40 diagnostic questions, never a confirmatory test. The direct-mcq-v2 evaluator accepts only a valid single option letter (optionally parenthesized); incomplete/truncated output scores zero. Version-one datasets retain their existing evaluator. Source materialization and GPU execution are separate steps; preparing data is not experimental evidence.

## Free-Colab experiment log

Open `notebooks/routing_pilot_colab.ipynb` at the exact reviewed feature SHA, select a free GPU, and execute top-to-bottom. It installs that source, freezes installed versions, prepares data and runs real component gates before exposing six development stages: collect, fit, diagnostic, timing, summarize and export. The CLI equivalents are `routing-pilot --config ... --workspace ... --stage ...`. No confirmation stage exists.

Bootstrap explicitly adds the verified checkout's `src` directory to the already-running notebook kernel after editable installation. If changing source commits after importing project modules, restart the kernel before rerunning; do not mix cached Python code with a new checkout identity. Component checks save the full environment/configuration before model initialization, including when initialization fails.

The loopback probe shares the loaded generators through a service copy with the probe's effective API flags. Both the app and its injected service must enable feedback; changing only app configuration leaves endpoint feature flags disabled. The caller's configuration remains unchanged. Preserve failed component/environment/configuration artifacts before retrying at a new source SHA; a probe bug does not justify switching model profiles.

Collection persists v2 cache entries and unique paired rows after every answer; it resumes missing entries only. Diagnostics use only the 40 persisted diagnostic IDs. Live timing uses three seeded policy-order blocks, a warmup per condition, no generation cache, and one initialized two-generator service per block. Partial timing blocks restart entirely. Both generators remain resident even in always-large conditions, so the result is conditional on that deployment, not an optimized large-only service comparison.

The primary semantic gate and numeric ablation use paired non-harm labels and a .98 empirical selective target on the 40 selection questions; this is threshold selection, not a two-point guarantee. The confidence cascade is secondary and fits confidence calibration on fitting IDs only. Matched-random routing has exactly the semantic gate's cheap count; timing assignments use a separately fixed seed.

Download the curated evidence ZIP and the executed notebook before Colab removes the VM. The exporter checks required evidence and hashes; it excludes caches, weights, environment variables, unrelated files and credentials. The authentication probe uses a temporary loopback-only credential and restores the prior environment. Runtime probes must contain actual non-fake generation, semantic, FAISS HNSW, scheduling, API and cleanup evidence; passed labels alone are insufficient.

