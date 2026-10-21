# Benchmarking protocol

## Lifecycle and timing

Model download and loading occur before measured requests. Service initialization time is recorded per policy. Configured warm-up requests run after initialization, their aggregate duration is recorded separately, and they are excluded from predictions and steady-state metrics. Each timed region uses `time.perf_counter_ns`, a monotonic high-resolution clock.

The trace separates scheduler queue, routing, retrieval, generation, escalation, and total request time and records actual batch size. Benchmark `batch_size` groups ordered requests into backend waves; `concurrency` runs that many batch workers while backend concurrency slots protect local runtimes. Transformers generation synchronizes CUDA immediately before and after the measured region. CUDA peak allocation counters are reset before generation and read afterward when available. `psutil` records process RSS.

Compilation warm-up belongs outside measured steady-state requests; metadata records whether compile or quantization was requested and actually enabled plus the first compiled execution duration. Padded batching groups requests by effective output-token limit to preserve request semantics. Never claim an optimization speed-up without directly comparable artifacts across batch-size/concurrency matrices.

## Sample size and percentiles

p50 and p95 are reported for non-empty samples, accompanied by a warning below 20 values. p99 is omitted below 100 values. Tail values from tiny datasets are illustrative and should not be compared as stable estimates.

Failed requests are recorded in `errors.jsonl` and excluded from successful latency summaries, while failure counts remain visible. Cold-start and warm-start values must be labeled separately; steady-state request values preserve initialization/compilation fields separately.

Grouped percentile bootstrap intervals use the configured seed and resample independent groups, not related rows. `minimum_samples_for_claims` adds a visible warning when evidence is too small for a comparative claim. Neither mechanism fixes selection bias or dataset mismatch.

## Generation collection and replay

`collect-baselines` writes backend-neutral generations into a content-addressed cache. Keys include the cache schema, backend/model/tokenizer revisions, precision, quantization, generation settings, prompt, request controls, and generation-relevant metadata. Request IDs are excluded. `read_only` replay never initializes generation model weights and fails visibly on a missing entry.

Replay preserves recorded generation/escalation latency and token/memory observations. It measures routing/retrieval and replay overhead in the current process, but it cannot reproduce concurrent model contention, batching, thermals, or queueing. `run.json` and the report identify replay runs. `verify-replay` refreshes a live sample and requires exact agreement on semantic response fields before replay artifacts are trusted.

## Comparison discipline

Compare policies only when model identifiers, revisions, runtime endpoint/version, prompt templates, generation parameters, data, warm-up, measured count, batch size, concurrency, queue/deadline settings, precision, quantization, compiler state, cost-unit definitions, hardware, drivers, and software environment are compatible. Load-aware policies additionally require comparable arrival patterns. Inspect resolved configuration and environment metadata before interpreting differences.

Fake artificial delay verifies instrumentation only. It is not a proxy for real hardware or model behavior.

`compatibility-matrix` validates initialization plus single/batch behavior for exact configurations and writes environment-backed evidence without making a speed comparison. `load-test` sends a fixed number of real HTTP requests to one or more authorized endpoints with closed-loop bounded concurrency. It calculates throughput from the measured wall interval and request latency only while holding a concurrency slot, along with errors, overload, p50/p95/p99, and a bounded claim statement. It is not an open-loop arrival/soak model. Its replica multiplier is suppressed when there are non-overload failures or no successful requests and is otherwise only a heuristic to test in a subsequent steady-state run.

## Published evidence

The first curated real result is [Qwen2.5 routing on stratified MMLU-100](../reports/benchmarks/qwen25-mmlu-100-rtx3060.md). Its sibling JSON file preserves exact metrics. Raw predictions, model generations, downloaded data, and caches remain ignored; the card records the clean source commit, hashes, revisions, environment, replay audit, uncertainty, and claim boundary needed to interpret the result.

The follow-up [paired-quality held-out MMLU study](../reports/benchmarks/qwen25-mmlu-500-learned-rtx3060.md) expands baseline collection to 500 paired outcomes, freezes disjoint 300/100/100 train/calibration/test partitions, and reports the learned router's negative result rather than claiming an unsupported optimization. Its sibling JSON preserves exact split, calibration, outcome, live-system, and replay-audit evidence.
# Comparison populations

Quality denominators include every scheduled answer, including errors; latency distributions are successful-only diagnostics. Repeated trials are clustered by original question, not counted as independent new questions. Paired accuracy differences and confidence intervals are descriptive. Speedup ratios are unavailable without verified compatible live trial contexts, and never calculated from failures assigned zero latency.

