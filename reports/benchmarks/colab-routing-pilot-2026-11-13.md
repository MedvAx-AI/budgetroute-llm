# Free-Colab routing development pilot — 2026-11-13

**Completed pilot, negative feasibility result for this configuration.** The semantic gate preserved always-large accuracy by routing none of the 40 diagnostic questions to the small model. Its mean live latency was 406.7 ms versus 359.6 ms for always-large: **13.1% slower**, not the target of at least 20% faster. Do not advance this frozen configuration to confirmation.

## Results

Across all 200 development questions, always-small answered **85/200 (42.5%)** correctly and always-large **148/200 (74.0%)**. These totals include fitting and selection questions, so they are baseline descriptions rather than held-out estimates.

| Policy | Correct / 40 diagnostic | Finished on small / 40 | Mean live latency | Ratio to always-large |
| --- | ---: | ---: | ---: | ---: |
| Always-large | 32 | 0 | 359.6 ms | 1.000× |
| Always-small | 21 | 40 | 347.3 ms | 0.966× |
| Numeric gate | 32 | 0 | 373.7 ms | 1.039× |
| Semantic gate | 32 | 0 | 406.7 ms | 1.131× |
| Matched random | 32 | 0 | 359.1 ms | 0.998× |
| Confidence cascade | 32 | 0 | 710.4 ms | 1.975× |

Always-small gained 3 answers and lost 14 versus always-large on the same questions (−27.5 percentage points; descriptive grouped 95% interval −45 to −10 points). The semantic gate's descriptive live latency-ratio interval was 1.094–1.172. Its zero accuracy delta is not evidence of safe cheap substitution: there were no cheap substitutions. The matched-random policy also made no cheap selections by design. The cascade tried small before escalating; its zero final-small count does not mean it avoided the small-model call.

Always-large measured 334.7 ms median and 484.0 ms p95; the semantic gate measured 378.2 ms median and 561.3 ms p95. The cascade escalated all 120 timed requests. Peak reported CUDA allocation across the live comparisons was 10,588.5 MiB; peak reported process RSS was 2,732.6 MiB. Those are observed counters, not capacity guarantees.

## What was verified

- Free Tesla T4 (15,360 MiB), CUDA 13.0, driver 580.82.07, PyTorch 2.14.1+cu130, Python 3.13.16. Qwen3.5-0.8B and Qwen3.5-4B ran co-resident, FP16, greedy non-thinking, at most 64 output tokens, one request at a time. No paid compute, fallback model, or quantization was used.
- Clean [source `842d92f`](https://github.com/MedvAx-AI/budgetroute-llm/tree/842d92fe007c29c4c5ac5a0445efdb9910bb18ad). The exact-source [CI run](https://github.com/MedvAx-AI/budgetroute-llm/actions/runs/42831640638) passed 203 tests each on Python 3.11 and 3.12, plus two PostgreSQL 17 contract tests. CI is software evidence, not GPU evidence.
- Real generation, confidence, batching and order; MiniLM; FAISS HNSW save/reload/search; scheduling; authenticated loopback API, feedback and metrics; cleanup. Retrieval was probed separately and was disabled in the routing comparisons.
- 200 frozen questions independent of the prior 500, split 120 fitting / 40 selection / 40 diagnostic. The accepted archive contained 400 baseline predictions, 240 diagnostic predictions, 720 cache-disabled measurements (three repeats of 40 questions for six policies), and 80/80 matching diagnostic baseline replay signatures. No request errors in the accepted evidence; strict parser failures and truncation were still counted as incorrect.
- Verified ZIP CRC, allowlisted membership, per-file hashes, real-artifact acceptance and executed notebook outputs. Accuracy, cheap coverage and mean latency ratios were also recomputed from the prediction rows. Raw artifacts remain outside Git; the [machine-readable card](colab-routing-pilot-2026-11-13.json) records their checksums.

## Decision and limits

This establishes a working remote experiment, not a general verdict on model routing. The small model was materially less accurate and delivered only a 3.4% mean latency reduction in this setup. Even unconditional small routing missed the 20% speed target on the point estimate, while the semantic gate added overhead without saving a generation call. Keep always-large as the default for this workload. If revisiting the research, first screen candidate pairs on separate development data for both a real latency gap and a useful set of safe cheap substitutions, before investing in a more complex gate. Do not tune on these 40 questions and call the result confirmation.

The notebook's rough planning output (491 independent questions, about 1.05 hours of timed requests at the observed mix) comes from a normal approximation with a discordance floor. It is not a power guarantee, excludes setup and interruptions, and does not justify confirmation when the measured effect misses the target. The 120 timed requests per policy are only 40 independent questions; grouped intervals here are descriptive. No monetary cost, production throughput, provider compatibility, or non-inferiority claim is made.
