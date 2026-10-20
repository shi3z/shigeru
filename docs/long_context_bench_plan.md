# SHIGERU long-context benchmark plan

Principle: every change ships with a measurement; the Phase-1 short-context benchmarks (`bench/phase1_all.sh`) must still run
and must not regress (same prompts, same GPU class, greedy).

## Lengths and what is measured at each
Input lengths: 8K, 32K, 64K, 128K, 256K, 512K, 1,048,576 (generation 256 tokens unless the task needs more).
Per length and engine/backend: TTFT (prefill), decode tokens/s, end-to-end latency, peak VRAM, CPU RAM, GPU↔GPU/CPU bytes
moved per step (from the pager's counters), accuracy of the task. JSON per run in `results/long/`.

## Tasks
1. **Needle-in-a-haystack** (`bench/long/niah.py`): a passkey/fact inserted at depth 0, 10, …, 100 % of a filler document
   (English filler from public-domain text; Japanese filler from Aozora Bunko); accuracy per depth → position-bias curve.
2. **Multi-needle retrieval**: 4–8 needles, all must be returned (exact-match per needle).
3. **RULER-style**: variable tracking (chains of variable assignments), common-word extraction, multi-hop needle QA.
4. **LongBench-style** QA over multiple documents (English: public-domain documents; question answerable only by
   combining two documents).
5. **Japanese long QA**: Aozora Bunko texts with questions whose answers are specific sentences.
6. **Cross-file code reasoning**: synthetic repositories (N files); the answer requires following a call across files.
7. Short-context regression: Phase-1 prompts, GSM8K-50, HumanEval.

## Fairness rules
Same GPU set, same input tokens, same generation length, same decoding (greedy, same threshold), one request at a time
unless the metric is throughput; sparse/retrieval attention variants are always reported next to full attention at the
same length; a configuration "supports N tokens" only when the N-token tasks score within 5 points of the 8K scores.

## Order of work
128K first (Phase 1 cache + Phase 2 paging on Fast-dLLM v2 class models, YaRN as needed) → measure the task suite →
256K → 512K → 1M. Training (Phase 5) only when the measured ceiling of the available checkpoints is reached.
