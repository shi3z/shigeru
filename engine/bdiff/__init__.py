"""SHIGERU B-series: true discrete (absorbing-mask) diffusion over a block of K future tokens with block-bidirectional
attention and optional time conditioning, on top of the frozen/LoRA-adapted Qwen3-4B.  Independent of engine/cdlm
(A-series).  docs/discrete_diffusion_b_series.md.  Modules: masks (attention), corrupt (forward process q(x_t|x_0,t) and
the other corruption types), model (HF wrapper: 4D mask, time embedding, LoRA), sampler (reverse process)."""
