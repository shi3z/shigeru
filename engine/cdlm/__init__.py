"""engine.cdlm — causal diffusion LM (WeDLM-style dual-stream training) groundwork for SHIGERU Phase C.

Modules: data (example/batch builder), mask (2-D attention mask, dense + FlexAttention mask_mod), loss (MLM 1/gamma + AR),
model_glue (run a HF Qwen3ForCausalLM on the dual stream, mask-token embedding row), decoder (streaming-window
inference over a DynamicCache). See README.md.
"""
from .data import IGNORE, build_example, collate, uniform_gamma
from .mask import allowed, batch_mask, dense_mask, make_mask_mod, to_additive
from .loss import ar_loss, mlm_loss, total_loss
from .model_glue import add_mask_token, forward_batch, forward_rows, model_attention_mask
from .decoder import generate, select_positions, window_attention_mask
