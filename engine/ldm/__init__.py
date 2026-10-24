"""SHIGERU-LDM (Phase H): continuous latent diffusion / flow matching over a block of K future tokens, conditioned on the
frozen 4B's prefix hidden states.  docs/latent_diffusion_research.md.  Modules: codec (token <-> continuous latent),
denoiser (small conditional transformer predicting the flow velocity), sampling (Euler / Heun)."""
