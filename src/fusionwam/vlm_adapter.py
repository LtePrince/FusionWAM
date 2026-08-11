"""Frozen PaliGemma semantic expert -> projected KV tokens for the action DiT.

Stage 1 contract: the VLM is a frozen feature bank. Its tokens are projected
into the action expert's `context` space (text_dim) and concatenated after the
umt5 text tokens with a source segment embedding. Only `proj`, `norm`,
`segment` (and optional `layerscale`) train.

Weights: google/paligemma-3b-pt-224 by default (scripts/download_weights.sh);
a pi0.5-finetuned PaliGemma can be swapped in after JAX->HF conversion
(scripts/convert_pi05_vlm.md) — run the frozen-probe experiment first to see
whether the base model's features already carry OOD object position.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class FrozenPaliGemmaEncoder(nn.Module):
    def __init__(
        self,
        model_path: str,
        out_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        layerscale_init: Optional[float] = 1e-3,
    ):
        super().__init__()
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype
        )
        self.vlm.requires_grad_(False)
        self.vlm.eval()
        vlm_dim = self.vlm.config.text_config.hidden_size

        self.proj = nn.Linear(vlm_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.segment = nn.Parameter(torch.zeros(1, 1, out_dim))
        # Per-source LayerScale: the two context sources (umt5, VLM) have
        # different activation norms; start the VLM contribution small so it
        # cannot destabilize the pretrained cross-attention early in training.
        self.layerscale = (
            nn.Parameter(torch.full((out_dim,), layerscale_init))
            if layerscale_init is not None
            else None
        )

    @torch.no_grad()
    def _encode(self, images: torch.Tensor, prompts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """images: [B,3,H,W] in [0,1] (RGB); prompts: list[str] of length B.
        Returns (hidden [B,L,vlm_dim], mask [B,L])."""
        inputs = self.processor(
            text=prompts,
            images=[img for img in images],
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)
        out = self.vlm(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = inputs["attention_mask"].bool()
        return hidden, mask

    def forward(self, images: torch.Tensor, prompts: list) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, mask = self._encode(images, prompts)
        tokens = self.norm(self.proj(hidden.to(self.proj.weight.dtype)))
        if self.layerscale is not None:
            tokens = tokens * self.layerscale
        tokens = tokens + self.segment
        return tokens, mask
