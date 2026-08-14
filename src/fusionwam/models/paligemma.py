"""Frozen PaliGemma semantic expert: full (image + instruction) forward,
per-layer hidden states out.

The fusion design (openpi-style, generalized to heterogeneous towers): the
VLM runs its OWN transformer over [image tokens + language instruction];
every Gemma layer's hidden states are exposed so the action expert can attend
a depth-mapped KV prefix at each of its layers. Probe evidence
(2026-08-12): object position lives dose-flat in the early/vision layers and
decays through the Gemma pass, while instruction binding lives late —
per-layer access serves both without choosing a single tap.

Nothing here trains; the K/V adapters live on the fusion side
(models/fusion.py), mirroring how openpi's experts each project their own
width into a shared attention-head space.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class FrozenPaliGemmaEncoder(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.vlm = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype
        )
        self.vlm.requires_grad_(False)
        self.vlm.eval()
        self.hidden_dim = self.vlm.config.text_config.hidden_size
        self.num_layers = self.vlm.config.text_config.num_hidden_layers

    @torch.no_grad()
    def layer_hidden(
        self, images: torch.Tensor, prompts: list
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """images [B,3,H,W] in [0,1]; prompts list[str].

        Returns (hidden_states, mask): hidden_states is a list over depth —
        index 0 = embeddings, 1..L = after each Gemma layer — each
        [B, S, hidden_dim] over the full multimodal sequence (256 image
        tokens first, then instruction tokens); mask [B, S] marks real
        tokens (padding False).
        """
        inputs = self.processor(
            text=prompts,
            images=[img for img in images],
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)
        out = self.vlm(**inputs, output_hidden_states=True)
        mask = inputs["attention_mask"].bool()
        return list(out.hidden_states), mask
