"""FusionWAM: semantic (PaliGemma) + dynamics (Wan2.2 video DiT) + action DiT.

Stage 1 (this file's default path)
----------------------------------
Minimal-surgery fusion, faithful to both parents' mechanisms:
- video<->action coupling stays EXACTLY FastWAMJoint's MoT joint attention;
- the VLM enters as extra `context` tokens for the action expert's
  cross-attention: context = [umt5_text ⊕ proj(vlm_tokens)], each with a
  source segment embedding, VLM branch behind a LayerScale.
Nothing in the two frozen backbones moves; new params = projection + segment
embeddings (+ the action expert itself, which trains).

Source-level dropout (campaign lesson "挂上≠用上" / Gate 1, B2): with
p_drop_video the action->video block of the MoT mask is zeroed for the batch;
with p_drop_vlm the VLM context slice is masked. Each pathway must learn to
carry the task alone.

Stage 2 (big-machine): `build_tri_mot_attention_mask` extends the MoT to
three segments so the video expert can also read the VLM (replacing umt5).
Enable via config `fusion.tri_mot=true` — requires unfreezing decisions
documented in configs/stage1.yaml comments.
"""

from typing import Any, Dict, Optional

import torch

try:  # FastWAM is installed on the training machine (uv pip install -e ../FastWAM);
    # the mask builder below stays importable without it (smoke test, tooling).
    from fastwam.models.wan22.fastwam_joint import FastWAMJoint
except ModuleNotFoundError:  # pragma: no cover
    FastWAMJoint = object

from .vlm_adapter import FrozenPaliGemmaEncoder


def build_tri_mot_attention_mask(
    vlm_len: int,
    video_len: int,
    action_len: int,
    video_to_video_mask: torch.Tensor,
    device: torch.device,
    drop_action_to_video: bool = False,
    drop_action_to_vlm: bool = False,
) -> torch.Tensor:
    """[VLM ⊕ video ⊕ action] joint-attention mask (True = attend).

    VLM    -> VLM              (frozen prefix, self only)
    video  -> video (own mask) + VLM        (stage-2: semantics condition dynamics)
    action -> action + video + VLM          (dual-KV consumption, per dropout flags)
    """
    total = vlm_len + video_len + action_len
    m = torch.zeros((total, total), dtype=torch.bool, device=device)
    v0, a0 = vlm_len, vlm_len + video_len
    m[:v0, :v0] = True
    m[v0:a0, v0:a0] = video_to_video_mask
    m[v0:a0, :v0] = True
    m[a0:, a0:] = True
    if not drop_action_to_video:
        m[a0:, v0:a0] = True
    if not drop_action_to_vlm:
        m[a0:, :v0] = True
    return m


class FusionWAM(FastWAMJoint):
    """Stage-1 fusion: FastWAMJoint + VLM context injection into the action expert."""

    def attach_vlm(
        self,
        vlm_path: str,
        p_drop_vlm: float = 0.3,
        p_drop_video: float = 0.3,
        layerscale_init: float = 1e-3,
    ):
        text_dim = self.action_expert.text_dim if hasattr(self.action_expert, "text_dim") else None
        if text_dim is None:
            # action expert embeds `context` via text_embedding: Linear(text_dim, hidden)
            text_dim = self.action_expert.text_embedding[0].in_features
        self.vlm_encoder = FrozenPaliGemmaEncoder(
            vlm_path, out_dim=text_dim, layerscale_init=layerscale_init
        )
        self.p_drop_vlm = float(p_drop_vlm)
        self.p_drop_video = float(p_drop_video)
        return self

    def fused_context(
        self,
        text_context: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        images: torch.Tensor,
        prompts: list,
    ) -> Dict[str, Any]:
        """Assemble [umt5 ⊕ VLM] context for the action expert.

        Returns dict(context, context_mask, dropped_vlm) — during training the
        VLM slice is dropped with p_drop_vlm (mask zeroed, tokens kept so
        shapes/compile stay static).
        """
        B = text_context.shape[0]
        if text_mask is None:
            text_mask = torch.ones(
                text_context.shape[:2], dtype=torch.bool, device=text_context.device
            )
        vlm_tokens, vlm_mask = self.vlm_encoder(images, prompts)
        vlm_tokens = vlm_tokens.to(text_context.dtype)
        dropped = False
        if self.training and torch.rand(()) < self.p_drop_vlm:
            vlm_mask = torch.zeros_like(vlm_mask)
            dropped = True
        context = torch.cat([text_context, vlm_tokens], dim=1)
        mask = torch.cat([text_mask, vlm_mask], dim=1)
        return {"context": context, "context_mask": mask, "dropped_vlm": dropped}

    def maybe_drop_video_block(self, mot_mask: torch.Tensor, video_seq_len: int) -> torch.Tensor:
        """Source dropout on the action->video MoT block (training only)."""
        if self.training and torch.rand(()) < self.p_drop_video:
            mot_mask = mot_mask.clone()
            mot_mask[video_seq_len:, :video_seq_len] = False
        return mot_mask
