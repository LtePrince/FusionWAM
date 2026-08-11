"""FusionWAM: semantic (PaliGemma) + dynamics (Wan2.2 video DiT) + action DiT.

Integration contract (kept deliberately narrow):
- WAM's trainer calls ``model.training_loss(sample)`` — we override it to
  inject projected VLM tokens into the action expert's ``context`` before
  delegating to the parent implementation. Trainer and datasets are reused
  verbatim (see fusionwam.trainer.FusionTrainer for the one freeze-mode fix).
- Video<->action coupling stays exactly WAMJoint's MoT joint attention;
  ``_build_mot_attention_mask`` is overridden only to apply source dropout on
  the action->video block during training.

Campaign-derived design rules encoded here (see README, "Design rationale"):
frozen VLM by default, source-level dropout on both KV sources, LayerScale on
the VLM branch starting near zero.
"""

from typing import Any, Dict, Optional, Tuple

import torch

try:  # WAM is installed on the training machine (uv pip install -e ../WAM);
    # the mask builder below stays importable without it (smoke test, tooling).
    from fusionwam.wam.models.wan22.wam_joint import WAMJoint
except ModuleNotFoundError:  # pragma: no cover
    WAMJoint = object

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
    """[VLM ⊕ video ⊕ action] joint-attention mask (True = attend). Stage 2.

    VLM    -> VLM                          (frozen prefix, self only)
    video  -> video (own mask) + VLM       (semantics condition dynamics)
    action -> action + video + VLM         (dual-KV consumption, per dropout flags)
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


class FusionWAM(WAMJoint):
    """Stage-1 fusion: WAMJoint + VLM context injection into the action expert."""

    # ------------------------------------------------------------------ build
    @classmethod
    def from_wan22_pretrained(
        cls,
        vlm_path: Optional[str] = None,
        p_drop_vlm: float = 0.3,
        p_drop_video: float = 0.3,
        vlm_layerscale_init: float = 1e-3,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.p_drop_vlm = float(p_drop_vlm)
        model.p_drop_video = float(p_drop_video)
        if vlm_path:
            model.attach_vlm(vlm_path, layerscale_init=vlm_layerscale_init)
        else:
            model.vlm_encoder = None
        return model

    def attach_vlm(self, vlm_path: str, layerscale_init: float = 1e-3):
        # The action expert embeds `context` via text_embedding: Sequential
        # whose first Linear maps text_dim -> hidden_dim.
        text_dim = self.action_expert.text_embedding[0].in_features
        self.vlm_encoder = FrozenPaliGemmaEncoder(
            vlm_path, out_dim=text_dim, layerscale_init=layerscale_init
        ).to(device=self.device, dtype=self.torch_dtype)
        return self

    # -------------------------------------------------------------- utilities
    @staticmethod
    def _first_frame_rgb01(video: torch.Tensor) -> torch.Tensor:
        """sample['video'] -> [B,3,H,W] in [0,1] for the VLM processor.

        Accepts [B,C,T,H,W] or [B,T,C,H,W]; rescales from [-1,1] when the
        tensor's minimum is negative (Wan-style normalization).
        """
        if video.ndim != 5:
            raise ValueError(f"expected 5D video tensor, got {tuple(video.shape)}")
        frame0 = video[:, :, 0] if video.shape[1] == 3 else video[:, 0]
        if frame0.shape[1] != 3:
            raise ValueError(f"cannot locate RGB axis in video shape {tuple(video.shape)}")
        frame0 = frame0.float()
        if frame0.min() < -1e-3:
            frame0 = (frame0 + 1.0) / 2.0
        return frame0.clamp(0.0, 1.0)

    def _fused_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video: torch.Tensor,
        prompts,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[umt5 ⊕ VLM] with source dropout on the VLM slice (mask zeroed,
        tokens kept, so shapes stay static for compile/accumulate)."""
        vlm_tokens, vlm_mask = self.vlm_encoder(
            self._first_frame_rgb01(video), list(prompts)
        )
        vlm_tokens = vlm_tokens.to(device=context.device, dtype=context.dtype)
        vlm_mask = vlm_mask.to(device=context_mask.device)
        if self.training and torch.rand(()) < self.p_drop_vlm:
            vlm_mask = torch.zeros_like(vlm_mask)
        return (
            torch.cat([context, vlm_tokens], dim=1),
            torch.cat([context_mask, vlm_mask], dim=1),
        )

    # ------------------------------------------------------------- overrides
    def training_loss(self, sample: Dict[str, Any]):
        if self.vlm_encoder is not None:
            context = sample["context"]
            context_mask = sample["context_mask"]
            if context.ndim == 2:  # unbatched collation guard, mirror parent
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=self.device, dtype=self.torch_dtype)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool)
            fused, fused_mask = self._fused_context(
                context, context_mask, sample["video"], sample["prompt"]
            )
            sample = dict(sample)
            sample["context"] = fused
            sample["context_mask"] = fused_mask
        return super().training_loss(sample)

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = super()._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # Source dropout, action->video block (Gate-1/B2: each pathway must
        # learn to carry the task alone).
        if self.training and torch.rand(()) < getattr(self, "p_drop_video", 0.0):
            mask = mask.clone()
            mask[video_seq_len:, :video_seq_len] = False
        return mask
