"""FusionWAM: semantic (PaliGemma) + dynamics (Wan2.2 video DiT) + action DiT.

Coupling (openpi-style per-layer KV, generalized to heterogeneous towers):
the frozen VLM runs its own forward over [image + instruction]; every MoT
layer i attends a KV prefix computed from the depth-mapped Gemma layer g(i)
by a per-layer trainable adapter. The action expert therefore reads the
VLM's WHOLE depth — early layers where object position survives dose-flat,
late layers where instruction binding lives (probe verdict, 2026-08-12) —
exactly how pi0's action expert reads PaliGemma, except our towers were not
born head-compatible, so the adapter projects Gemma hidden states into the
MoT's shared attention-head space (openpi's per-expert width projections,
made explicit).

Video rows never attend the VLM prefix in stage 1 (frozen video tower keeps
its training distribution); action rows do, behind source dropout.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from fusionwam.models.wam_joint import WAMJoint
from fusionwam.models.paligemma import FrozenPaliGemmaEncoder


def uniform_layer_map(n_mot_layers: int, n_vlm_layers: int) -> List[int]:
    """Map MoT layer i -> Gemma layer index in [1..L] (0 is embeddings),
    uniform stride, monotone; both endpoints used."""
    if n_mot_layers == 1:
        return [n_vlm_layers]
    return [round(1 + i * (n_vlm_layers - 1) / (n_mot_layers - 1))
            for i in range(n_mot_layers)]


class VLMKVAdapter(nn.Module):
    """Per-MoT-layer K/V projections from depth-mapped VLM hidden states into
    the MoT attention-head space. V zero-initialized: the prefix contributes
    nothing at step 0 and cannot destabilize the pretrained towers."""

    def __init__(self, vlm_dim: int, inner_dim: int, layer_map: List[int]):
        super().__init__()
        self.layer_map = list(layer_map)
        self.k_projs = nn.ModuleList(
            [nn.Linear(vlm_dim, inner_dim) for _ in layer_map])
        self.v_projs = nn.ModuleList(
            [nn.Linear(vlm_dim, inner_dim) for _ in layer_map])
        for lin in self.v_projs:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, hidden_states: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        out = []
        for i, g in enumerate(self.layer_map):
            h = hidden_states[g].to(self.k_projs[i].weight.dtype)
            out.append({"k": self.k_projs[i](h), "v": self.v_projs[i](h)})
        return out


class FusionWAM(WAMJoint):
    """WAMJoint + per-layer VLM KV prefix on the action expert's attention."""

    # ------------------------------------------------------------------ build
    @classmethod
    def from_wan22_pretrained(
        cls,
        vlm_path: Optional[str] = None,
        p_drop_vlm: float = 0.3,
        p_drop_video: float = 0.3,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.p_drop_vlm = float(p_drop_vlm)
        model.p_drop_video = float(p_drop_video)
        model.vlm_encoder = None
        model.vlm_adapter = None
        model._extra_len = 0
        model._extra_kv_stash = None
        if vlm_path:
            model.attach_vlm(vlm_path)
        return model

    def attach_vlm(self, vlm_path: str):
        self.vlm_encoder = FrozenPaliGemmaEncoder(vlm_path).to(device=self.device)
        inner = int(self.mot.num_heads * self.mot.attn_head_dim)
        lmap = uniform_layer_map(self.mot.num_layers, self.vlm_encoder.num_layers)
        self.vlm_adapter = VLMKVAdapter(
            self.vlm_encoder.hidden_dim, inner, lmap
        ).to(device=self.device, dtype=self.torch_dtype)
        return self

    # -------------------------------------------------------------- prefix io
    @staticmethod
    def _first_frame_rgb01(video: torch.Tensor) -> torch.Tensor:
        """sample['video'] -> [B,3,H,W] in [0,1]; accepts [B,C,T,H,W] or
        [B,T,C,H,W], rescales from [-1,1] when min is negative."""
        if video.ndim != 5:
            raise ValueError(f"expected 5D video tensor, got {tuple(video.shape)}")
        frame0 = video[:, :, 0] if video.shape[1] == 3 else video[:, 0]
        if frame0.shape[1] != 3:
            raise ValueError(f"cannot locate RGB axis in {tuple(video.shape)}")
        frame0 = frame0.float()
        if frame0.min() < -1e-3:
            frame0 = (frame0 + 1.0) / 2.0
        return frame0.clamp(0.0, 1.0)

    def _build_vlm_prefix(self, images01: torch.Tensor, prompts) -> None:
        """Compute the per-layer KV prefix and arm the stash consumed by the
        threaded mot calls. Sets _extra_len for the mask builder."""
        hiddens, mask = self.vlm_encoder.layer_hidden(images01, list(prompts))
        # Padding columns would need a batched mask; MoT masks are global
        # [S_q, S_kv]. Keep shapes static: zero out padded tokens' hidden
        # states instead (their V contribution vanishes; K of a zeroed hidden
        # is a constant bias key shared batch-wide).
        m = mask.unsqueeze(-1)
        hiddens = [h * m for h in hiddens]
        self._extra_kv_stash = self.vlm_adapter(hiddens)
        self._extra_len = int(hiddens[0].shape[1])

    def _clear_vlm_prefix(self) -> None:
        self._extra_kv_stash = None
        self._extra_len = 0

    # ------------------------------------------------------------- overrides
    def training_loss(self, sample: Dict[str, Any]):
        use_vlm = (
            self.vlm_encoder is not None
            and not (self.training and torch.rand(()) < self.p_drop_vlm)
        )
        try:
            if use_vlm:
                self._build_vlm_prefix(
                    self._first_frame_rgb01(sample["video"]), sample["prompt"])
            return super().training_loss(sample)
        finally:
            self._clear_vlm_prefix()

    @torch.no_grad()
    def infer_action(self, *args, **kwargs):
        try:
            if self.vlm_encoder is not None:
                prompt = kwargs.get("prompt")
                image = kwargs.get("input_image")
                if image is not None and prompt is not None:
                    img = image if image.ndim == 4 else image.unsqueeze(0)
                    self._build_vlm_prefix(img.float().clamp(0, 1),
                                           [prompt] * img.shape[0])
            return super().infer_action(*args, **kwargs)
        finally:
            self._clear_vlm_prefix()

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
        # Source dropout on the action->video block (Gate-1/B2: each pathway
        # must learn to carry the task alone).
        if self.training and torch.rand(()) < getattr(self, "p_drop_video", 0.0):
            mask = mask.clone()
            mask[video_seq_len:, :video_seq_len] = False
        extra = getattr(self, "_extra_len", 0)
        if extra:
            # KV-only prefix columns: video rows never see the VLM (frozen
            # tower keeps its distribution); action rows do.
            left = torch.zeros(mask.shape[0], extra, dtype=torch.bool, device=device)
            left[video_seq_len:, :] = True
            mask = torch.cat([left, mask], dim=1)
        return mask

    # ---------------------------------------------------------- checkpointing
    def save_checkpoint(self, path, optimizer=None, step=None, trainable_only=False):
        # Upstream WAM.save_checkpoint persists only mot+proprio_encoder;
        # without this override the trained adapter is silently dropped from
        # every checkpoint. The frozen VLM is excluded (reloads from its
        # published weights; persisting it would add ~6G per step).
        payload = {
            "mot": self._mot_state_dict(trainable_only),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "trainable_only": bool(trainable_only),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.vlm_adapter is not None:
            payload["vlm_adapter"] = self.vlm_adapter.state_dict()
            payload["vlm_adapter_meta"] = {
                "layer_map": self.vlm_adapter.layer_map,
                "p_drop_vlm": float(self.p_drop_vlm),
                "p_drop_video": float(self.p_drop_video),
            }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        super().load_checkpoint(path, optimizer=optimizer)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "vlm_adapter" in payload:
            if self.vlm_adapter is None:
                raise ValueError(
                    "Checkpoint carries a VLM adapter but no VLM is attached; "
                    "construct the model with vlm_path set.")
            self.vlm_adapter.load_state_dict(payload["vlm_adapter"])
