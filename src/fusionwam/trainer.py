"""FusionTrainer: Wan22Trainer with one fix — keep the VLM adapter trainable.

Wan22Trainer._apply_dit_only_train_mode freezes the whole model and re-enables
only `dit` (+ proprio_encoder). That would silently freeze the fusion
parameters (projection / norm / segment / LayerScale), leaving the VLM branch
untrained while everything appears to run — exactly the class of silent
failure the campaign taught us to close structurally, not by convention.
"""

from fastwam.trainer import Wan22Trainer


class FusionTrainer(Wan22Trainer):
    @staticmethod
    def _apply_dit_only_train_mode(model):
        Wan22Trainer._apply_dit_only_train_mode(model)
        vlm_encoder = getattr(model, "vlm_encoder", None)
        if vlm_encoder is not None:
            # Adapter trains; the VLM itself stays frozen (stage 1).
            vlm_encoder.train()
            vlm_encoder.requires_grad_(True)
            vlm_encoder.vlm.eval()
            vlm_encoder.vlm.requires_grad_(False)
