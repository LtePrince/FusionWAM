"""FusionTrainer: Wan22Trainer with one fix — keep the VLM adapter training.

Wan22Trainer._apply_dit_only_train_mode freezes the whole model and re-enables
only `dit` (+ proprio_encoder). That would silently freeze the fusion
parameters, leaving the VLM branch untrained while everything appears to run —
exactly the class of silent failure the campaign taught us to close
structurally, not by convention.

Two overrides, and they must stay in lockstep:
- _apply_dit_only_train_mode re-enables `model.vlm_adapter` (the trainable
  K/V projections). The encoder wrapper (`model.vlm_encoder`) stays frozen —
  it owns no parameters outside the published VLM itself.
- _collect_trainable_params adds the adapter's parameters to the optimizer;
  requires_grad alone only produces gradients, it does not update anything.
"""

from fusionwam.training.trainer import Wan22Trainer


class FusionTrainer(Wan22Trainer):
    @staticmethod
    def _apply_dit_only_train_mode(model):
        Wan22Trainer._apply_dit_only_train_mode(model)
        vlm_encoder = getattr(model, "vlm_encoder", None)
        if vlm_encoder is not None:
            vlm_encoder.eval()
            vlm_encoder.requires_grad_(False)
        vlm_adapter = getattr(model, "vlm_adapter", None)
        if vlm_adapter is not None:
            vlm_adapter.train()
            vlm_adapter.requires_grad_(True)

    @staticmethod
    def _collect_trainable_params(model):
        params = Wan22Trainer._collect_trainable_params(model)
        vlm_adapter = getattr(model, "vlm_adapter", None)
        if vlm_adapter is not None:
            params.extend(vlm_adapter.parameters())
        return params
