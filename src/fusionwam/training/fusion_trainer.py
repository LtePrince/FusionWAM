"""FusionTrainer: stage-1 trainable-set policy over Wan22Trainer.

The base class's "dit-only train mode" comes from the upstream video world
model, where `model.dit` IS the training target — and on WAM, `self.dit =
self.mot`, the WHOLE tri-tower mixture (frozen 5B video tower included).
Inheriting that here broke the stage-1 contract (configs/stage1.yaml:
"frozen backbones, train action expert + fusion adapter") in three ways:

1. The 5B video tower became trainable and entered the optimizer — off-design
   fine-tuning, and ZeRO's fp32 master/grad buffers over 6.4B params alone
   need ~75G (+ a 23.8G foreach transient in Adam: the H200 OOM, batch-size
   independent).
2. The VLM adapter was never re-enabled (the old override touched the
   parameterless vlm_encoder wrapper) nor collected into the optimizer.
3. The base sets the ROOT module to eval() and only re-enters train() on
   `dit` — but FusionWAM.training_loss gates source dropout on the root's
   self.training, so p_drop_vlm/p_drop_video 0.3/0.3 were silently OFF.

Policy here: root (and thus MoT) in train() mode — MoT gates gradient
checkpointing on its own training flag, and no dropout/BN lives in the
frozen towers, so train-mode is semantically safe for them; requires_grad
only on action expert + proprio + VLM adapter; the VLM itself eval+frozen.
`_collect_trainable_params` must mirror `_apply_dit_only_train_mode`
exactly: anything re-enabled but not collected gets gradients yet never
updates. Both hooks are also used by evaluate()'s mode restoration.
"""

from fusionwam.training.trainer import Wan22Trainer


class FusionTrainer(Wan22Trainer):
    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.train()
        model.requires_grad_(False)
        model.action_expert.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.requires_grad_(True)
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
        params = list(model.action_expert.parameters())
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            params.extend(proprio_encoder.parameters())
        vlm_adapter = getattr(model, "vlm_adapter", None)
        if vlm_adapter is not None:
            params.extend(vlm_adapter.parameters())
        return params
