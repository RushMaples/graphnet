"""Deterministic training-only augmentation for the P14 ablation."""

from pytorch_lightning import Callback

from graphnet.training.callbacks import RandomRotationCallback


class AugmentationCallback(Callback):
    """Use independent epoch/batch seeds without changing global dropout RNG."""

    def __init__(self, seed, rotation=False):
        self.seed = seed
        self.rotation = (
            RandomRotationCallback(
                seed=seed,
                node_xy_origin=(46.3 / 500.0, -34.9 / 500.0),
                position_xy_origin=(46.3, -34.9),
            )
            if rotation
            else None
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        seed = (
            self.seed
            + trainer.current_epoch * 1_000_000_007
            + batch_idx * 1_000_003
            + int(getattr(trainer, "global_rank", 0)) * 10_007
        ) % (2**63 - 1)
        pl_module.backbone.set_token_drop_seed(seed)
        if self.rotation is not None:
            self.rotation._generator.manual_seed(seed)
            self.rotation.on_train_batch_start(
                trainer, pl_module, batch, batch_idx
            )
