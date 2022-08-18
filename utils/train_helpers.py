import logging
import time
from typing import Optional, Union, Any, Sequence, Dict, Callable

import torch
import pytorch_lightning as pl

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn, optim


def make_divisible(
        value: float,
        divisor: int,
        divide: bool = True,
        min_value: Optional[int] = None
):
    """
    Edited from: https://github.com/d-li14/efficientnetv2.pytorch/blob/775326e6c16bfc863e9b8400eca7d723dbfeb06e/effnetv2.py#L16
    """

    return max(
        divisor if min_value is None else min_value,
        int(value + divisor / 2)
    ) // (divisor if divide else 1)


def maybe_repeat_layer(layer: Union[Any, Sequence], repetitions: int) -> Sequence:
    if not isinstance(layer, Sequence):
        return [layer] * repetitions
    else:
        assert len(layer) == repetitions
        return layer


class ChannelsLast(pl.Callback):
    def on_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage=None) -> None:
        # Inplace model modification
        pl_module.to(memory_format=torch.channels_last)
        pl_module.configure_optimizers()


class ElementsPerSecond(pl.Callback):

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        if not trainer.loggers:
            raise MisconfigurationException("Cannot use DeviceStatsMonitor callback with Trainer that has no logger.")

    def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:

        if not trainer.loggers:
            raise MisconfigurationException("Cannot use `ElementsPerSecond` callback with `Trainer(logger=False)`.")

        if not trainer._logger_connector.should_update_logs:
            return

        self.train_start_time = time.perf_counter()

    def on_train_batch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            unused: int = 0,
    ) -> None:
        if not trainer.loggers:
            raise MisconfigurationException("Cannot use `ElementsPerSecond` callback with `Trainer(logger=False)`.")

        if not trainer._logger_connector.should_update_logs:
            return

        try:
            elapsed_time = time.perf_counter() - self.train_start_time
            del self.train_start_time  # to enable this check for subsequent iters
        except AttributeError:
            raise RuntimeError("Reached batch_end before batch_start")

        # else assumes that batch is a tuple of which the first element is the data
        batch_size = (
            batch.shape[0]
            if isinstance(batch, torch.Tensor)
            else batch[0].shape[0]
        )

        imgs_per_sec = round(batch_size / elapsed_time, 3)

        for logger in trainer.loggers:
            logger.log_metrics({
                'train_unnormalized_batch_elems_per_sec': imgs_per_sec
            }, step=trainer.fit_loop.epoch_loop._batches_that_stepped)


class Camelyon16BCELoss(torch.nn.BCEWithLogitsLoss):

    def __init__(
            self,
            weight: Optional[torch.Tensor] = None,
            size_average=None,
            reduce=None,
            reduction='mean',
            pos_weight=None,
            label_smoothing: float = 0,
    ):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
        self.register_buffer('label_smoothing', torch.as_tensor(label_smoothing))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (0 <= target).logical_and(target <= 2).all():
            raise ValueError(
                "Camelyon16 Targets values are assumed to be 0 (background), 1 (tissue) and 2 (cancer)."
                f" Instead, found {target.unique()}"
            )

        mask = target != 0
        target = target.clone()

        input, target = (
            input[mask.unsqueeze(dim=1)].unsqueeze(dim=0),
            (target[mask] - 1).unsqueeze(dim=0)
        )

        if not target.is_floating_point():
            target = target.type_as(input)

        if self.label_smoothing != 0:
            # target = (1 - (target + torch.randn_like(target) * self.label_smoothing % 2)) % 1
            target = (1 - ((1 + target + torch.randn_like(target) * self.label_smoothing) % 2)).abs()
            # target = (target + torch.randn_like(target) * self.label_smoothing).clamp(0, 1)

        return super().forward(input, target)


class IntelCPUInit(pl.Callback):
    """
    Initializes IPEX, oneccl_bindings_for_pytorch, and optimizes model based on IPEX.optimize
    """

    def __init__(self, optimize: Optional[Callable[[nn.Module, optim.Optimizer], Optional[nn.Module]]] = None):
        # optimize should be e.g. a functools.partial intel_extension_for_pytorch.optimize
        # only including it, so we can add optimizes hparams in a config
        self.optimize = optimize

        import oneccl_bindings_for_pytorch  # import magic
        import intel_extension_for_pytorch


    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage=None) -> None:

        logger = logging.getLogger(__name__)
        logger.info("Calling optimize")
        return
        if isinstance(pl_module.optimizers(), list):
            raise ValueError("Model has two or more optimizers, Intel CPU optimization only supports one!")

        if not self.optimize:
            # noinspection PyUnresolvedReferences
            intel_extension_for_pytorch.optimize(
                model=pl_module,
                optimizer=pl_module.optimizers(),
                inplace=True,
                split_master_weight_for_bf16=True,
                fuse_update_step=True,
                auto_kernel_selection=True
            )
        else:
            breakpoint()
            self.optimize(model=pl_module, optimizer=pl_module.optimizers(), dtype=torch.bfloat16)
