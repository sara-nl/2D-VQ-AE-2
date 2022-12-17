import logging
import os
import time
from typing import Optional, Union, Any, Sequence, Dict, Callable

import torch
import pytorch_lightning as pl
from torch import nn, optim
from lightning_lite.plugins.environments.slurm import SLURMEnvironment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


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


class ElementsPerSecond(pl.callbacks.Callback):

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
        logger.info("Calling IntelCPUInit")

        pl_module = pl_module.to(memory_format=torch.channels_last)
        optimizers = pl_module.configure_optimizers()

        if isinstance(optimizers, list):
            raise ValueError("Model has two or more optimizers, Intel CPU optimization only supports one!")

        dtype = {
            'bf16': torch.bfloat16,
            '16': torch.float16,
            '32': torch.float32,
            '64': torch.float64,
        }[str(trainer.precision)]

        _, optims = (
            self.optimize(model=pl_module, optimizer=optimizers, inplace=True, dtype=dtype)
            if self.optimize is not None
            else intel_extension_for_pytorch.optimize(
                model=pl_module,
                optimizer=optimizers,
                inplace=True,
                dtype=dtype
            )
        )

        logger.info("Overwriting pl_module.configure_optimizers")
        pl_module.configure_optimizers = lambda: optims

        def on_before_batch_transfer(batch, dataloader_idx):
            if isinstance(batch, torch.Tensor):
                return batch.to(memory_format=torch.channels_last)
            else:
                raise NotImplementedError(
                    "batch is not a straight torch.Tensor, so can't immediately cast to channels_last"
                    " you need to either stack your dataloader output as a single tensor,"
                    " or implement your batch transform here (in 2D-VQ-AE-2/utils/train_helpers.py)"
                )

        pl_module.on_before_batch_transfer = on_before_batch_transfer


class IMPIEnvironment(SLURMEnvironment):
    def __init__(self):
        super().__init__()

    def world_size(self) -> int:
        return int(os.environ['PMI_SIZE'])

    def global_rank(self) -> int:
        return int(os.environ['PMI_RANK'])

    def local_rank(self) -> int:
        return int(os.environ['MPI_LOCALRANKID'])
