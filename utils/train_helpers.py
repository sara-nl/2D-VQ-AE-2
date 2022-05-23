import time
from typing import Optional, Union, Any, Sequence, Dict

import torch
import pytorch_lightning as pl

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
    def on_setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage = None) -> None:
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