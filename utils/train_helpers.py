from typing import Optional, Union, Any, Sequence

import torch
import pytorch_lightning as pl


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
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Inplace model modification
        pl_module.to(memory_format=torch.channels_last)
