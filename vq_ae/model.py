from dataclasses import dataclass
from functools import partial
from argparse import Namespace

import torch
import pytorch_lightning as pl
from torch import nn
from omegaconf import MISSING
from hydra.utils import instantiate

from conf.experiment.optimizer import OptimizerConf


@torch.no_grad()
def _eval_metrics_log_dict(orig, pred):
    # FIXME: remove hardcoded data range
    metrics = (
        ('nmse', nmse),
        ('psnr', partial(psnr, data_range=4)),
    )
    return {
        func_name: func(orig, pred)
        for func_name, func in metrics
    }


@dataclass(eq=False)
class VQAE(pl.LightningModule):

    # first in line is the default
    optim_conf: OptimizerConf = MISSING

    num_layers: int = MISSING
    input_channels: int = MISSING
    base_network_channels: int = MISSING


    def __post_init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(self.input_channels, self.base_network_channels, kernel_size=1)
        self.output_conv = nn.Conv2d(self.base_network_channels, self.input_channels, kernel_size=1)

    def forward(self, data):
        x = data

        x = self.input_conv(x)
        x = self.output_conv(x)

        return x
        # encoded = self.encode(data)
        # decoded = self.decode(encoded)

        # return decoded

    def encode(self, data):
        pass

    def decode(self, quantizations):
        pass

    def configure_optimizers(self):
        return instantiate(self.optim_conf, params=self.parameters())

    def training_step(self, batch, batch_idx):
        breakpoint()
        return self.shared_step(batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, mode='val')

    def shared_step(self, batch, batch_idx, mode='train'):
        assert mode in ('train', 'val')

    # def huber(self, batch, batch_idx) -> Tuple[torch.Tensor, dict]:
    #     return self.loc_metric(batch, batch_idx, F.smooth_l1_loss)

    # def _parse_input_args(self, args: Namespace) -> None:
    #     attr_setter_(
    #         self,
    #         args,
    #         ('input_channels', lambda x: x >= 1),
    #         ('base_network_channels', lambda x: x >= 1)
    #     )
