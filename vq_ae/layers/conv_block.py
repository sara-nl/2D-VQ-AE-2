from collections.abc import Iterable, Sequence
from functools import partial
from math import isclose
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn

from utils.conf_helpers import ModuleConf

# List of elements | single element | single element + repetitions
ModuleConfList = Union[List[ModuleConf], Union[ModuleConf, Tuple[ModuleConf, int]]]


class DownBlock(nn.Module):
    out_channels: int

    def __init__(
        self,
        in_channels: int,
        n_down: int,
        conv_conf: ModuleConf,
        n_pre_layers: Optional[int],
        n_post_layers: Optional[int]
    ):
        super().__init__()

        pre_layers, post_layers = [
            [{**conv_conf, **{'mode': 'same'}}] * n_layers
            for n_layers in (n_pre_layers, n_post_layers)
        ]
        self.layers = nn.Sequential(
            *(
                EnvelopBlock(
                    envelop_conf={**conv_conf, **{'mode': 'down'}},
                    in_channels=in_c,
                    out_channels=out_c,
                    pre_layers=pre_layers,
                    post_layers=post_layers
                )
                for in_c, out_c in (
                (in_channels * (2 ** j), in_channels * (2 ** (j + 1))) for j in range(n_down)
            )
            )
        )
        self.out_channels = in_channels * 2 ** n_down

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    """basically a DownBlock in reverse"""
    in_channels: int

    def __init__(
        self,
        out_channels: int,
        n_up: int,
        conv_conf: ModuleConf,
        n_pre_layers: Optional[int],
        n_post_layers: Optional[int]
    ):
        super().__init__()

        pre_layers, post_layers = [
            [{**conv_conf, **{'mode': 'same'}}] * n_layers
            for n_layers in (n_pre_layers, n_post_layers)
        ]
        self.layers = nn.Sequential(
            *(
                EnvelopBlock(
                    envelop_conf={**conv_conf, **{'mode': 'up'}},
                    in_channels=in_c,
                    out_channels=out_c,
                    pre_layers=pre_layers,
                    post_layers=post_layers
                )
                for in_c, out_c in (
                (out_channels * (2 ** (j + 1)), out_channels * (2 ** j)) for j in
                range(n_up - 1, -1, -1)
            )
            )
        )
        self.in_channels = out_channels * 2 ** n_up

    def forward(self, x):
        return self.layers(x)


class EnvelopBlock(nn.Module):
    def __init__(
        self,
        envelop_conf: ModuleConf,
        in_channels: int,
        out_channels: int,
        pre_layers: Optional[ModuleConfList] = None,
        post_layers: Optional[ModuleConfList] = None
    ):
        super().__init__()

        def instantiate_layers(
            layers: Optional[ModuleConfList],
            in_channels: int,
            out_channels: int
        ) -> Iterable:
            return map(
                partial(instantiate, in_channels=in_channels, out_channels=out_channels),
                filter(
                    None, (
                        ((layers[0] for _ in range(layers[1]))
                         if len(layers) == 2 and isinstance(layers[1], int)
                         else layers)
                        if isinstance(layers, Sequence)
                        else (layers,))
                )
            )

        self.layers = nn.Sequential(
            *instantiate_layers(pre_layers, in_channels, in_channels),
            instantiate(envelop_conf, in_channels=in_channels, out_channels=out_channels),
            *instantiate_layers(post_layers, out_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class PreActFixupResBlock(nn.Module):
    # Adapted from:
    # https://github.com/hongyi-zhang/Fixup/blob/master/imagenet/models/fixup_resnet_imagenet.py#L20

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str,
        bottleneck_divisor: float,
        activation: ModuleConf,
        conv_conf: ModuleConf,
        n_layers: Optional[int] = None,
    ):
        super().__init__()

        assert mode in ("down", "same", "up", "out")
        conv_conf = conv_conf[mode]

        max_channels = max(in_channels, out_channels)
        assert isclose(max_channels % bottleneck_divisor, 0), (
            f"residual channels: {max_channels} not divisible by bottleneck divisor: {bottleneck_divisor}!"
        )
        branch_channels = max(round(max_channels / bottleneck_divisor), 1)

        self.activation = instantiate(activation)

        self.params = nn.ParameterDict({
            **{bias: nn.Parameter(torch.zeros(1)) for bias in
                ("bias1a", "bias1b", "bias2a", "bias2b", "bias3a", "bias3b", "bias4")},
            'scale': nn.Parameter(torch.ones(1))
        })

        self.branch_conv1 = instantiate(
            conv_conf['branch_conv1'],
            in_channels=in_channels,
            out_channels=branch_channels
        )
        self.branch_conv2 = instantiate(
            conv_conf['branch_conv2'],
            in_channels=branch_channels,
            out_channels=branch_channels
        )
        self.branch_conv3 = instantiate(
            conv_conf['branch_conv3'],
            in_channels=branch_channels,
            out_channels=out_channels
        )

        if not (mode in ("same", "out") and in_channels == out_channels):
            self.params['bias1c'], self.params['bias1d'] = (
                nn.Parameter(torch.zeros(1))
                for _ in range(2)
            )
            self.skip_conv = instantiate(
                conv_conf['skip_conv'],
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.skip_conv = None

        if n_layers is not None:
            self.initialize_weights(n_layers)

    def forward(self, inp: torch.Tensor):
        out = inp

        out = self.activation(out + self.params['bias1a'])
        out = self.branch_conv1(out + self.params['bias1b'])

        out = self.activation(out + self.params['bias2a'])
        out = self.branch_conv2(out + self.params['bias2b'])

        out = self.activation(out + self.params['bias3a'])
        out = self.branch_conv3(out + self.params['bias3b'])

        out = out * self.params['scale'] + self.params['bias4']

        out = out + (
            self.skip_conv(inp + self.params['bias1c']) + self.params['bias1d']
            if self.skip_conv is not None
            else inp
        )

        return out

    @torch.no_grad()
    def initialize_weights(self, num_layers):

        # branch_conv1
        weight = self.branch_conv1.weight
        nn.init.normal_(
            weight,
            mean=0,
            std=np.sqrt(2 / (weight.shape[0] * np.prod(weight.shape[2:]))) * num_layers ** (-0.5)
        )

        # branch_conv2
        nn.init.kaiming_normal_(self.branch_conv2.weight)

        # branch_conv3
        nn.init.constant_(self.branch_conv3.weight, val=0)

        # skip_conv
        if self.skip_conv is not None:
            nn.init.xavier_normal_(self.skip_conv.weight)


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str,
        expand_ratio: float,
        activation_conf: ModuleConf,
        conv_conf: ModuleConf,
        batchnorm_conf: Optional[ModuleConf],
        se_conf: Optional[ModuleConf]
    ):
        super().__init__()

        assert mode in ("down", "same", "up", "out")
        conv_conf = conv_conf[mode]

        max_channels = max(in_channels, out_channels)
        assert isclose(max_channels * expand_ratio % 1, 0), (
            f"max_channels: {max_channels} x expand_ratio: {expand_ratio} % 1 !≈ 0!"
        )
        max_channels = round(max_channels * expand_ratio)

        self.branch = nn.Sequential(
            *filter(
                None, map(
                    lambda x: instantiate(**x), (
                        {
                            'config': conv_conf['branch_conv1'],
                            'in_channels': in_channels,
                            'out_channels': max_channels
                        }, {
                            'config': batchnorm_conf,
                            'num_features': max_channels
                        }, {
                            'config': activation_conf
                        }, {
                            'config': conv_conf['branch_conv2'],
                            'in_channels': max_channels,
                            'out_channels': max_channels,
                            'groups': max_channels
                        }, {
                            'config': batchnorm_conf,
                            'num_features': max_channels
                        }, {
                            'config': activation_conf
                        }, {
                            'config': se_conf,
                            'in_channels': max_channels,
                            'out_channels': max_channels,
                        }, {
                            'config': conv_conf['branch_conv3'],
                            'in_channels': max_channels,
                            'out_channels': out_channels
                        }, {
                            'config': batchnorm_conf,
                            'num_features': out_channels
                        }
                    )
                )
            )
        )

        if not (mode in ("same", "out") and in_channels == out_channels):
            self.skip_conv = instantiate(
                conv_conf['skip_conv'],
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.skip_conv = None

        # init batchnorm gamma to 0
        with torch.no_grad():
            self.branch[-1].weight *= 0

    def forward(self, x):
        return self.branch(x) + (
            x
            if self.skip_conv is None
            else self.skip_conv(x)
        )
