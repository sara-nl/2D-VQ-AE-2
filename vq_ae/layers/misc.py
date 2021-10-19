import torch
from torch import nn

from utils.train_helpers import make_divisible


class SELayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_divisor: int,
    ):
        super().__init__()
        intermediate_channels = make_divisible(in_channels, bottleneck_divisor, divide=True)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, intermediate_channels),
            nn.SiLU(),
            nn.Linear(intermediate_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        b, c, *dim = x.size()
        ndims = len(dim)

        y = x.mean(dim=tuple(d for d in range(2, ndims+2)))
        y = self.fc(y).view(b, c, *(1 for _ in range(ndims)))

        return x * y
