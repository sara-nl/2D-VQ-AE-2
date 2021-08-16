from abc import ABC
from dataclasses import dataclass
from typing import Any, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

@dataclass
class OptimizerConf(ABC):
    _target_: str = 'torch.optim.Optimizer'

@dataclass
class AdamConf(OptimizerConf):
    _target_: str = 'torch.optim.Adam'

    lr: float = 1e-3
    betas: Tuple[float, float] = 0.9, 0.999
    eps: float = 1e-8
    weight_decay: float = 0
    amsgrad: bool = False

# cs = ConfigStore.instance()
# cs.store(
#     group='optim',
#     name='adam',
#     node=AdamConf
# )