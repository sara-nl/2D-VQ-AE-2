from typing import Any, List
from dataclasses import dataclass, field
from abc import ABC

from hydra.core.config_store import ConfigStore

from omegaconf import MISSING

@dataclass
class TransformConf(ABC):
    _target_: str


@dataclass
class Compose(TransformConf):
    _target_: str = 'albumentations.Compose'
    transforms: List[TransformConf] = field(default_factory=list)


@dataclass
class RandomAugment(TransformConf):
    _target_: str = 'albumentations.RandomCrop'
    width: int = MISSING
    height: int = MISSING


# cs = ConfigStore.instance()
# cs.store(
#     group="transforms",
#     name="compose",
#     node=Compose,
# )
# cs.store(
#     group="transforms",
#     name="random_augment",
#     node=RandomAugment,
# )