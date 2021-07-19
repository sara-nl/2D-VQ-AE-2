from typing import List
from dataclasses import dataclass
from abc import ABC

from hydra.core.config_store import ConfigStore

from omegaconf import MISSING

@dataclass
class TransformConf(ABC):
    _target_: str


@dataclass
class RandomAugment:
    _target_: str = 'albumentations.RandomCrop'
    width: int = MISSING
    height: int = MISSING


@dataclass
class TransformCompose:
    _target_: str = 'albumentations.Compose'
    transforms: List[TransformConf] = MISSING



cs = ConfigStore.instance()
cs.store(
    group="transforms",
    name="compose",
    node=TransformCompose,
)
cs.store(
    group="transforms",
    name="random_augment",
    node=RandomAugment,
)