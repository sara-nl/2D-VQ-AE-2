from conf.preprocessing.transforms import TransformCompose
from dataclasses import dataclass
from abc import ABC

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from conf.preprocessing.transforms import TransformCompose

@dataclass
class DatasetConf(ABC):
    _target_: str = 'torch.utils.data.Dataset'


@dataclass
class DataloaderConf:
    _target_: str = 'torch.utils.data.DataLoader'

    dataset: DatasetConf = MISSING

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 6
    pin_memory: bool = True
    drop_last: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


cs = ConfigStore.instance()
cs.store(
    group="torch_utils_data",
    name='train_dataloader',
    node=DataloaderConf,
)
cs.store(
    group="torch",
    name='val_dataloader',
    node=DataloaderConf(shuffle=False, persistent_workers=False),
)


