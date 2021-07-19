from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from conf.preprocessing.torch_data import DataloaderConf, DatasetConf
from conf.pytorch_lightning.abstacts import DataModuleConf


@dataclass
class Camelyon16DatasetConf(DatasetConf):
    _target_: str = 'datamodules.camelyon16.Camelyon16Dataset'


@dataclass
class Camelyon16DataModuleConf(DataModuleConf):
    _target_: str = 'datamodules.camelyon16.Camelyon16DataModule'
    train_dataloader_conf: DataloaderConf = MISSING
    val_dataloader_conf: DataloaderConf = MISSING
    dataset_conf: Camelyon16DatasetConf = MISSING


cs = ConfigStore.instance()
cs.store(
    group="preprocessing",
    name="camelyon16_dataset",
    node=Camelyon16DatasetConf,
)
cs.store(
    group="preprocessing",
    name="camelyon16_datamodule",
    node=Camelyon16DataModuleConf,
)