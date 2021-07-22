from dataclasses import dataclass
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from conf.preprocessing.torch_data import DataloaderConf, DatasetConf
from conf.preprocessing.transforms import TransformCompose
from conf.pytorch_lightning.abstacts import DataModuleConf
from utils.train_helpers import Stage



@dataclass
class CAMELYON16DatasetConf(DatasetConf):
    _target_: str = 'datamodules.camelyon16.CAMELYON16RandomPatchDataSet'

    path: str = MISSING

    train: Stage = Stage.TRAIN
    train_frac: float = 0.95

    spacing: float = 0.5
    spacing_tolerance: float = 0.15
    patch_size: Tuple[int, int] = 128, 128
    n_patches_per_wsi: int = 1000
    transforms: Optional[TransformCompose] = None


@dataclass
class CAMELYON16DataloaderConf(DataloaderConf):
    dataset: CAMELYON16DatasetConf = CAMELYON16DatasetConf

@dataclass
class CAMELYON16DataModuleConf(DataModuleConf):
    _target_: str = 'datamodules.camelyon16.CAMELYON16DataModule'
    _recursive_: bool = False # To ensure dataloaders can be repeatedly init

    train_dataloader_conf: CAMELYON16DataloaderConf = CAMELYON16DataloaderConf()
    val_dataloader_conf: CAMELYON16DataloaderConf = CAMELYON16DataloaderConf(
        dataset=CAMELYON16DatasetConf(train=Stage.VALIDATION),
        shuffle=False
    )
    test_dataloader_conf: Optional[CAMELYON16DataloaderConf] = None


cs = ConfigStore.instance()
cs.store(
    group="preprocessing",
    name="camelyon16_datamodule",
    node=CAMELYON16DataModuleConf,
)