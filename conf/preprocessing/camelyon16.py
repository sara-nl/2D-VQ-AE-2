from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from conf.preprocessing.torch_data import DataloaderConf, DatasetConf
from conf.preprocessing.transforms import Compose, RandomAugment, TransformConf
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
    transforms: Optional[TransformConf] = None


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


# cs = ConfigStore.instance()
# cs.store(
#     group="train_datamodule",
#     name="camelyon16_datamodule",
#     node=CAMELYON16DataModuleConf,
# )
# cs.store(
#     group="train_datamodule/train_dataloader",
#     name="camelyon16_dataloader",
#     node=CAMELYON16DataloaderConf,
# )
# cs.store(
#     group="transforms",
#     name="camelyon16_transforms",
#     node=Compose(

#     )
# )