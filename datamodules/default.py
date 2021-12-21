from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data.dataloader import DataLoader

from utils.conf_helpers import DataloaderConf


@dataclass
class DefaultDataModule(pl.LightningDataModule):
    train_dataloader_conf: DataloaderConf
    val_dataloader_conf: DataloaderConf
    test_dataloader_conf: Optional[DataloaderConf] = None

    def __post_init__(self):
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        return instantiate(self.train_dataloader_conf)

    def val_dataloader(self) -> DataLoader:
        return instantiate(self.val_dataloader_conf)

    def test_dataloader(self) -> DataLoader:
        return instantiate(self.test_dataloader_conf)

