from dataclasses import dataclass
from typing import Optional, Callable

import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data.dataloader import DataLoader

from utils.conf_helpers import DataloaderConf


@dataclass
class DefaultDataModule(pl.LightningDataModule):
    train_dataloader_conf: Callable[[], DataLoader]
    val_dataloader_conf: Callable[[], DataLoader]
    test_dataloader_conf: Optional[Callable[[], DataLoader]] = None

    def __post_init__(self):
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        return self.train_dataloader_conf()

    def val_dataloader(self) -> DataLoader:
        return self.val_dataloader_conf()

    def test_dataloader(self) -> DataLoader:
        return self.test_dataloader_conf()

