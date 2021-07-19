from abc import ABC
from dataclasses import dataclass

@dataclass
class TrainerConf(ABC):
    _target_: str = 'pytorch_lightning.Trainer'

@dataclass
class ModelConf(ABC):
    _target_: str = 'pytorch_lightning.LightningModule'

@dataclass
class DatamoduleConf(ABC):
    _target_: str = 'pytorch_lightning.DataModule'



