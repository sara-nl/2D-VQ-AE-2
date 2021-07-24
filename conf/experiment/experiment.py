from dataclasses import MISSING, dataclass
from typing import Optional

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from conf.pytorch_lightning.abstacts import TrainerConf, ModelConf, DataModuleConf

@dataclass
class Experiment:
    trainer: TrainerConf = MISSING
    model: ModelConf = MISSING
    train_datamodule: DataModuleConf = MISSING

cs = ConfigStore.instance()
cs.store(
    name="base_experiment",
    node=Experiment,
)
