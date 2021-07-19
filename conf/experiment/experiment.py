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
    val_datamodule: Optional[DataModuleConf] = None

@dataclass
class SeedEverything:
    _target_: str = 'pytorch_lighting.trainer.seed_everything'
    seed: int = MISSING
    workers: bool = True # no reason why this should be False

cs = ConfigStore.instance()
cs.store(
    group="experiments",
    name="base_experiment",
    node=Experiment,
)
cs.store(
    group="experiments",
    name="seed_everything",
    node=SeedEverything
)