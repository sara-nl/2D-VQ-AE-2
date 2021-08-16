from typing import List, Any
from dataclasses import dataclass, field

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from conf.pytorch_lightning.abstacts import TrainerConf, ModelConf, DataModuleConf

defaults = [
    {'trainer': MISSING},
    {'model': MISSING},
    {'train_datamodule': MISSING}
]

@dataclass
class Experiment:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    trainer: TrainerConf = MISSING
    model: ModelConf = MISSING
    train_datamodule: DataModuleConf = MISSING

# cs = ConfigStore.instance()
# cs.store(
#     name="base_experiment",
#     node=Experiment,
# )
