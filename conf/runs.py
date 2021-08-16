from typing import List, Any
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf.omegaconf import MISSING

from conf.experiment.experiment import Experiment


# @dataclass
# class TrainVQAE(Experiment):
#     defaults: List[Any] = field(default_factory=lambda: [
#         {"trainer": "ddp_trainer"},
#         {"model": "vq_ae"},
#         {"train_datamodule": "camelyon16_datamodule"}
#     ])


# cs = ConfigStore.instance()
# cs.store(
#     name="train_vqae",
#     node=TrainVQAE
# )