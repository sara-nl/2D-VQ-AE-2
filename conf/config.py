from typing import List, Any

from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING

from conf.slurm.lisa_nodes import *

defaults = [
    {"launcher": "gpu_titanrtx"}
]

@dataclass
class Config:
    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Hydra will populate this field based on the defaults list
    launcher: Any = MISSING

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_path=None, config_name="config")
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()