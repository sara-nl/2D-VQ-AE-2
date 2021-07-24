from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from conf.pytorch_lightning.abstacts import TrainerConf


@dataclass
class DDPGPUTrainer(TrainerConf):
    gpus: str = "-1"
    distributed_backend: str = "ddp"

    benchmark: bool = True

    num_sanity_val_steps: int = 0
    precision: int = 16

    log_every_n_steps: int = 50
    val_check_interval: float = 0.5
    flush_logs_every_n_steps: int = 100
    weights_summary: str = 'full'

    max_epochs: int = int(1e4)


@dataclass
class SeedEverything:
    _target_: str = 'pytorch_lighting.trainer.seed_everything'
    seed: int = MISSING
    workers: bool = True # no reason why this should be False


cs = ConfigStore.instance()
cs.store(
    group='trainer',
    name="ddp_trainer",
    node=DDPGPUTrainer,
)
cs.store(
    group="trainer/options",
    name="seed_everything",
    node=SeedEverything
)