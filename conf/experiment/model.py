from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from conf.experiment.optimizer import AdamConf, OptimizerConf
from conf.pytorch_lightning.abstacts import ModelConf

@dataclass
class VQAE(ModelConf):
    _target_: str = 'vq_ae.model.VQAE'
    _recursive_: bool = False # we init the optim locally

    optim_conf: OptimizerConf = AdamConf

    num_layers: int = 5
    input_channels: int = 1
    base_network_channels: int = 4

# cs = ConfigStore.instance()
# cs.store(
#     group='model',
#     name='vq_ae',
#     node=VQAE
# )