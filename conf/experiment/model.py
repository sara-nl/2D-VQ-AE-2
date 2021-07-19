from dataclasses import dataclass

from conf.pytorch_lightning.abstacts import ModelConf

@dataclass
class VQAE(ModelConf):
    num_layers: int = 5