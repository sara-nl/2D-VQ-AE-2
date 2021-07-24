from typing import List, Any
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf

@dataclass
class GPU_TitanRTX(SlurmQueueConf):
    time: int = 5
    partition: str = 'gpu_titanrtx'
    cpus_per_task: int = 6
    ntasks_per_node: int = 4
    gpus_per_node: int = 4

@dataclass
class GPU_TitanRTX_Shared(SlurmQueueConf):
    time: int = 5
    partition: str = 'gpu_titanrtx_shared'
    cpus_per_task: int = 6
    ntasks_per_node: int = 1
    gpus_per_node: int = 1

@dataclass
class GPU_RTX2080Ti(SlurmQueueConf):
    time: int = 5
    partition: str = 'gpu_rtx2080ti'
    cpus_per_task: int = 6
    ntasks_per_node: int = 4
    gpus_per_node: int = 4


@dataclass
class GPU_RTX2080Ti_Shared(SlurmQueueConf):
    time = 5
    partition: str = 'gpu_rtx2080ti_shared'
    cpus_per_task: int = 6
    ntasks_per_node: int = 1
    gpus_per_node: int = 1


for node in (
    GPU_TitanRTX,
    GPU_TitanRTX_Shared,
    GPU_RTX2080Ti,
    GPU_RTX2080Ti_Shared
):
    ConfigStore.instance().store(
        group="hydra/launcher",
        name=node.partition,
        node=node,
        provider="submitit_launcher",
    )