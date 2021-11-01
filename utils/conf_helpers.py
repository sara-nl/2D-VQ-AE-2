from typing import List, Union, Any
from dataclasses import dataclass
from abc import ABC
from pathlib import Path
from functools import partial, reduce
from operator import add, mul, pow

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig, ListConfig, MISSING


OmegaConf.register_new_resolver(
    name="path.stem",
    resolver= lambda path: Path(path).stem,
    replace=True # need this for multirun
)
OmegaConf.register_new_resolver(
    name="path.absolute",
    resolver= lambda path: Path(path).absolute(),
    replace=True # need this for multirun
)

OmegaConf.register_new_resolver(
    name="len",
    resolver=lambda iterable: len([elem for elem in iterable if not (isinstance(elem, str) and len(elem) > 0 and elem[0] == '_')]),
    replace=True # need this for multirun
)

OmegaConf.register_new_resolver(
    name="add",
    resolver=lambda *x: reduce(add, x),
    replace=True # need this for multirun
)
OmegaConf.register_new_resolver(
    name="mul",
    resolver=lambda *x: reduce(mul, x),
    replace=True # need this for multirun
)
OmegaConf.register_new_resolver(
    name="pow",
    resolver=lambda x, y: pow(x, y),
    replace=True # need this for multirun
)



@dataclass
class DatasetConf(ABC):
    _target_: str = 'torch.utils.data.Dataset'

@dataclass
class DataloaderConf:
    _target_: str = 'torch.utils.data.DataLoader'

    dataset: DatasetConf = MISSING

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 6
    pin_memory: bool = True
    drop_last: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

@dataclass
class OptimizerConf(ABC):
    _target_: str = 'torch.optim.Optimizer'

@dataclass
class ModuleConf(ABC):
    _target_: str = 'torch.nn.Module'


def instantiate_nested_dictconf(**nested_conf) -> Any:
    listified_obj = instantiate_dictified_listconf(**nested_conf)

    assert len(listified_obj) == 1, f"more than one root object found in {listified_obj}"

    return listified_obj[0]


def instantiate_dictified_listconf(**nested_conf) -> List:
    '''
    Warning:
    set `_recursive_: False`
    at the same level as `_target_: utils.conf_helpers.instantiate_nested_conf`
    inside your config!
    '''

    de_nested = listify_nested_conf(nested_conf)

    if isinstance(de_nested, ListConfig):
        return [instantiate(elem) for elem in de_nested]

    return [instantiate(de_nested)]


def listify_nested_conf(conf: Any) -> Union[DictConfig, ListConfig]:
    '''
    Given the keys and values of a nested config,
    removes keys and makes their corresponding value a ListConfig,
    if the nest level doesn't contain the key `_target_`.

    Example input:
    ```
    compose:
        transforms:
            random_crop:
                _target_: albumentations.RandomCrop
                width: 128
                height: 128
            to_tensor_v2:
                _target_: albumentations.pytorch.transforms.ToTensorV2
        _target_: albumentations.Compose
    ```
    Example output:
    ```
    - transforms:
        - _target_: albumentations.RandomCrop
          width: 128
          height: 128
        - _target: albumentations.pytorch.transforms.ToTensorV2
      _target_: albumentations.Compose
    ```
    '''

    if isinstance(conf, (DictConfig, dict)):
        return (
            listify_nested_conf(ListConfig(list(conf.values())))
            if '_target_' not in conf.keys()
            else DictConfig({
                key: listify_nested_conf(value)
                for key, value in conf.items()
            })
        )
    elif isinstance(conf, (ListConfig, list)):
        return ListConfig([
            listify_nested_conf(value)
            for value in conf
        ])
    else:
        return conf




if __name__ == '__main__':
    conf = {
        'compose': {
            'transforms': {
                'random_crop': {
                    '_target_': 'albumentations.RandomCrop',
                    'width': 128,
                    'height': 128,
                },
                'to_tensor_v2': {
                    '_target_': 'albumentations.pytorch.transforms.ToTensorV2'
                }
            },
            '_target_': 'albumentations.Compose'
        }
    }
    print(listify_nested_conf(conf))