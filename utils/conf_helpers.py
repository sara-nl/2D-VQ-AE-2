from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import add, mul, pow
from pathlib import Path
from typing import Any, List, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, MISSING, OmegaConf


def add_resolvers() -> None:
    """Adds resolvers to the OmegaConf parsers"""

    def add_resolver(name: str, resolver: Callable):
        OmegaConf.register_new_resolver(
            name=name,
            resolver=resolver,
            replace=True  # need this for multirun
        )

    for name_, resolver_ in (
        ("path.stem", lambda path: Path(path).stem),
        ("path.absolute", lambda path: Path(path).absolute()),
        # calculate length of any list or object,
        # but skip list elements if they are a `str` which starts with '_'
        ("len", lambda iterable: len([
            elem for elem in iterable
            if not (isinstance(elem, str) and len(elem) > 0 and elem[0] == '_')
        ])),
        ("add", lambda *x: reduce(add, x)),
        ("mul", lambda *x: reduce(mul, x)),
        ("pow", lambda x, y: pow(x, y))
    ):
        add_resolver(name_, resolver_)


@dataclass
class DatasetConf(DictConfig):
    _target_: str = 'torch.utils.data.Dataset'


@dataclass
class DataloaderConf(DictConfig):
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
class OptimizerConf(DictConfig):
    _target_: str = 'torch.optim.Optimizer'


@dataclass
class ModuleConf(DictConfig):
    _target_: str = 'torch.nn.Module'


def instantiate_nested_dictconf(**nested_conf: DictConfig) -> Any:
    listified_obj = instantiate_dictified_listconf(**nested_conf)

    assert len(listified_obj) == 1, f"more than one root object found in {listified_obj}"

    return listified_obj[0]


def instantiate_dictified_listconf(**nested_conf: DictConfig) -> List:
    """
    Warning:
    set `_recursive_: False`
    at the same level as `_target_: utils.conf_helpers.instantiate_nested_conf`
    inside your config!
    """

    de_nested = listify_nested_conf(nested_conf)

    if isinstance(de_nested, ListConfig):
        return [instantiate(elem) for elem in de_nested]

    return [instantiate(de_nested)]


def listify_nested_conf(conf: Any) -> Union[DictConfig, ListConfig]:
    """
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
    """

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
