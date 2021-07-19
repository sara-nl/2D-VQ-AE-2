from dataclasses import dataclass

from omegaconf import MISSING

@dataclass
class Path:
    _target_: str = 'pathlib.Path'
    pathsegments: str = MISSING