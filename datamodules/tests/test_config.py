from typing import Union
from dataclasses import dataclass

from hydra.utils import instantiate


class ObjA:
  def __init__(self, a: int):
    self.a = a

class ObjB:
  def __init__(self, obj_a: ObjA, b: int):
    self.b = b
    self.obj_a = obj_a

@dataclass
class ObjAConf:
  _target_: str = 'datamodules.tests.test_config.ObjA'
  a: int = 5

@dataclass
class ObjBConf:
  _target_: str = 'datamodules.tests.test_config.ObjB'
  obj_a: Union[ObjA, ObjAConf] = ObjAConf
  b: int = 7
  
# two different ways to obtain the same objects
obj_b_inst = instantiate(ObjBConf)
obj_b_init = ObjB(ObjA(a=5), b = 7)

print(f"From instantiate: obj_b.b={obj_b_inst.b}, obj_b.obj_a.a={obj_b_inst.obj_a.a}")
print(f"From initialise:  obj_b.b={obj_b_init.b}, obj_b.obj_a.a={obj_b_init.obj_a.a}")

breakpoint()
