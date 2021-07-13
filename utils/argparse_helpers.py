from typing import Union, Tuple, Callable
from argparse import Namespace

def attr_setter_(obj: object, args: Namespace, *attr_names: Union[str, Tuple[str, Callable]]):
    '''
    Takes attrs from args based on attr_names, and sets them as attrs in obj.
    Will perform an optional assert if provided
    '''
    for attr_name in attr_names:
        if isinstance(attr_name, tuple) and len(attr_name) == 2:
            attr_name, condition = attr_name
        else:
            condition = None

        attr = getattr(args, attr_name)
        if condition is not None:
            assert condition(attr)

        setattr(obj, attr_name, attr)


def booltype(inp: str) -> bool:
    if type(inp) is str:
        if inp.lower() == 'true':
            return True
        elif inp.lower() == 'false':
            return False
    raise ValueError(f"input should be either 'True', or 'False', found {inp}")