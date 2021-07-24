from pathlib import Path
import importlib
import pkgutil
# __all__ = [
#     str(f)
#     for f in Path(__file__).parent.rglob("*.py")
#     if "__" != f.stem[:2]
# ]

# del Path

# breakpoint()

# from . import *

def import_submodules(package, recursive=True):
    """ Import all submodules of a module, recursively, including subpackages
    :param recursive: bool
    :param package: package (name or actual module)
    :type package: str | module
    :rtype: dict[str, types.ModuleType]
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results


import conf
res = import_submodules(conf)