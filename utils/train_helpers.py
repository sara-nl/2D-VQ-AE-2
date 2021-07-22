from enum import Enum, auto

class Stage(Enum):
    '''model training stage'''
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()