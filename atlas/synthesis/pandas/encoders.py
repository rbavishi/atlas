from enum import Enum, auto


class NodeFeatures(Enum):
    COLUMN = auto()
    INDEX = auto()
    OBJECT = auto()
    FLOAT = auto()
    INT = auto()
    STR = auto()
    DATE = auto()
    DELTA = auto()
    BOOL = auto()
    NAN = auto()
    NONE = auto()
    DEFAULT = auto()
    TERMINAL = auto()
    REPRESENTOR = auto()


class EdgeFeatures(Enum):
    #  Naming Convention : Name of edge describes the role of src for dst

    ADJ_LEFT = auto()
    ADJ_RIGHT = auto()
    ADJ_ABOVE = auto()
    ADJ_BELOW = auto()
    EQUALITY = auto()

    INDEX = auto()      # From index to cell
    INDEX_FOR = auto()  # From cell to index

    SUBSTR = auto()
    SUPSTR = auto()

    REPRESENTOR = auto()
    REPRESENTED = auto()
