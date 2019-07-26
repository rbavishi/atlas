from abc import ABC, abstractmethod
from typing import Any, Set, Union, List

import pandas as pd
import numpy as np

from enum import Enum, auto


class NodeDataTypeFeatures(Enum):
    OBJECT = auto()
    FLOAT = auto()
    INT = auto()
    STR = auto()
    DATE = auto()
    DELTA = auto()
    BOOL = auto()
    NAN = auto()
    NONE = auto()

    @classmethod
    def from_value(cls, val):
        val_type = type(val)

        if np.issubdtype(val_type, np.floating):
            if pd.isnull(val):
                val_type = cls.NAN
            else:
                val_type = cls.FLOAT
        elif np.issubdtype(val_type, np.signedinteger) or np.issubdtype(val_type, np.unsignedinteger):
            val_type = cls.INT
        elif np.issubdtype(val_type, np.str_):
            val_type = cls.STR
        elif np.issubdtype(val_type, np.bool_):
            val_type = cls.BOOL
        elif isinstance(val, pd.datetime):
            val_type = cls.DATE
        elif isinstance(val, pd.Timedelta):
            val_type = cls.DELTA
        elif val is None:
            val_type = cls.NONE
        else:
            try:
                if pd.isnull(val):
                    val_type = cls.NAN
                else:
                    val_type = cls.OBJECT
            except:
                val_type = cls.OBJECT

        return val_type


class NodeClassFeatures(Enum):
    COLUMN = auto()
    INDEX = auto()
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

    INDEX = auto()  # From index to cell
    INDEX_FOR = auto()  # From cell to index

    SUBSTR = auto()
    SUPSTR = auto()

    REPRESENTOR = auto()
    REPRESENTED = auto()


class GraphNode:
    def __init__(self, label: str, features: Set[Union[NodeClassFeatures,
                                                       NodeDataTypeFeatures]]):
        self.label = label
        self.features = features


class GraphEdge:
    def __init__(self, src: GraphNode, dst: GraphNode, feature: EdgeFeatures):
        self.src = src
        self.dst = dst
        self.feature = feature


class ValueEncoding(ABC):
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []

    @abstractmethod
    def apply(self, value: Any):
        pass


class ScalarEncoding(ValueEncoding):
    def apply(self, value):
        pass


class DataFrameEncoding(ValueEncoding):
    def apply(self, value: pd.DataFrame):
        pass
