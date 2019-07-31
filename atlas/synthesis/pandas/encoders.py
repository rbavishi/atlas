import collections
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Set, List, Dict

import numpy as np
import pandas as pd


class NodeFeatures(Enum):
    pass


class NodeDataTypes(NodeFeatures):
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


class NodeRoles(NodeFeatures):
    COLUMN = auto()
    INDEX = auto()
    DEFAULT = auto()
    TERMINAL = auto()
    REPRESENTOR = auto()

    INDEX_NAME = auto()
    COLUMN_NAME = auto()


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
    def __init__(self, label: str, features: Set[NodeFeatures]):
        self.label = label
        self.features = features


class GraphEdge:
    def __init__(self, src: GraphNode, dst: GraphNode, feature: EdgeFeatures):
        self.src = src
        self.dst = dst
        self.feature = feature


class ValueEncoding(ABC):
    EQUALITY_EDGES = True

    def __init__(self, label: str, val: Any):
        self.label = label
        self.val = val
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []

        self.val_node_map: Dict[Any, List[GraphNode]] = collections.defaultdict(list)

    @abstractmethod
    def build(self):
        pass


class ScalarEncoding(ValueEncoding):
    def __init__(self, label: str, val: Any, features: Set[NodeFeatures] = None):
        super().__init__(label, val)

        if features is None:
            features = set()

        features.add(NodeDataTypes.from_value(val))
        self.features = features

    def build(self):
        self.nodes.append(GraphNode(self.label, self.features))
        self.val_node_map[self.val].append(self.nodes[-1])


class DataFrameEncoding(ValueEncoding):
    CELL_NODES = True
    INDEX_NODES = True
    COLUMN_NODES = True
    INDEX_NAME_NODES = False
    COLUMN_NAME_NODES = False

    INNER_EQUALITY_EDGES = True
    ADJACENCY_EDGES = True
    INDEX_EDGES = True
    COLUMN_EDGES = True
    SUBSTR_EDGES = True
    SUPSTR_EDGES = True
    INDEX_NAME_EDGES = False
    COLUMN_NAME_EDGES = False

    REPRESENTOR_NODE = True
    REPRESENTOR_EDGES = True

    def __init__(self, label: str, df: pd.DataFrame):
        super().__init__(label, df)
        self.df = df

        self.index_nodes: List[List[GraphNode]] = []  # Shape : num_rows x num_levels (highest level first)
        self.column_nodes: List[List[GraphNode]] = []  # Shape : num_cols x num_levels (highest level first)
        self.cell_nodes: List[List[GraphNode]] = []  # Row-major
        self.index_name_nodes: List[GraphNode] = []  # Shape : num_levels (highest level first)
        self.column_name_nodes: List[GraphNode] = []  # Shape : num_levels (highest level first)

        self.build()

    def create_node(self, value: Any, label: str, features: Set[NodeFeatures]):
        node = GraphNode(self.label + ":" + label, features)
        self.nodes.append(node)
        self.val_node_map[value].append(node)
        return node

    def get_index_node(self, val, level: int, idx: int, num_levels: int):
        label = '[{},{}]'.format(idx, level - num_levels)
        node = self.create_node(val, label, {NodeRoles.INDEX, NodeDataTypes.from_value(val)})
        return node

    def get_column_node(self, val, level: int, idx: int, num_levels: int):
        label = '[{},{}]'.format(level - num_levels, idx)
        node = self.create_node(val, label, {NodeRoles.COLUMN, NodeDataTypes.from_value(val)})
        return node

    def get_index_name_node(self, val, level: int, num_levels: int):
        label = '[{},{}]'.format(-1, level - num_levels)
        node = self.create_node(val, label, {NodeRoles.INDEX_NAME, NodeDataTypes.from_value(val)})
        return node

    def get_column_name_node(self, val, level: int, num_levels: int):
        label = '[{},{}]'.format(level - num_levels, -1)
        node = self.create_node(val, label, {NodeRoles.COLUMN_NAME, NodeDataTypes.from_value(val)})
        return node

    def add_index_nodes(self, index: pd.Index, mode='df.index'):
        if isinstance(index, pd.MultiIndex):
            index_nodes = []

            for idx, vals in enumerate(index):
                index_nodes.append([])
                for level, val in enumerate(vals):
                    if mode == 'df.index':
                        node = self.get_index_node(val, level=level, idx=idx, num_levels=index.nlevels)
                    else:
                        node = self.get_column_node(val, level=level, idx=idx, num_levels=index.nlevels)

                    index_nodes[-1].append(node)

            return index_nodes

        else:
            index_nodes = []
            for idx, val in enumerate(index):
                if mode == 'df.index':
                    node = self.get_index_node(val, level=0, idx=idx, num_levels=1)
                else:
                    node = self.get_column_node(val, level=0, idx=idx, num_levels=1)

                index_nodes.append(node)

            return np.transpose([index_nodes]).tolist()

    def add_nodes(self):
        if self.CELL_NODES:
            cells = self.df.values
            for r_idx, row in enumerate(cells):
                self.cell_nodes.append([])
                for c_idx, val in enumerate(row):
                    node = self.create_node(val, label='[{},{}]'.format(r_idx, c_idx),
                                            features={NodeDataTypes.from_value(val)})
                    self.cell_nodes[-1].append(node)

        if self.INDEX_NODES:
            self.index_nodes = self.add_index_nodes(self.df.index, mode='df.index')

        if self.COLUMN_NODES:
            self.column_nodes = self.add_index_nodes(self.df.columns, mode='df.columns')

        if self.INDEX_NAME_NODES:
            num_index_levels = len(self.df.index.names)
            for level, name in enumerate(self.df.index.names):
                if name is None:
                    continue

                node = self.get_index_name_node(name, level=level, num_levels=num_index_levels)
                self.index_name_nodes.append(node)

        if self.COLUMN_NAME_NODES:
            num_column_levels = len(self.df.columns.names)
            for level, name in enumerate(self.df.columns.names):
                if name is None:
                    continue

                node = self.get_column_name_node(name, level=level, num_levels=num_column_levels)
                self.column_name_nodes.append(node)

    def add_internal_edges(self):
        pass

    def build(self):
        self.add_nodes()
        self.add_internal_edges()
