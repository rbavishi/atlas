import collections
import itertools
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Set, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from atlas.synthesis.pandas.checker import Checker
from atlas.operators import unpack_sid, OpInfo, OpResolvable, find_known_operators, resolve_operator, operator


class NodeFeatures(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


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


class NodeSources(NodeFeatures):
    INPUT = auto()
    OUTPUT = auto()
    DOMAIN = auto()


class EdgeTypes(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    #  Naming Convention : Name of edge describes the role of src for dst

    ADJ_LEFT = auto()
    ADJ_RIGHT = auto()
    ADJ_ABOVE = auto()
    ADJ_BELOW = auto()
    EQUALITY = auto()
    INNER_EQUALITY = auto()

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

    def __repr__(self):
        return self.label


class GraphEdge:
    def __init__(self, src: GraphNode, dst: GraphNode, etype: EdgeTypes):
        self.src = src
        self.dst = dst
        self.etype = etype

    def __repr__(self):
        return f"({self.src}, {self.etype.value}, {self.dst})"


class ValueEncoding(ABC):
    def __init__(self, label: str, val: Any):
        self.label = label
        self.val = val
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []

        self.val_node_map: Dict[Any, List[GraphNode]] = collections.defaultdict(list)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def get_representor_node(self):
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

    def get_representor_node(self):
        return self.nodes[-1]


class DataFrameEncoding(ValueEncoding):
    CELL_NODES = True
    INDEX_NODES = True
    COLUMN_NODES = True
    INDEX_NAME_NODES = False
    COLUMN_NAME_NODES = False

    INNER_EQUALITY_EDGES = False
    ADJACENCY_EDGES = True
    INDEX_EDGES = True
    COLUMN_EDGES = True
    INDEX_NAME_EDGES = False
    COLUMN_NAME_EDGES = False
    REPRESENTOR_EDGES = True

    def __init__(self, label: str, df: pd.DataFrame):
        super().__init__(label, df)
        self.df = df

        self.index_nodes: List[List[GraphNode]] = []  # Shape : num_rows x num_levels (highest level first)
        self.column_nodes: List[List[GraphNode]] = []  # Shape : num_cols x num_levels (highest level first)
        self.cell_nodes: List[List[GraphNode]] = []  # Row-major
        self.index_name_nodes: List[GraphNode] = []  # Shape : num_levels (highest level first)
        self.column_name_nodes: List[GraphNode] = []  # Shape : num_levels (highest level first)
        self.representor_node: Optional[GraphNode] = None

    def create_node(self, value: Any, label: str, features: Set[NodeFeatures]):
        node = GraphNode(self.label + ":" + label, features)
        self.nodes.append(node)
        self.val_node_map[value].append(node)
        return node

    def create_edge(self, src: GraphNode, dst: GraphNode, etype: EdgeTypes):
        self.edges.append(GraphEdge(src, dst, etype))

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
        if self.INDEX_EDGES:
            for index_nodes, cell_nodes in zip(self.index_nodes, self.cell_nodes):
                for n1, n2 in itertools.product(index_nodes, cell_nodes):
                    self.create_edge(n1, n2, EdgeTypes.INDEX)
                    self.create_edge(n2, n1, EdgeTypes.INDEX_FOR)

        if self.COLUMN_EDGES:
            for col_nodes, cell_nodes in zip(self.column_nodes, np.transpose(self.cell_nodes)):
                for n1, n2 in itertools.product(col_nodes, cell_nodes):
                    self.create_edge(n1, n2, EdgeTypes.INDEX)
                    self.create_edge(n2, n1, EdgeTypes.INDEX_FOR)

        if self.INDEX_NAME_EDGES:
            for index_name_node, index_nodes in zip(self.index_name_nodes, np.transpose(self.index_nodes)):
                for n in index_nodes:
                    self.create_edge(index_name_node, n, EdgeTypes.INDEX_NAME)
                    self.create_edge(n, index_name_node, EdgeTypes.INDEX_NAME_FOR)

        if self.COLUMN_NAME_EDGES:
            for col_name_node, col_nodes in zip(self.column_name_nodes, np.transpose(self.column_nodes)):
                for n in col_nodes:
                    self.create_edge(col_name_node, n, EdgeTypes.INDEX_NAME)
                    self.create_edge(n, col_name_node, EdgeTypes.INDEX_NAME_FOR)

        if self.ADJACENCY_EDGES:
            def adjacency_to_the_right(vals):
                for row_vals in vals:
                    for n1, n2 in zip(row_vals, row_vals[1:]):
                        self.create_edge(n1, n2, EdgeTypes.ADJ_RIGHT)
                        self.create_edge(n2, n1, EdgeTypes.ADJ_LEFT)

            def adjacency_below(vals):
                for col_vals in np.transpose(vals):
                    for n1, n2 in zip(col_vals, col_vals[1:]):
                        self.create_edge(n1, n2, EdgeTypes.ADJ_BELOW)
                        self.create_edge(n2, n1, EdgeTypes.ADJ_ABOVE)

            for node_set in [self.index_nodes, np.transpose(self.column_nodes), self.cell_nodes]:
                adjacency_to_the_right(node_set)
                adjacency_below(node_set)

        if self.INNER_EQUALITY_EDGES:
            for val, nodes in self.val_node_map.items():
                for n1, n2 in itertools.combinations(nodes, 2):
                    self.create_edge(n1, n2, EdgeTypes.INNER_EQUALITY)
                    self.create_edge(n2, n1, EdgeTypes.INNER_EQUALITY)

    def build(self):
        self.add_nodes()
        self.add_internal_edges()
        self.representor_node = GraphNode(self.label, {NodeRoles.REPRESENTOR})
        self.nodes.append(self.representor_node)

    def get_representor_node(self):
        return self.representor_node


class ValueCollection:
    EQUALITY_EDGES = True
    SUBSTR_EDGES = True
    SUPSTR_EDGES = True

    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []

        self.value_encodings: List[ValueEncoding] = []

    def add_external_edges(self, v1: ValueEncoding, v2: ValueEncoding):
        if self.EQUALITY_EDGES:
            for val1, nodes1 in v1.val_node_map.items():
                if val1 in v2.val_node_map:
                    val2 = val1
                    nodes2 = v2.val_node_map[val2]
                    try:
                        #  This can fail for NaNs etc.
                        if val1 == val2:
                            for n1, n2 in itertools.product(nodes1, nodes2):
                                self.edges.append(GraphEdge(n1, n2, EdgeTypes.EQUALITY))
                                self.edges.append(GraphEdge(n2, n1, EdgeTypes.EQUALITY))

                    except Exception as e:
                        print(f"Error comparing {val1} and {val2}", file=sys.stderr)
                        logging.exception(e)

        if self.SUBSTR_EDGES or self.SUPSTR_EDGES:
            pass

    def add_value_encoding(self, val_encoding: ValueEncoding):
        for v in self.value_encodings:
            self.add_external_edges(v, val_encoding)

        self.nodes.extend(val_encoding.nodes)
        self.edges.extend(val_encoding.edges)
        self.value_encodings.append(val_encoding)

    def to_dict(self) -> Tuple[Dict, Dict[GraphNode, int]]:
        node_to_int: Dict[GraphNode, int] = {n: idx for idx, n in enumerate(self.nodes)}
        nodes = [[f.value for f in n.features] for n in self.nodes]
        edges = [[node_to_int[e.src],
                  e.etype.value,
                  node_to_int[e.dst]] for e in self.edges]

        return {'edges': edges, 'nodes': nodes}, node_to_int


class PandasGraphEncoder(OpResolvable):
    def __init__(self):
        self.edge_type_mapping = {}
        self.node_feature_mapping = {}

        for idx, etype in enumerate(EdgeTypes.__members__.keys()):
            self.edge_type_mapping[etype] = idx

        for idx, feature in enumerate({**NodeDataTypes.__members__,
                                       **NodeRoles.__members__, **NodeSources.__members__}.keys()):
            self.node_feature_mapping[feature] = idx

        self.encoder_definitions = find_known_operators(self)

    def get_num_edge_types(self):
        return len(self.edge_type_mapping)

    def get_num_node_features(self):
        return len(self.node_feature_mapping)

    def convert_edge_type(self, etype: str) -> int:
        if etype not in self.edge_type_mapping:
            raise KeyError(f"Could not map edge type {etype}")

        return self.edge_type_mapping[etype]

    def convert_node_features(self, features: List[str]) -> List[int]:
        for feature in features:
            if feature not in self.node_feature_mapping:
                raise KeyError(f"Could not map node feature {feature}")

        return [self.node_feature_mapping[f] for f in features]

    def encode_value(self, label: str, val: Any) -> ValueEncoding:
        if np.isscalar(val) or val is None:
            return ScalarEncoding(label, val)

        if isinstance(val, pd.DataFrame):
            return DataFrameEncoding(label, val)

        if isinstance(val, pd.Series):
            return DataFrameEncoding(label, pd.DataFrame(val))

        raise TypeError(f"Cannot encode value {val} of type {type(val)} ")

    def post_process(self, encoding: Dict[str, Any]):
        for e in encoding['edges']:
            e[1] = self.convert_edge_type(e[1])

        encoding['nodes'] = [self.convert_node_features(n) for n in encoding['nodes']]

    def get_encoder(self, op_info: OpInfo):
        return resolve_operator(self.encoder_definitions, op_info)

    @operator
    def Select(self, domain, context=None, choice=None, mode='training', **kwargs):
        #  We expect domain to be a list of values
        #  We expect context to be a dictionary with keys as labels and values as the raw values
        #  For example {'I0': Input DataFrame, 'O': Output DataFrame}
        if context is None:
            context = {}

        encoded_domain: Dict[str, ValueEncoding] = {f"D{idx}": self.encode_value(f"D{idx}", v)
                                                    for idx, v in enumerate(domain)}
        encoded_context: Dict[str, ValueEncoding] = {k: self.encode_value(k, v) for k, v in context.items()}

        val_collection: ValueCollection = ValueCollection()
        for k, v in encoded_domain.items():
            v.build()
            val_collection.add_value_encoding(v)
        for k, v in encoded_context.items():
            v.build()
            val_collection.add_value_encoding(v)

        for node in val_collection.nodes:
            if node.label.startswith("I"):
                node.features.add(NodeSources.INPUT)
            elif node.label.startswith("O"):
                node.features.add(NodeSources.OUTPUT)
            elif node.label.startswith("D"):
                node.features.add(NodeSources.DOMAIN)

        encoding, node_to_int = val_collection.to_dict()
        encoding['domain'] = [node_to_int[v.get_representor_node()]
                              for v in encoded_domain.values()]
        if mode == 'training':
            for k, v in encoded_domain.items():
                if Checker.check(v.val, choice):
                    encoding['choice'] = node_to_int[v.get_representor_node()]
                    break
            else:
                raise ValueError(f"Passed choice {choice} could not be found in domain {domain}")

        else:
            encoding['mapping'] = {node_to_int[encoded_domain[f"D{idx}"].get_representor_node()]: v
                                   for idx, v in enumerate(domain)}

        self.post_process(encoding)
        return encoding

    @operator
    def SelectFixed(self, domain, context=None, choice=None, mode='training', **kwargs):
        #  We expect domain to be a list of values
        #  We expect context to be a dictionary with keys as labels and values as the raw values
        #  For example {'I0': Input DataFrame, 'O': Output DataFrame}
        if context is None:
            context = {}

        encoded_context: Dict[str, ValueEncoding] = {k: self.encode_value(k, v) for k, v in context.items()}

        val_collection: ValueCollection = ValueCollection()
        for k, v in encoded_context.items():
            v.build()
            val_collection.add_value_encoding(v)

        for node in val_collection.nodes:
            if node.label.startswith("I"):
                node.features.add(NodeSources.INPUT)
            elif node.label.startswith("O"):
                node.features.add(NodeSources.OUTPUT)
            elif node.label.startswith("D"):
                node.features.add(NodeSources.DOMAIN)

        encoding, node_to_int = val_collection.to_dict()
        if mode == 'training':
            encoding['choice'] = domain.index(choice)

        else:
            encoding['mapping'] = {idx: v for idx, v in enumerate(domain)}

        self.post_process(encoding)
        return encoding

    @operator
    def Subset(self, domain, context=None, choice=None, mode='training', **kwargs):
        #  We expect domain to be a list of values
        #  We expect context to be a dictionary with keys as labels and values as the raw values
        #  For example {'I0': Input DataFrame, 'O': Output DataFrame}
        if context is None:
            context = {}

        encoded_domain: Dict[str, ValueEncoding] = {f"D{idx}": self.encode_value(f"D{idx}", v)
                                                    for idx, v in enumerate(domain)}
        encoded_context: Dict[str, ValueEncoding] = {k: self.encode_value(k, v) for k, v in context.items()}

        val_collection: ValueCollection = ValueCollection()
        for k, v in encoded_domain.items():
            v.build()
            val_collection.add_value_encoding(v)
        for k, v in encoded_context.items():
            v.build()
            val_collection.add_value_encoding(v)

        for node in val_collection.nodes:
            if node.label.startswith("I"):
                node.features.add(NodeSources.INPUT)
            elif node.label.startswith("O"):
                node.features.add(NodeSources.OUTPUT)
            elif node.label.startswith("D"):
                node.features.add(NodeSources.DOMAIN)

        encoding, node_to_int = val_collection.to_dict()
        encoding['domain'] = [node_to_int[v.get_representor_node()]
                              for v in encoded_domain.values()]
        if mode == 'training':
            encoding['choice'] = []
            for c in choice:
                for k, v in encoded_domain.items():
                    if Checker.check(v.val, c):
                        encoding['choice'].append(node_to_int[v.get_representor_node()])
                        break
                else:
                    raise ValueError(f"Passed choice element {c} could not be found in domain {domain}")

        else:
            encoding['mapping'] = {node_to_int[encoded_domain[f"D{idx}"].get_representor_node()]: v
                                   for idx, v in enumerate(domain)}

        self.post_process(encoding)
        return encoding

    @operator
    def OrderedSubset(self, domain, context=None, choice=None, mode='training', **kwargs):
        #  We expect domain to be a list of values
        #  We expect context to be a dictionary with keys as labels and values as the raw values
        #  For example {'I0': Input DataFrame, 'O': Output DataFrame}
        if context is None:
            context = {}

        encoded_domain: Dict[str, ValueEncoding] = {f"D{idx}": self.encode_value(f"D{idx}", v)
                                                    for idx, v in enumerate(domain)}
        encoded_context: Dict[str, ValueEncoding] = {k: self.encode_value(k, v) for k, v in context.items()}

        val_collection: ValueCollection = ValueCollection()
        for k, v in encoded_domain.items():
            v.build()
            val_collection.add_value_encoding(v)
        for k, v in encoded_context.items():
            v.build()
            val_collection.add_value_encoding(v)

        for node in val_collection.nodes:
            if node.label.startswith("I"):
                node.features.add(NodeSources.INPUT)
            elif node.label.startswith("O"):
                node.features.add(NodeSources.OUTPUT)
            elif node.label.startswith("D"):
                node.features.add(NodeSources.DOMAIN)

        terminal = GraphNode("TERMINAL", {NodeRoles.TERMINAL})
        val_collection.nodes.append(terminal)

        encoding, node_to_int = val_collection.to_dict()
        encoding['domain'] = [node_to_int[v.get_representor_node()]
                              for v in encoded_domain.values()] + [node_to_int[terminal]]
        if mode == 'training':
            encoding['choice'] = []
            for elem in choice:
                for k, v in encoded_domain.items():
                    if Checker.check(v.val, elem):
                        encoding['choice'].append(node_to_int[v.get_representor_node()])
                        break
                else:
                    raise ValueError(f"Element {elem} of passed choice {choice} could not be found in domain {domain}")

            encoding['choice'].append(node_to_int[terminal])
            encoding['terminal'] = node_to_int[terminal]

        else:
            encoding['mapping'] = {node_to_int[encoded_domain[f"D{idx}"].get_representor_node()]: v
                                   for idx, v in enumerate(domain)}
            encoding['terminal'] = node_to_int[terminal]

        self.post_process(encoding)
        return encoding

    @operator(name='Sequence', tags=['function_sequence_prediction'])
    def Sequence(self, domain, context=None, choice=None, mode='training', **kwargs):
        #  We expect domain to be a list of values
        #  We expect context to be a dictionary with keys as labels and values as the raw values
        #  For example {'I0': Input DataFrame, 'O': Output DataFrame}
        if context is None:
            context = {}

        encoded_context: Dict[str, ValueEncoding] = {k: self.encode_value(k, v) for k, v in context.items()}

        val_collection: ValueCollection = ValueCollection()
        for k, v in encoded_context.items():
            v.build()
            val_collection.add_value_encoding(v)

        for node in val_collection.nodes:
            if node.label.startswith("I"):
                node.features.add(NodeSources.INPUT)
            elif node.label.startswith("O"):
                node.features.add(NodeSources.OUTPUT)
            elif node.label.startswith("D"):
                node.features.add(NodeSources.DOMAIN)

        encoding, node_to_int = val_collection.to_dict()
        if mode == 'training':
            encoding['choice'] = [domain.index(c) for c in choice]

        else:
            encoding['mapping'] = {idx: v for idx, v in enumerate(domain)}

        self.post_process(encoding)
        return encoding
