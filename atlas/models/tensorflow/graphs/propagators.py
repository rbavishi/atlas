import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple

from atlas.models.tensorflow.graphs.gnn import GNNComponent
from atlas.models.tensorflow.graphs.utils import SegmentBasedSoftmax


class GGNNPropagator(GNNComponent):
    """The original message passing logic as described in
    https://github.com/microsoft/gated-graph-neural-network-samples/blob/master/chem_tensorflow_sparse.py"""

    def __init__(self,
                 layer_timesteps: List[int],
                 node_dimension: int,
                 num_edge_types: int,
                 residual_connections: Dict[int, List[int]] = None,
                 graph_rnn_cell: str = 'gru',
                 graph_rnn_activation: str = 'tanh',
                 use_propagation_attention: bool = True,
                 edge_msg_aggregation: str = 'avg',
                 graph_state_dropout: float = 0.0,  # The rate of dropout, i.e. 1-the probability of 'keeping' it.
                 edge_weight_dropout: float = 0.2,  # The rate of dropout, i.e. 1-the probability of 'keeping' it.
                 name: str = 'propagator',
                 **kwargs):

        super().__init__()
        self.layer_timesteps = layer_timesteps
        self.node_dimension = node_dimension
        self.num_edge_types = num_edge_types
        self.redidual_connections = residual_connections or {}
        self.graph_rnn_cell = graph_rnn_cell
        self.graph_rnn_activation = graph_rnn_activation
        self.use_propagation_attention = use_propagation_attention
        self.edge_msg_aggregation = edge_msg_aggregation
        self.graph_state_dropout = graph_state_dropout
        self.edge_weight_dropout = edge_weight_dropout

        self.name = name

        if self.graph_rnn_cell not in ['gru']:
            raise ValueError("Graph RNN Cell must be one of {'gru'}")

        if self.graph_rnn_activation not in ['tanh', 'relu']:
            raise ValueError("Graph RNN Activation must be one of {'tanh', 'relu'}")

        if self.edge_msg_aggregation not in ['avg', 'sum']:
            raise ValueError("Edge Message aggregation type should be one of {'avg', 'sum'}")

    def get_adjacency_list(self, edges: List[Tuple[int, int, int]]):
        adj_lists: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_edge_types)]
        for src, e_type, dst in edges:
            adj_lists[e_type].append((src, dst))

        adj_lists = [np.array(sorted(l), dtype=np.int32) if len(l) > 0 else np.zeros((0, 2), dtype=np.int32)
                     for l in adj_lists]

        return adj_lists

    def construct_node_embedding(self, features: List[int]):
        embedding = [0] * self.node_dimension
        for i in features:
            embedding[i] = 1

        return embedding

    def define_batch(self, graphs: List[Dict], is_training: bool = True) -> Dict:
        node_offset = 0
        adjacency_lists = [[] for _ in range(self.num_edge_types)]
        initial_node_embeddings = []
        for idx, g in enumerate(graphs):
            adj_lists = [a + node_offset for a in self.get_adjacency_list(g['edges'])]
            node_embeddings = [self.construct_node_embedding(f) for f in g['nodes']]

            for e_type, adj_list in enumerate(adj_lists):
                adjacency_lists[e_type].extend(adj_list)

            initial_node_embeddings.extend(node_embeddings)
            node_offset += len(g['nodes'])

        result = {
            self.placeholders['initial_node_embedding']: np.array(initial_node_embeddings),
            self.placeholders['num_nodes']: node_offset,

            self.placeholders['graph_state_dropout']: self.graph_state_dropout if is_training else 0.0,
            self.placeholders['edge_weight_dropout']: self.edge_weight_dropout if is_training else 0.0,
        }

        for idx, placeholder in enumerate(self.placeholders['adjacency_lists']):
            if len(adjacency_lists[idx]) > 0:

                result[placeholder] = adjacency_lists[idx]
            else:
                result[placeholder] = np.zeros((0, 2), dtype=np.int32)

        return result

    def build(self):
        """
        Constructs the tensorflow computation by unrolling the graph for the time-steps
        indicated by ``self.layer_timesteps``
        """
        self.define_placeholders()
        self.define_weights()
        self.define_message_passing()

    def define_placeholders(self):
        #  Is normally a one-hot encoding of the feature of a node, but in principle can be any arbitrary vector
        self.placeholders['initial_node_embedding'] = tf.placeholder(tf.float32, [None, self.node_dimension],
                                                                     name='node_embeddings')
        #  We have a separate adjacency matrix for every edge type
        self.placeholders['adjacency_lists'] = [tf.placeholder(tf.int32, [None, 2], name='adjacency_e{}'.format(e))
                                                for e in range(self.num_edge_types)]

        #  Need to create placeholders for these as dropout should not be used during inference
        self.placeholders['graph_state_dropout'] = tf.placeholder(tf.float32, None, name='graph_state_dropout')
        self.placeholders['edge_weight_dropout'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout')

        #  Needed for various gather/segment-sum operations
        self.placeholders['num_nodes'] = tf.placeholder(tf.int32, shape=(), name='num_nodes')

    def define_weights(self):
        #  We have separate edge weights, and RNN cells for each time-step
        all_edge_weights = []
        all_rnn_cells = []
        edge_type_attention_weights = []

        for l in range(len(self.layer_timesteps)):
            with tf.variable_scope("{name}_layer_{layer}".format(name=self.name, layer=l)):
                #  Edge weights are num_edge_types matrices of dimensions [node_dimension x node_dimension]
                #  The shape passed to get_variable has only two dimensions due to the fan-in/fan-out structure
                #  imposed by the ``glorot_uniform initializer``. So we have a reshaping wrapper around that
                edge_weights = tf.reshape(tf.get_variable("edge_weights_{layer}".format(layer=l),
                                                          [self.num_edge_types * self.node_dimension,
                                                           self.node_dimension],
                                                          initializer=tf.glorot_uniform_initializer()),
                                          [self.num_edge_types, self.node_dimension, self.node_dimension])

                edge_weights = tf.nn.dropout(edge_weights, rate=self.placeholders['edge_weight_dropout'])
                all_edge_weights.append(edge_weights)

                if self.graph_rnn_cell == 'gru':
                    cell = tf.keras.layers.GRUCell(self.node_dimension, activation=self.graph_rnn_activation)

                else:
                    raise ValueError("Cell type should be one of {'gru'}")

                #  Very simplistic attention parameter. A scalar value for each edge type for every layer
                if self.use_propagation_attention:
                    attn = tf.get_variable("edge_type_attn_{layer}".format(layer=l),
                                           shape=[self.num_edge_types],
                                           dtype=tf.float32,
                                           initializer=tf.initializers.constant(value=np.ones([self.num_edge_types])))
                    edge_type_attention_weights.append(attn)

            all_rnn_cells.append(cell)

        self.weights['edge_weights'] = all_edge_weights
        self.weights['rnn_cells'] = all_rnn_cells

        if self.use_propagation_attention:
            self.weights['edge_type_attention_weights'] = edge_type_attention_weights

    def define_message_passing(self):
        node_embeddings_per_round = [self.placeholders['initial_node_embedding']]

        for layer, num_time_steps in enumerate(self.layer_timesteps):
            for step in range(num_time_steps):
                current_node_embeddings = node_embeddings_per_round[-1]
                node_embeddings_per_round.append(self.define_round(layer, step, current_node_embeddings))

        self.ops['node_embeddings_per_round'] = node_embeddings_per_round
        self.ops['final_node_embeddings'] = node_embeddings_per_round[-1]

    def define_round(self, layer: int, time_step: int, node_embeddings):
        node_embeddings = tf.identity(node_embeddings)
        edge_weights = self.weights['edge_weights'][layer]
        src_node_ids, dst_node_ids, src_node_embeddings, dst_node_embeddings, messages = [], [], [], [], []

        for e_type, adj_list in enumerate(self.placeholders['adjacency_lists']):
            src_node_ids.append(adj_list[:, 0])
            dst_node_ids.append(adj_list[:, 1])

            src_node_embeddings.append(tf.nn.embedding_lookup(params=node_embeddings, ids=src_node_ids[-1]))
            dst_node_embeddings.append(tf.nn.embedding_lookup(params=node_embeddings, ids=dst_node_ids[-1]))

            messages.append(tf.matmul(src_node_embeddings[-1], edge_weights[e_type]))

        src_node_ids = tf.concat(src_node_ids, axis=0)
        src_node_embeddings = tf.concat(src_node_embeddings, axis=0)
        dst_node_ids = tf.concat(dst_node_ids, axis=0)
        dst_node_embeddings = tf.concat(dst_node_embeddings, axis=0)
        messages = tf.concat(messages, axis=0)

        #  Now weigh the messages using attention if configured to do so
        if self.use_propagation_attention:
            messages *= tf.expand_dims(self.define_message_attention(layer, src_node_ids, src_node_embeddings,
                                                                     dst_node_ids, dst_node_embeddings, messages), axis=-1)

        #  Accumulate all messages for a destination node
        if self.edge_msg_aggregation == 'avg':
            incoming_messages = tf.unsorted_segment_mean(data=messages,
                                                         segment_ids=dst_node_ids,
                                                         num_segments=self.placeholders['num_nodes'])
        elif self.edge_msg_aggregation == 'sum':
            incoming_messages = tf.unsorted_segment_sum(data=messages,
                                                        segment_ids=dst_node_ids,
                                                        num_segments=self.placeholders['num_nodes'])
        else:
            raise ValueError("Edge message aggregation type should be one of {'avg', 'sum'}")

        #  Compute new node states i.e. states for the next round of message passing (if any)
        return self.weights['rnn_cells'][layer](incoming_messages, [node_embeddings])[0]

    def define_message_attention(self, layer: int, src_node_ids, src_node_embeddings,
                                 dst_node_ids, dst_node_embeddings, messages):

        message_edge_types = []
        for e_type, adj_list in enumerate(self.placeholders['adjacency_lists']):
            message_edge_types.append(tf.ones_like(adj_list[:, 1], dtype=tf.int32) * e_type)

        message_edge_types = tf.concat(message_edge_types, axis=0)
        edge_attn_factors = tf.nn.embedding_lookup(params=self.weights['edge_type_attention_weights'][layer],
                                                   ids=message_edge_types)

        #  Basically the dot-product of src states and dst states
        msg_attention_scores = tf.einsum('mi,mi->m', src_node_embeddings, dst_node_embeddings)
        #  Multiply by the edge attention (a scalar value based on the edge-type)
        msg_attention_scores *= edge_attn_factors

        #  Since the number of targets is dynamic, can't use native tf.softmax-like operations, so need to implement
        #  the logexpsum trick manually
        return SegmentBasedSoftmax(data=msg_attention_scores, segment_ids=dst_node_ids,
                                   num_segments=self.placeholders['num_nodes'])
