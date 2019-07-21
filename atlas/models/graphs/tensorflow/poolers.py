"""
A Pooler takes all node embeddings and defines a pooling technique to get the final graph-level embeddings
"""

import tensorflow as tf


class AggPooler:
    """Simple aggregation (mean/sum) based pooler"""

    def __init__(self, agg: str = 'sum'):
        self.agg = agg

        if self.agg not in ['sum', 'mean']:
            raise ValueError("Aggregation must be one of {'sum', 'mean'}")

        self.placeholders = {}

    def define_placeholders(self):
        self.placeholders['node_graph_ids_list'] = tf.placeholder(tf.int32, [None], name='node_graph_ids_list')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, shape=(), name='num_graphs')

    def build(self, node_embeddings):
        self.define_placeholders()

        if self.agg == 'sum':
            return tf.unsorted_segment_sum(data=node_embeddings,
                                           segment_ids=self.placeholders['node_graph_ids_list'],
                                           num_segments=self.placeholders['num_graphs'])
        elif self.agg == 'mean':
            return tf.unsorted_segment_sum(data=node_embeddings,
                                           segment_ids=self.placeholders['node_graph_ids_list'],
                                           num_segments=self.placeholders['num_graphs'])
        else:
            raise ValueError("Aggregation must be one of {'sum', 'mean'}")
