"""
A Classifier takes all node embeddings and defines the classifier computation.
The simplest one simply pools the node embeddings into a graph-level embedding
and uses that to do single-label classification.
"""

import tensorflow as tf
from typing import List

from atlas.models.graphs.tensorflow.network import NetworkComponent
from atlas.models.graphs.tensorflow.utils import MLP


class GGNNGraphClassifier(NetworkComponent):
    """Simple aggregation (mean/sum) based pooler plus fixed num-classes softmax classifier"""

    def __init__(self,
                 num_classes: int,
                 classifier_hidden_dims: List[int],
                 agg: str = 'sum',
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.classifier_hidden_dims = classifier_hidden_dims
        self.agg = agg

        if self.agg not in ['sum', 'mean']:
            raise ValueError("Aggregation must be one of {'sum', 'mean'}")

    def build(self, node_embeddings):
        self.define_placeholders()
        self.define_prediction_with_loss(node_embeddings)

    def define_placeholders(self):
        self.placeholders['node_graph_ids_list'] = tf.placeholder(tf.int32, [None], name='node_graph_ids_list')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, shape=(), name='num_graphs')
        self.placeholders['labels'] = tf.placeholder(tf.int32, [None], name='labels')

    def define_pooling(self, node_embeddings):
        if self.agg == 'sum':
            return tf.unsorted_segment_sum(data=node_embeddings,
                                           segment_ids=self.placeholders['node_graph_ids_list'],
                                           num_segments=self.placeholders['num_graphs'])
        elif self.agg == 'mean':
            return tf.unsorted_segment_mean(data=node_embeddings,
                                            segment_ids=self.placeholders['node_graph_ids_list'],
                                            num_segments=self.placeholders['num_graphs'])
        else:
            raise ValueError("Aggregation must be one of {'sum', 'mean'}")

    def define_graph_logits(self, graph_embeddings):
        classifier = MLP(in_size=graph_embeddings.get_shape()[1], out_size=self.num_classes,
                         hid_sizes=self.classifier_hidden_dims)
        return classifier(graph_embeddings)

    def define_prediction_with_loss(self, node_embeddings):
        graph_embeddings = self.define_pooling(node_embeddings)
        graph_logits = self.define_graph_logits(graph_embeddings)

        self.ops['loss'] = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=graph_logits,
                                                           labels=self.placeholders['labels'])
        )

        probabilities = self.ops['probabilities'] = tf.nn.softmax(graph_logits)
        correct_prediction = tf.equal(tf.argmax(probabilities, -1, output_type=tf.int32), self.placeholders['labels'])
        self.ops['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))
