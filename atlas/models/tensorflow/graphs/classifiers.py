"""
A Classifier takes all node embeddings and defines the classifier computation.
The simplest one simply pools the node embeddings into a graph-level embedding
and uses that to do single-label classification.
"""

import tensorflow as tf
import numpy as np
from typing import List, Any, Dict

from atlas.models.tensorflow.graphs.gnn import GNNComponent
from atlas.models.tensorflow.graphs.utils import MLP


class GGNNGraphClassifier(GNNComponent):
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

    def define_batch(self, graphs: List[Dict[str, Any]], is_training: bool = True):
        labels = []
        node_graph_ids_list = []
        for idx, g in enumerate(graphs):
            label = g.get('label', 0)  # Can happen when doing inference
            labels.append(label)
            node_graph_ids_list.extend([idx for _ in range(len(g['nodes']))])

        return {
            self.placeholders['node_graph_ids_list']: np.array(node_graph_ids_list),
            self.placeholders['num_graphs']: len(graphs),
            self.placeholders['labels']: np.array(labels)
        }

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


class GGNNGraphSequentialClassifier(GNNComponent):
    """
    Like a graph classifier, but instead produces a sequence as output.
    The sequence can only consist of elements from a pre-determined set of elements.
    """

    def __init__(self,
                 num_classes: int,
                 max_length: int,
                 classifier_hidden_dims: List[int],
                 agg: str = 'sum',
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.max_length = max_length
        self.classifier_hidden_dims = classifier_hidden_dims
        self.agg = agg

        if self.agg not in ['sum', 'mean']:
            raise ValueError("Aggregation must be one of {'sum', 'mean'}")

    def define_batch(self, graphs: List[Dict[str, Any]], is_training: bool = True):
        labels = []
        loss_masks = []
        acc_masks = []
        node_graph_ids_list = []
        for idx, g in enumerate(graphs):
            label = g.get('label', [])[:]  # Can happen when doing inference
            loss_masks.append([1 for _ in range(len(label) + 1)] + [0 for _ in range(self.max_length - len(label))])
            acc_masks.append([0 for _ in range(len(label) + 1)] + [1 for _ in range(self.max_length - len(label))])

            if len(label) < self.max_length + 1:
                #  Max-length does not include the terminal token
                label.extend([self.num_classes for _ in range(self.max_length + 1 - len(label))])

            labels.append(label)
            node_graph_ids_list.extend([idx for _ in range(len(g['nodes']))])

        return {
            self.placeholders['node_graph_ids_list']: np.array(node_graph_ids_list),
            self.placeholders['num_graphs']: len(graphs),
            self.placeholders['labels']: np.array(labels),
            self.placeholders['loss_masks']: np.array(loss_masks),
            self.placeholders['acc_masks']: np.array(acc_masks)
        }

    def build(self, node_embeddings):
        self.define_placeholders()
        self.define_prediction_with_loss(node_embeddings)

    def define_placeholders(self):
        self.placeholders['node_graph_ids_list'] = tf.placeholder(tf.int32, [None], name='node_graph_ids_list')
        self.placeholders['num_graphs'] = tf.placeholder(tf.int32, shape=(), name='num_graphs')
        self.placeholders['labels'] = tf.placeholder(tf.int32, [None, None], name='labels')
        self.placeholders['loss_masks'] = tf.placeholder(tf.float32, [None, None], name='loss_mask')
        self.placeholders['acc_masks'] = tf.placeholder(tf.float32, [None, None], name='acc_mask')

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

    def define_prediction_with_loss(self, node_embeddings):
        #  Shape of graph_embeddings is (batch_size, H)
        graph_embeddings = self.define_pooling(node_embeddings)

        #  Now the shape is (batch_size, max_length, H)
        graph_embeddings_tiled = tf.tile(tf.expand_dims(graph_embeddings, 1), [1, self.max_length + 1, 1])
        #  Shape is (batch_size, max_length, H)
        rnn_output = tf.keras.layers.LSTM(self.classifier_hidden_dims[0], return_sequences=True)(graph_embeddings_tiled)

        mlp_rnn_output = MLP(
            in_size=rnn_output.get_shape()[-1],
            out_size=self.num_classes + 1, hid_sizes=self.classifier_hidden_dims)

        #  Shape is (batch_size, max_length, num_classes)
        logits_timestep = mlp_rnn_output(rnn_output)
        loss_matrix = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_timestep,
                                                                     labels=self.placeholders['labels'])
        loss_matrix *= self.placeholders['loss_masks']
        self.ops['loss'] = tf.reduce_mean(loss_matrix)
        probabilities = self.ops['probabilities'] = tf.nn.softmax(logits_timestep)

        predictions = tf.argmax(probabilities, -1, output_type=tf.int32)
        correct_prediction = tf.cast(tf.equal(predictions, self.placeholders['labels']), tf.float32)
        correct_prediction = tf.maximum(correct_prediction, self.placeholders['acc_masks'])
        self.ops['accuracy'] = tf.reduce_mean(tf.reduce_prod(correct_prediction, axis=1))
