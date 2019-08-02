import tensorflow as tf
import numpy as np
from typing import Mapping, Any, List, Dict

from atlas.models.tensorflow.graphs.classifiers import GGNNGraphClassifier
from atlas.models.tensorflow.graphs.gnn import GGNN
from atlas.models.tensorflow.graphs.utils import MLP, SegmentBasedSoftmax


class SelectGGNNClassifier(GGNNGraphClassifier):
    def define_batch(self, graphs: List[Dict[str, Any]], is_training: bool = True):
        batch_data = super().define_batch(graphs, is_training)
        batch_data.pop(self.placeholders['labels'])

        node_offset = 0
        domain = []
        domain_labels = []
        domain_node_graph_ids_list = []
        for idx, g in enumerate(graphs):
            domain.extend([i + node_offset for i in g['domain']])
            selected_domain_node = g['choice']
            domain_labels.extend([i == selected_domain_node for i in g['domain']])
            domain_node_graph_ids_list.extend([idx for _ in range(len(g['domain']))])

        batch_data.update({
            self.placeholders['domain']: np.array(domain),
            self.placeholders['domain_labels']: np.array(domain_labels),
            self.placeholders['domain_node_graph_ids_list']: np.array(domain_node_graph_ids_list)
        })

        return batch_data

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['domain'] = tf.placeholder(tf.int32, [None], name='domain')
        self.placeholders['domain_node_graph_ids_list'] = tf.placeholder(tf.int32, [None],
                                                                         name='domain_node_graph_ids_list')
        self.placeholders['domain_labels'] = tf.placeholder(tf.int32, [None], name='domain_labels')

    def define_prediction_with_loss(self, node_embeddings):
        graph_embeddings = self.define_pooling(node_embeddings)
        domain_nodes_embeddings = tf.gather(params=node_embeddings,
                                            indices=self.placeholders['domain'])
        graph_embeddings_copied = tf.gather(params=graph_embeddings,
                                            indices=self.placeholders['domain_node_graph_ids_list'])

        domain_node_score_calculator = MLP(
            in_size=domain_nodes_embeddings.get_shape()[1] + graph_embeddings_copied.get_shape()[1],
            out_size=1, hid_sizes=self.classifier_hidden_dims)

        domain_node_logits = domain_node_score_calculator(tf.concat([domain_nodes_embeddings,
                                                                     graph_embeddings_copied], axis=-1))
        domain_node_logits = tf.reshape(domain_node_logits, [-1])
        probs, log_probs = SegmentBasedSoftmax(data=domain_node_logits,
                                               segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                               num_segments=self.placeholders['num_graphs'],
                                               return_log=True)

        loss_per_domain_node = -tf.cast(self.placeholders['domain_labels'], tf.float32) * log_probs
        loss_per_graph = tf.unsorted_segment_sum(data=loss_per_domain_node,
                                                 segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                                 num_segments=self.placeholders['num_graphs'])
        self.ops['loss'] = tf.reduce_mean(loss_per_graph)

        domain_node_max_scores = tf.unsorted_segment_max(data=domain_node_logits,
                                                         segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                                         num_segments=self.placeholders['num_graphs'])
        copied_domain_node_max_scores = tf.gather(params=domain_node_max_scores,
                                                  indices=self.placeholders['domain_node_graph_ids_list'])

        selected_domain_nodes = tf.cast(tf.equal(copied_domain_node_max_scores, domain_node_logits), dtype=tf.float32)
        correct_prediction_per_node = selected_domain_nodes * tf.cast(self.placeholders['domain_labels'], tf.float32)
        correct_prediction = tf.unsorted_segment_max(data=correct_prediction_per_node,
                                                     segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                                     num_segments=self.placeholders['num_graphs'])

        self.ops['accuracy'] = tf.reduce_mean(correct_prediction)


class SelectGGNN(GGNN):
    def __init__(self, params: Mapping[str, Any], propagator=None, classifier=None, optimizer=None):
        if classifier is None:
            classifier = SelectGGNNClassifier(**params)

        super().__init__(params, propagator, classifier, optimizer)
