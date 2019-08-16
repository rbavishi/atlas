import collections

import tensorflow as tf
import numpy as np
from typing import Mapping, Any, List, Dict, Iterable

from atlas.models.tensorflow.graphs.classifiers import GGNNGraphClassifier
from atlas.models.tensorflow.graphs.ggnn import GGNN
from atlas.models.tensorflow.graphs.utils import MLP, SegmentBasedSoftmax


class SelectGGNNClassifier(GGNNGraphClassifier):
    def __init__(self, classifier_hidden_dims: List[int], agg: str = 'sum', **kwargs):
        super().__init__(num_classes=-1, classifier_hidden_dims=classifier_hidden_dims, agg=agg, **kwargs)

    def define_batch(self, graphs: List[Dict[str, Any]], is_training: bool = True):
        batch_data = super().define_batch(graphs, is_training)
        batch_data.pop(self.placeholders['labels'])

        node_offset = 0
        domain = []
        domain_labels = []
        domain_node_graph_ids_list = []
        for idx, g in enumerate(graphs):
            domain.extend([i + node_offset for i in g['domain']])
            selected_domain_node = g.get('choice', 0)
            domain_labels.extend([int(i == selected_domain_node) for i in g['domain']])
            domain_node_graph_ids_list.extend([idx for _ in range(len(g['domain']))])
            node_offset += len(g['nodes'])

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
        node_embeddings = tf.identity(node_embeddings)
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
        self.ops['probabilities'] = probs

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

        selected_domain_nodes = tf.cast(tf.equal(copied_domain_node_max_scores, domain_node_logits), dtype=tf.int32)
        correct_prediction_per_node = tf.cast(tf.equal(selected_domain_nodes, self.placeholders['domain_labels']),
                                              tf.float32)
        correct_prediction = tf.unsorted_segment_prod(data=correct_prediction_per_node,
                                                      segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                                      num_segments=self.placeholders['num_graphs'])

        self.ops['accuracy'] = tf.reduce_mean(correct_prediction)


class SelectGGNN(GGNN):
    def __init__(self, params: Mapping[str, Any], propagator=None, classifier=None, optimizer=None):
        if classifier is None:
            classifier = SelectGGNNClassifier(**params)

        super().__init__(params, propagator, classifier, optimizer)

    def infer(self, data: Iterable[Dict]):
        num_graphs, batch_data = next(self.get_batch_iterator(iter(data), -1, is_training=False))
        results = self.sess.run([self.classifier.ops['probabilities'],
                                 self.classifier.placeholders['domain_node_graph_ids_list']],
                                feed_dict=batch_data)

        per_graph_results = collections.defaultdict(list)
        for prob, graph_id in zip(*results):
            per_graph_results[graph_id].append(prob)

        inference = []
        for idx, graph in enumerate(iter(data)):
            if 'mapping' in graph:
                mapping = graph['mapping']
                inference.append([(mapping[domain_node], prob)
                                  for domain_node, prob in zip(graph['domain'], per_graph_results[idx])])

        return inference


class SubsetGGNNClassifier(GGNNGraphClassifier):
    def __init__(self, classifier_hidden_dims: List[int], agg: str = 'sum', **kwargs):
        super().__init__(num_classes=-1, classifier_hidden_dims=classifier_hidden_dims, agg=agg, **kwargs)

    def define_batch(self, graphs: List[Dict[str, Any]], is_training: bool = True):
        batch_data = super().define_batch(graphs, is_training)
        batch_data.pop(self.placeholders['labels'])

        node_offset = 0
        domain = []
        domain_labels = []
        domain_node_graph_ids_list = []
        for idx, g in enumerate(graphs):
            domain.extend([i + node_offset for i in g['domain']])
            selected_domain_nodes = g.get('choices', [])
            domain_labels.extend([int(i in selected_domain_nodes) for i in g['domain']])
            domain_node_graph_ids_list.extend([idx for _ in range(len(g['domain']))])
            node_offset += len(g['nodes'])

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
        node_embeddings = tf.identity(node_embeddings)
        graph_embeddings = self.define_pooling(node_embeddings)
        domain_nodes_embeddings = tf.gather(params=node_embeddings,
                                            indices=self.placeholders['domain'])
        graph_embeddings_copied = tf.gather(params=graph_embeddings,
                                            indices=self.placeholders['domain_node_graph_ids_list'])

        mlp_domain_nodes = MLP(
            in_size=domain_nodes_embeddings.get_shape()[1] + graph_embeddings_copied.get_shape()[1],
            out_size=2, hid_sizes=self.classifier_hidden_dims)

        domain_node_logits = mlp_domain_nodes(tf.concat([domain_nodes_embeddings,
                                                         graph_embeddings_copied], axis=-1))
        individual_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=domain_node_logits,
                                                                         labels=self.placeholders['domain_labels'])
        loss = self.ops['loss'] = tf.reduce_mean(individual_loss)
        probs = self.ops['probabilities'] = tf.nn.softmax(domain_node_logits)
        flat_correct_predictions = tf.cast(tf.equal(tf.argmax(probs, -1, output_type=tf.int32),
                                                    self.placeholders['domain_labels']),
                                           tf.float32)
        correct_predictions = tf.unsorted_segment_prod(data=flat_correct_predictions,
                                                       segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                                       num_segments=self.placeholders['num_graphs'])
        self.ops['accuracy'] = tf.reduce_mean(correct_predictions)


class SubsetGGNN(GGNN):
    def __init__(self, params: Mapping[str, Any], propagator=None, classifier=None, optimizer=None):
        if classifier is None:
            classifier = SubsetGGNNClassifier(**params)

        super().__init__(params, propagator, classifier, optimizer)
