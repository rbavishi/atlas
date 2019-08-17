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
            selected_domain_nodes = g.get('choice', [])
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


class OrderedSubsetGGNNClassifier(GGNNGraphClassifier):
    def __init__(self, classifier_hidden_dims: List[int], agg: str = 'sum', **kwargs):
        super().__init__(num_classes=-1, classifier_hidden_dims=classifier_hidden_dims, agg=agg, **kwargs)

    def define_batch(self, graphs: List[Dict[str, Any]], is_training: bool = True):
        batch_data = super().define_batch(graphs, is_training)
        batch_data.pop(self.placeholders['labels'])

        node_offset = 0
        domain = []
        domain_labels = []
        domain_node_graph_ids_list = []
        domain_node_timestep_graph_ids_list = []
        domain_node_timestep_graph_ids_list_unshifted = []
        loss_mask = []
        acc_mask = []
        max_length = max(len(g['domain']) for g in graphs)
        for idx, g in enumerate(graphs):
            domain.extend([i + node_offset for i in g['domain']])
            domain_node_graph_ids_list.extend([idx for _ in range(len(g['domain']))])

            domain_node_timestep_graph_ids_list.extend([[idx * max_length + timestep for timestep in range(max_length)]
                                                        for _ in range(len(g['domain']))])
            domain_node_timestep_graph_ids_list_unshifted.extend([[idx for _ in range(max_length)]
                                                                  for _ in range(len(g['domain']))])

            selected_domain_nodes = g.get('choice', [])
            loss_mask_single = [1] * len(selected_domain_nodes) + [0] * (max_length - len(selected_domain_nodes))
            loss_mask.extend([loss_mask_single for _ in range(len(g['domain']))])
            acc_mask_single = [0] * len(selected_domain_nodes) + [1] * (max_length - len(selected_domain_nodes))
            acc_mask.extend([acc_mask_single for _ in range(len(g['domain']))])
            if len(selected_domain_nodes) < max_length:
                selected_domain_nodes.extend([g['terminal'] for _ in range(max_length - len(selected_domain_nodes))])

            domain_labels.extend([[int(i == selected_domain_nodes[t]) for t in range(max_length)]
                                  for i in g['domain']])
            node_offset += len(g['nodes'])

        batch_data.update({
            self.placeholders['max_length']: max_length,
            self.placeholders['domain']: np.array(domain),
            self.placeholders['domain_labels']: np.array(domain_labels),
            self.placeholders['domain_node_graph_ids_list']: np.array(domain_node_graph_ids_list),
            self.placeholders['loss_mask']: np.array(loss_mask),
            self.placeholders['acc_mask']: np.array(acc_mask),
            self.placeholders['domain_node_timestep_graph_ids_list']: np.array(domain_node_timestep_graph_ids_list),
            self.placeholders['domain_node_timestep_graph_ids_list_unshifted']: np.array(
                domain_node_timestep_graph_ids_list_unshifted),
        })

        return batch_data

    def define_placeholders(self):
        super().define_placeholders()
        self.placeholders['max_length'] = tf.placeholder(tf.int32, None, name="max_length")
        self.placeholders['domain'] = tf.placeholder(tf.int32, [None], name='domain')
        self.placeholders['domain_node_graph_ids_list'] = tf.placeholder(tf.int32, [None],
                                                                         name='domain_node_graph_ids_list')
        self.placeholders['domain_node_timestep_graph_ids_list'] = tf.placeholder(tf.int32, [None, None],
                                                                                  name='domain_node_timestep_graph_ids_list')
        self.placeholders['domain_node_timestep_graph_ids_list_unshifted'] = tf.placeholder(tf.int32, [None, None],
                                                                                            name='domain_node_timestep_graph_ids_list_unshifted')
        self.placeholders['domain_labels'] = tf.placeholder(tf.int32, [None, None], name='domain_labels')
        self.placeholders['loss_mask'] = tf.placeholder(tf.float32, [None, None], name='loss_mask')
        self.placeholders['acc_mask'] = tf.placeholder(tf.float32, [None, None], name='acc_mask')

    def define_prediction_with_loss(self, node_embeddings):
        node_embeddings = tf.identity(node_embeddings)
        graph_embeddings = self.define_pooling(node_embeddings)
        domain_nodes_embeddings = tf.gather(params=node_embeddings,
                                            indices=self.placeholders['domain'])
        domain_nodes_pooled = tf.unsorted_segment_sum(data=domain_nodes_embeddings,
                                                      segment_ids=self.placeholders['domain_node_graph_ids_list'],
                                                      num_segments=self.placeholders['num_graphs'])

        #  Shape is (batch_size, H) for both the tensors below
        graph_embeddings_copied = tf.gather(params=graph_embeddings,
                                            indices=self.placeholders['domain_node_graph_ids_list'])
        domain_nodes_pooled_copied = tf.gather(params=domain_nodes_pooled,
                                               indices=self.placeholders['domain_node_graph_ids_list'])

        #  Shape is now (batch_size, max_length, H) for both the tensors below
        tiling = [1, self.placeholders['max_length'], 1]
        tiled_graph_embeddings_copied = tf.tile(tf.expand_dims(graph_embeddings_copied, 1), tiling)
        tiled_domain_nodes_pooled_copied = tf.tile(tf.expand_dims(domain_nodes_pooled_copied, 1), tiling)

        #  Shape is (num-nodes-in-batch, max_length, H)
        tiled_domain_nodes_embeddings = tf.tile(tf.expand_dims(domain_nodes_embeddings, 1), tiling)

        #  Shape is (batch_size, max_length, 2H)
        rnn_input = tf.concat([tiled_domain_nodes_pooled_copied, tiled_graph_embeddings_copied], axis=-1)
        print(rnn_input.get_shape())
        #  Shape is (batch_size, max_length, 2H)
        rnn_output = tf.keras.layers.LSTM(self.classifier_hidden_dims[0], return_sequences=True)(rnn_input)
        #  Shape is (max_length, batch_size, 2H)
        # rnn_output = tf.transpose(rnn_output, perm=[1, 0, 2])
        #  Shape is (num-nodes-in-batch, max_length, H)
        rnn_output_copied = tf.gather(params=rnn_output, indices=self.placeholders['domain_node_graph_ids_list'])

        mlp_domain_nodes = MLP(
            in_size=rnn_output_copied.get_shape()[-1] + tiled_domain_nodes_embeddings.get_shape()[-1],
            out_size=1, hid_sizes=self.classifier_hidden_dims)

        #  Shape is (num-nodes-in-batch, max_length, 1)
        print(tiled_domain_nodes_embeddings.get_shape())
        domain_node_logits = mlp_domain_nodes(tf.concat([rnn_output_copied,
                                                         tiled_domain_nodes_embeddings], axis=-1))
        #  Shape is (num-nodes-in-batch, max_length)
        domain_node_logits = tf.squeeze(domain_node_logits, [-1])
        #  Shape is (num-nodes-in-batch, max_length) for both
        probs, log_probs = SegmentBasedSoftmax(data=domain_node_logits,
                                               segment_ids=self.placeholders['domain_node_timestep_graph_ids_list'],
                                               num_segments=self.placeholders['num_graphs'] * self.placeholders['max_length'],
                                               return_log=True)
        self.ops['probabilities'] = probs

        loss_per_domain_node = -tf.cast(self.placeholders['domain_labels'], tf.float32) * log_probs
        loss_per_domain_node *= self.placeholders['loss_mask']
        loss_per_graph = tf.unsorted_segment_sum(data=loss_per_domain_node,
                                                 segment_ids=self.placeholders[
                                                     'domain_node_timestep_graph_ids_list_unshifted'],
                                                 num_segments=self.placeholders['num_graphs'])
        self.ops['loss'] = tf.reduce_mean(loss_per_graph)

        domain_node_max_scores = tf.unsorted_segment_max(data=domain_node_logits,
                                                         segment_ids=self.placeholders[
                                                             'domain_node_timestep_graph_ids_list'],
                                                         num_segments=self.placeholders['num_graphs'] * self.placeholders['max_length'])
        copied_domain_node_max_scores = tf.gather(params=domain_node_max_scores,
                                                  indices=self.placeholders['domain_node_timestep_graph_ids_list'])

        selected_domain_nodes = tf.cast(tf.equal(copied_domain_node_max_scores, domain_node_logits), dtype=tf.int32)
        correct_prediction_per_node = tf.cast(tf.equal(selected_domain_nodes, self.placeholders['domain_labels']),
                                              tf.float32)
        correct_prediction_per_node = tf.maximum(correct_prediction_per_node, self.placeholders['acc_mask'])
        correct_prediction = tf.unsorted_segment_prod(data=correct_prediction_per_node,
                                                      segment_ids=self.placeholders[
                                                          'domain_node_timestep_graph_ids_list_unshifted'],
                                                      num_segments=self.placeholders['num_graphs'])

        self.ops['accuracy'] = tf.reduce_mean(correct_prediction)


class OrderedSubsetGGNN(GGNN):
    def __init__(self, params: Mapping[str, Any], propagator=None, classifier=None, optimizer=None):
        if classifier is None:
            classifier = OrderedSubsetGGNNClassifier(**params)

        super().__init__(params, propagator, classifier, optimizer)

