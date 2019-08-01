import tensorflow as tf
from typing import Mapping, Any

from atlas.models.tensorflow.graphs.classifiers import GGNNGraphClassifier
from atlas.models.tensorflow.graphs.gnn import GGNN
from atlas.models.tensorflow.graphs.utils import MLP, SegmentBasedSoftmax


class SelectGGNNClassifier(GGNNGraphClassifier):
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

        graph_logits = self.define_graph_logits(graph_embeddings)

        self.ops['loss'] = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=graph_logits,
                                                           labels=self.placeholders['labels'])
        )

        probabilities = self.ops['probabilities'] = tf.nn.softmax(graph_logits)
        correct_prediction = tf.equal(tf.argmax(probabilities, -1, output_type=tf.int32), self.placeholders['labels'])
        self.ops['accuracy'] = tf.reduce_mean(tf.cast(correct_prediction, "float"))


class SelectGGNN(GGNN):
    def __init__(self, params: Mapping[str, Any], propagator=None, classifier=None, optimizer=None):
        if classifier is None:
            classifier = SelectGGNNClassifier(**params)

        super().__init__(params, propagator, classifier, optimizer)
