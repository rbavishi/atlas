import random
import unittest

from atlas.models.tensorflow.graphs.operators import SelectGGNN


class SanityTests(unittest.TestCase):
    def test_select_small_1(self):
        def create_random_datapoint():
            #  A set of domain nodes, a set of context nodes, and the selected domain node has an edge
            #  to one of the context nodes. Thus it should be very easy for the network
            #  to learn to predict the right node
            domain = list(range(random.randrange(1, 10)))
            context = [i + len(domain) for i in range(random.randrange(1, 10))]
            #  The only node feature is whether it's a domain node or a context node
            nodes = [[0] for _ in domain] + [[1] for _ in context]
            choice = random.choice(domain)
            edges = [(choice, 0, random.choice(context))]

            return {
                'nodes': nodes,
                'edges': edges,
                'domain': domain,
                'choice': choice
            }

        training = [create_random_datapoint() for i in range(500)]
        validation = [create_random_datapoint() for i in range(50)]

        config = {
            'random_seed': 0,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 30000,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,
            'num_node_features': 2,
            'num_edge_types': 1
        }

        model = SelectGGNN(config)
        model.train(training, validation, 10)
