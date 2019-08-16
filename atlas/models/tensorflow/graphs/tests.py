import random
import unittest

from atlas.models.tensorflow.graphs.operators import SelectGGNN, SubsetGGNN


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
            matched = random.choice(context)
            edges = [(choice, 0, matched), (matched, 0, choice)]

            return {
                'nodes': nodes,
                'edges': edges,
                'domain': domain,
                'choice': choice
            }

        training = [create_random_datapoint() for i in range(500)]
        validation = [create_random_datapoint() for i in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1],
            'num_node_features': 2,
            'num_edge_types': 1
        }

        model = SelectGGNN(config)
        history = model.train(training, validation, 50)
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.90)

    def test_select_small_1_strict(self):
        def create_random_datapoint():
            #  A set of domain nodes, a set of context nodes, and the selected domain node has an edge
            #  to one of the context nodes. Thus it should be very easy for the network
            #  to learn to predict the right node
            domain = list(range(random.randrange(1, 10)))
            context = [i + len(domain) for i in range(random.randrange(1, 10))]
            #  The only node feature is whether it's a domain node or a context node
            nodes = [[0] for _ in domain] + [[1] for _ in context]
            choice = random.choice(domain)
            matched = random.choice(context)
            edges = [(choice, 0, matched), (matched, 0, choice)]

            return {
                'nodes': nodes,
                'edges': edges,
                'domain': domain,
                'choice': choice
            }

        training = [create_random_datapoint() for i in range(500)]
        validation = [create_random_datapoint() for i in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1],
            'num_node_features': 2,
            'num_edge_types': 1
        }

        model = SelectGGNN(config)
        history = model.train(training, validation, 50)
        #  Strict because we want 100% accuracy
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.0)

    def test_subset_small_1(self):
        def create_random_datapoint():
            #  A set of domain nodes, a set of context nodes, and the selected domain node has an edge
            #  to one of the context nodes. Thus it should be very easy for the network
            #  to learn to predict the right node
            domain = list(range(random.randrange(2, 10)))
            context = [i + len(domain) for i in range(random.randrange(1, 10))]
            #  The only node feature is whether it's a domain node or a context node
            nodes = [[0] for _ in domain] + [[1] for _ in context]
            choices = random.sample(domain, random.randrange(1, len(domain)))
            matched = random.choice(context)
            edges = []
            for choice in choices:
                edges.extend([(choice, 0, matched), (matched, 0, choice)])

            return {
                'nodes': nodes,
                'edges': edges,
                'domain': domain,
                'choices': choices
            }

        training = [create_random_datapoint() for i in range(500)]
        validation = [create_random_datapoint() for i in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1],
            'num_node_features': 2,
            'num_edge_types': 1,
            'learning_rate': 0.01
        }

        model = SubsetGGNN(config)
        history = model.train(training, validation, 50)
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.90)
