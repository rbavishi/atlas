import random
import unittest

from atlas.models.tensorflow.graphs.operators import SelectGGNN, SubsetGGNN, OrderedSubsetGGNN


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
            'num_edge_types': 1,
            'learning_rate': 0.01
        }

        model = SelectGGNN(config)
        history = model.train(training, validation, 50)
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.90)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i])[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice']:
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 0.90)

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

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i])[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice']:
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 1.00)

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
                'choice': choices
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

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if set(inferred[0][0]) == set(i['choice']):
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 0.90)

    def test_subset_small_1_strict(self):
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
                'choice': choices
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
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.00)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if set(inferred[0][0]) == set(i['choice']):
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 1.00)

    def test_ordered_subset_small_1(self):
        def create_random_datapoint():
            #  A set of domain nodes, a set of context nodes, and the selected domain nodes have an edge to one of
            #  the context nodes. The domain nodes also have a direction edge establishing the relative order
            #  between them. Thus it should be very easy for the network to learn to predict the right node
            domain = list(range(random.randrange(3, 10)))
            context = [i + len(domain) for i in range(random.randrange(1, 10))]
            terminal = len(domain) + len(context)
            #  The only node feature is whether it's a domain node or a context node
            nodes = [[0] for _ in domain] + [[1] for _ in context] + [[2]]
            choices = sorted(random.sample(domain, random.randrange(2, min(4, len(domain)))))
            choices.append(terminal)
            matched = random.choice(context)
            edges = []
            for choice in choices:
                edges.extend([(choice, 0, matched), (matched, 0, choice)])

            for i, j in zip(choices, choices[1:]):
                edges.append((i, 1, j))

            return {
                'nodes': nodes,
                'edges': edges,
                'domain': domain + [terminal],
                'choice': choices,
                'terminal': terminal
            }

        training = [create_random_datapoint() for i in range(500)]
        validation = [create_random_datapoint() for i in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1],
            'num_node_features': 3,
            'num_edge_types': 2,
            'learning_rate': 0.01
        }

        model = OrderedSubsetGGNN(config)
        history = model.train(training, validation, 200)
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.90)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice'][:-1]:  # Don't compare the terminal node
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 0.90)
