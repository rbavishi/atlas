import random
import unittest

import pytest
from atlas.models.tensorflow.graphs.earlystoppers import SimpleEarlyStopper
from atlas.models.tensorflow.graphs.operators import SelectGGNN, SubsetGGNN, OrderedSubsetGGNN, SelectFixedGGNN, \
    SequenceGGNN, SequenceFixedGGNN


class TestOperatorsBasic(unittest.TestCase):
    def select_fixed_small(self):
        #  Three domain nodes and some context nodes. Each domain node has a distinct node feature.
        #  The correct classification corresponds to the node having all the context nodes pointing towards it.
        #  Should be easy for the network to learn by simply looking at the node feature of domain node with all
        #  the incoming edges
        num_domain_nodes = 3
        domain = list(range(num_domain_nodes))
        context = [i + len(domain) for i in range(random.randrange(1, 10))]
        #  The only node feature is whether it's a domain node or a context node
        nodes = [[i] for i in domain] + [[num_domain_nodes] for _ in context]
        choice = random.choice(domain)
        edges = [(ctx, 0, choice) for ctx in context]

        return {
            'nodes': nodes,
            'domain': domain,
            'edges': edges,
            'choice': choice
        }

    def select_small(self):
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

    def subset_small(self):
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

    def ordered_subset_small(self):
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

    def sequence_fixed_small(self, num_classes: int, max_length: int):
        #  This is a replay the sequence task. Number of possibilities for each sequence element is fixed (as dictated
        #  by num_classes). There will be n nodes in the graph (n <= num_domain_nodes) with node features equal
        #  to the element IDs corresponding to the correct sequence. The edges amongst the nodes will dictate the order.

        domain = list(range(num_classes))
        choices = random.sample(domain, random.randrange(1, max_length))
        random.shuffle(choices)
        nodes = [[i] for i in choices]
        edges = []

        for i in range(len(choices)-1):
            edges.append((i, 0, i+1))

        return {
            'nodes': nodes,
            'edges': edges,
            'domain': domain,
            'choice': choices
        }

    def sequence_small(self):
        #  A set of domain nodes, a set of context nodes, and the selected domain nodes have an edge to one of
        #  the context nodes. Every selected node is repeated *twice* which is the main change from the OrderedSubset
        #  test above. The domain nodes also have a direction edge establishing the relative order
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
            'choice': sum([[i] * 2 for i in choices[:-1]], []) + [terminal],  # Repeated twice
            'terminal': terminal
        }

    def test_select_fixed_small_1(self):
        training = [self.select_fixed_small() for _ in range(500)]
        validation = [self.select_fixed_small() for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1],
            'num_node_features': 1 + len(training[0]['domain']),
            'num_edge_types': 1,
            'learning_rate': 0.01,
            'domain_size': len(training[0]['domain'])
        }

        model = SelectFixedGGNN(config)
        history = model.train(training, validation, 500, early_stopper=SimpleEarlyStopper(patience_zero_threshold=0.9,
                                                                                          patience=100))
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.90)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i])[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice']:
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 0.90)

    def test_select_fixed_small_1_strict(self):
        training = [self.select_fixed_small() for _ in range(500)]
        validation = [self.select_fixed_small() for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1],
            'num_node_features': 1 + len(training[0]['domain']),
            'num_edge_types': 1,
            'learning_rate': 0.01,
            'domain_size': len(training[0]['domain'])
        }

        model = SelectFixedGGNN(config)
        history = model.train(training, validation, 500, early_stopper=SimpleEarlyStopper(patience_zero_threshold=1.0,
                                                                                          patience=100))
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.00)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i])[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice']:
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 1.00)

    def test_select_small_1(self):
        training = [self.select_small() for _ in range(500)]
        validation = [self.select_small() for _ in range(50)]

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
        history = model.train(training, validation, 500, early_stopper=SimpleEarlyStopper(patience_zero_threshold=0.9,
                                                                                          patience=100))
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
        training = [self.select_small() for _ in range(500)]
        validation = [self.select_small() for _ in range(50)]

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
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.0)
        history = model.train(training, validation, 500, early_stopper=SimpleEarlyStopper(patience_zero_threshold=1.0,
                                                                                          patience=100))
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
        training = [self.subset_small() for _ in range(500)]
        validation = [self.subset_small() for _ in range(50)]

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
        history = model.train(training, validation, 500, early_stopper=SimpleEarlyStopper(patience_zero_threshold=0.9,
                                                                                          patience=100))
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
        training = [self.subset_small() for _ in range(500)]
        validation = [self.subset_small() for _ in range(50)]

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
        history = model.train(training, validation, 500, early_stopper=SimpleEarlyStopper(patience_zero_threshold=1.0,
                                                                                          patience=100))
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.0)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if set(inferred[0][0]) == set(i['choice']):
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 1.0)

    @pytest.mark.slow
    def test_ordered_subset_small_1(self):
        training = [self.ordered_subset_small() for _ in range(500)]
        validation = [self.ordered_subset_small() for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1, 1],
            'num_node_features': 3,
            'num_edge_types': 2,
            'learning_rate': 0.001
        }

        model = OrderedSubsetGGNN(config)
        history = model.train(training, validation, 500)
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.9)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice'][:-1]:  # Don't compare the terminal node
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 0.90)

    @pytest.mark.slow
    def test_ordered_subset_small_1_strict(self):
        training = [self.ordered_subset_small() for _ in range(500)]
        validation = [self.ordered_subset_small() for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1, 1],
            'num_node_features': 3,
            'num_edge_types': 2,
            'learning_rate': 0.001
        }

        model = OrderedSubsetGGNN(config)
        history = model.train(training, validation, 500)
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.0)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice'][:-1]:  # Don't compare the terminal node
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 1.0)

    def test_sequence_fixed_small_1(self):
        max_length = 3
        num_classes = 3
        training = [self.sequence_fixed_small(num_classes, max_length) for _ in range(500)]
        validation = [self.sequence_fixed_small(num_classes, max_length) for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1, 1],
            'num_node_features': num_classes,
            'num_edge_types': 1,
            'learning_rate': 0.01,
            'domain_size': num_classes,
            'max_length': max_length
        }

        model = SequenceFixedGGNN(config)
        history = model.train(training, validation, 1000, early_stopper=SimpleEarlyStopper(patience=1000,
                                                                                           patience_zero_threshold=0.9))
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.9)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice']:
                acc += 1
            else:
                print(inferred[0][0], i['choice'])

        self.assertGreaterEqual(acc / len(validation), 0.90)

    def test_sequence_fixed_small_1_strict(self):
        max_length = 3
        num_classes = 3
        training = [self.sequence_fixed_small(num_classes, max_length) for _ in range(500)]
        validation = [self.sequence_fixed_small(num_classes, max_length) for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1, 1],
            'num_node_features': num_classes,
            'num_edge_types': 1,
            'learning_rate': 0.01,
            'domain_size': num_classes,
            'max_length': max_length
        }

        model = SequenceFixedGGNN(config)
        history = model.train(training, validation, 1000, early_stopper=SimpleEarlyStopper(patience=1000,
                                                                                           patience_zero_threshold=1.0))
        self.assertGreaterEqual(history[-1]['valid_acc'], 1.0)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice']:
                acc += 1
            else:
                print(inferred[0][0], i['choice'])

        self.assertGreaterEqual(acc / len(validation), 1.0)

    @pytest.mark.slow
    def test_sequence_small_1(self):
        training = [self.sequence_small() for _ in range(500)]
        validation = [self.sequence_small() for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1, 1],
            'num_node_features': 3,
            'num_edge_types': 2,
            'learning_rate': 0.001,
            'max_length': max([len(i['choice']) for i in training + validation])
        }

        model = SequenceGGNN(config)
        history = model.train(training, validation, 1000, early_stopper=SimpleEarlyStopper(patience=1000,
                                                                                           patience_zero_threshold=0.9))
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.9)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice'][:-1]:  # Don't compare the terminal node
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 0.90)

    @pytest.mark.slow
    def test_sequence_small_1_strict(self):
        training = [self.sequence_small() for _ in range(500)]
        validation = [self.sequence_small() for _ in range(50)]

        config = {
            'node_dimension': 10,
            'classifier_hidden_dims': [10],
            'batch_size': 100000,
            'layer_timesteps': [1, 1, 1],
            'num_node_features': 3,
            'num_edge_types': 2,
            'learning_rate': 0.001,
            'max_length': max([len(i['choice']) for i in training + validation])
        }

        model = SequenceGGNN(config)
        history = model.train(training, validation, 1000, early_stopper=SimpleEarlyStopper(patience=1000,
                                                                                           patience_zero_threshold=1.0))
        self.assertGreaterEqual(history[-1]['valid_acc'], 0.9)

        #  Now test inference
        acc = 0
        for i in validation:
            #  Inference has the form [[(val, prob), (val, prob) ... (for every domain node) ] ... for every graph]
            inferred = sorted(model.infer([i], top_k=10)[0], key=lambda x: -x[1])
            if inferred[0][0] == i['choice'][:-1]:  # Don't compare the terminal node
                acc += 1

        self.assertGreaterEqual(acc / len(validation), 1.0)
