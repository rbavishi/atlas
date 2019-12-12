import os
from typing import Collection, Any, Mapping, Dict

from atlas.operators import operator, OpInfo
from atlas.models.imitation import IndependentOperatorsModel
from atlas.models.tensorflow.graphs.operators import SelectGGNN, SelectFixedGGNN, OrderedSubsetGGNN, SequenceGGNN, \
    SequenceFixedGGNN, SubsetGGNN
from atlas.synthesis.pandas.encoders import PandasGraphEncoder
from atlas.tracing import OpTrace
from atlas.utils.ioutils import IndexedFileReader, IndexedFileWriter


def dump_encodings(data: Collection[OpTrace], encoder: PandasGraphEncoder, op_info: OpInfo, path: str = None):
    if isinstance(data, IndexedFileReader):
        if path is None:
            path = f"{data.path}.encoded"

        if os.path.exists(path):
            return IndexedFileReader(path)

    if path is None:
        path = 'train.pkl'

    encoding_file = IndexedFileWriter(path)
    encoder_func = encoder.get_encoder(op_info)
    for op in data:
        encoding_file.append(encoder_func(
            domain=op.domain,
            context=op.context,
            choice=op.choice,
            sid=op.op_info.sid
        ))

    encoding_file.close()
    return IndexedFileReader(path)


class PandasSelect(SelectGGNN):
    def __init__(self, params: Dict[str, Any], op_info: OpInfo):
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types()
        })

        self.encoder = PandasGraphEncoder()
        self.op_info = op_info

        super().__init__(params)

    def train(self, training_data: Collection[OpTrace], validation_data: Collection[OpTrace], *args, **kwargs):

        encoded_train = dump_encodings(training_data, self.encoder, self.op_info)
        if validation_data is not None:
            encoded_valid = dump_encodings(validation_data, self.encoder, self.op_info)
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context: Any = None, op_info: OpInfo = None, **kwargs):
        encoding = self.encoder.get_encoder(self.op_info)(domain, context, mode='inference', op_info=op_info)
        inference = super().infer([encoding])[0]
        return [val for val, prob in sorted(inference, key=lambda x: -x[1])]


class PandasSelectFixed(SelectFixedGGNN):
    def __init__(self, params: Dict[str, Any], domain_size: int, op_info: OpInfo):
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types(),
            'domain_size': domain_size
        })

        self.encoder = PandasGraphEncoder()
        self.op_info = op_info

        super().__init__(params)

    def train(self, training_data: Collection[OpTrace], validation_data: Collection[OpTrace], *args, **kwargs):

        encoded_train = dump_encodings(training_data, self.encoder, self.op_info)
        if validation_data is not None:
            encoded_valid = dump_encodings(validation_data, self.encoder, self.op_info)
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context: Any = None, op_info: OpInfo = None, **kwargs):
        encoding = self.encoder.get_encoder(self.op_info)(domain, context, mode='inference', op_info=op_info)
        inference = super().infer([encoding])[0]
        return [val for val, prob in sorted(inference, key=lambda x: -x[1])]


class PandasSubset(SubsetGGNN):
    def __init__(self, params: Dict[str, Any], op_info: OpInfo):
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types()
        })

        self.encoder = PandasGraphEncoder()
        self.op_info = op_info

        super().__init__(params)

    def train(self, training_data: Collection[OpTrace], validation_data: Collection[OpTrace], *args, **kwargs):

        encoded_train = dump_encodings(training_data, self.encoder, self.op_info)
        if validation_data is not None:
            encoded_valid = dump_encodings(validation_data, self.encoder, self.op_info)
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context: Any = None, op_info: OpInfo = None, **kwargs):
        encoding = self.encoder.get_encoder(self.op_info)(domain, context, mode='inference', op_info=op_info)
        inference = super().infer([encoding], top_k=100)[0]
        return [val for val, prob in sorted(inference, key=lambda x: -x[1])]


class PandasOrderedSubset(OrderedSubsetGGNN):
    def __init__(self, params: Dict[str, Any], op_info: OpInfo):
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types()
        })

        self.encoder = PandasGraphEncoder()
        self.op_info = op_info

        super().__init__(params)

    def train(self, training_data: Collection[OpTrace], validation_data: Collection[OpTrace], *args, **kwargs):

        encoded_train = dump_encodings(training_data, self.encoder, self.op_info)
        if validation_data is not None:
            encoded_valid = dump_encodings(validation_data, self.encoder, self.op_info)
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context: Any = None, op_info: OpInfo = None, **kwargs):
        encoding = self.encoder.get_encoder(self.op_info)(domain, context, mode='inference', op_info=op_info)
        inference = super().infer([encoding], top_k=100)[0]
        return [val for val, prob in sorted(inference, key=lambda x: -x[1])]


class PandasFuncSequence(SequenceFixedGGNN):
    def __init__(self, params: Dict[str, Any], num_classes: int, max_length: int, op_info: OpInfo):
        params.update({
            'num_node_features': self.encoder.get_num_node_features(),
            'num_edge_types': self.encoder.get_num_edge_types(),
            'num_classes': num_classes,
            'max_length': max_length,
        })

        self.encoder = PandasGraphEncoder()
        self.op_info = op_info

        super().__init__(params)

    def train(self, training_data: Collection[OpTrace], validation_data: Collection[OpTrace], *args, **kwargs):

        encoded_train = dump_encodings(training_data, self.encoder, self.op_info)
        if validation_data is not None:
            encoded_valid = dump_encodings(validation_data, self.encoder, self.op_info)
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context: Any = None, op_info: OpInfo = None, **kwargs):
        encoding = self.encoder.get_encoder(self.op_info)(domain, context, mode='inference', op_info=op_info)
        inference = super().infer([encoding], top_k=100)[0]
        return [val for val, prob in sorted(inference, key=lambda x: -x[1])]


class PandasModelBasic(IndependentOperatorsModel):
    common_config = {
        'random_seed': 0,
        'clamp_gradient_norm': 1.0,
        'use_propagation_attention': True,
        'edge_msg_aggregation': 'avg',
        'residual_connections': {},
        'graph_rnn_cell': 'gru',
        'graph_rnn_activation': 'tanh',
        'edge_weight_dropout': 0.1,
    }

    @operator
    def Select(self, op_info: OpInfo, **kwargs):
        config = {
            **self.common_config,
            'learning_rate': 0.001,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 30000,
            'layer_timesteps': [1, 1, 1],
        }

        return PandasSelect(config, op_info)

    @operator
    def SelectFixed(self, op_info: OpInfo, dataset: Collection[OpTrace] = None, **kwargs):
        if dataset is None:
            raise ValueError("SelectFixed needs access to dataset to compute domain size")

        config = {
            **self.common_config,
            'learning_rate': 0.001,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 30000,
            'layer_timesteps': [1, 1, 1],
        }

        domain_size = len(next(iter(dataset)).domain)
        return PandasSelectFixed(config, domain_size, op_info)

    @operator
    def OrderedSubset(self, op_info: OpInfo, **kwargs):
        config = {
            **self.common_config,
            'learning_rate': 0.001,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 30000,
            'layer_timesteps': [1, 1, 1],
        }

        return PandasOrderedSubset(config, op_info)

    @operator(name='Sequence', tags=['function_sequence_prediction'])
    def FuncSequence(self, op_info: OpInfo, dataset: Collection[OpTrace] = None, **kwargs):
        if dataset is None:
            raise ValueError("SelectFixed needs access to dataset to compute domain size")

        num_classes = len(next(iter(dataset)).domain)
        max_length = 3

        config = {
            **self.common_config,
            'learning_rate': 0.001,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 30000,
            'layer_timesteps': [1, 1, 1],
        }

        return PandasFuncSequence(config, num_classes, max_length, op_info)
