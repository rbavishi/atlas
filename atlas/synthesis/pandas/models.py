import os
from typing import Collection, Any, Mapping

from atlas.models.imitation import IndependentOperatorsModel
from atlas.models.tensorflow.graphs.operators import SelectGGNN
from atlas.synthesis.pandas.encoders import PandasGraphEncoder
from atlas.tracing import OpTrace
from atlas.utils.ioutils import IndexedFileReader, IndexedFileWriter


def dump_encodings(data: Collection[OpTrace], encoder: PandasGraphEncoder, sid: str, path: str = None):
    if isinstance(data, IndexedFileReader):
        if path is None:
            path = f"{data.path}.encoded"

        if os.path.exists(path):
            return IndexedFileReader(path)

    if path is None:
        path = 'train.pkl'

    encoding_file = IndexedFileWriter(path)
    encoder_func = encoder.get_encoder(sid)
    for op in data:
        encoding_file.append(encoder_func(
            domain=op.domain,
            context=op.context,
            choice=op.choice,
            sid=op.sid
        ))

    encoding_file.close()
    return IndexedFileReader(path)


class PandasSelect(SelectGGNN):
    def __init__(self, params: Mapping[str, Any], sid: str):
        super().__init__(params)
        self.encoder = PandasGraphEncoder()
        self.sid = sid

    def train(self, training_data: Collection[OpTrace], validation_data: Collection[OpTrace], *args, **kwargs):

        encoded_train = dump_encodings(training_data, self.encoder, self.sid)
        if validation_data is not None:
            encoded_valid = dump_encodings(validation_data, self.encoder, self.sid)
        else:
            encoded_valid = None

        super().train(encoded_train, encoded_valid, *args, **kwargs)

    def infer(self, domain, context: Any = None, sid: str = ''):
        encoding = self.encoder.get_encoder(self.sid)(domain, context, mode='inference', sid=sid)
        inference = super().infer([encoding])[0]
        print(inference, sid)
        return [val for val, prob in sorted(inference, key=lambda x: -x[1])]


class PandasModelBasic(IndependentOperatorsModel):
    def Select_input_selection(self, *args, **kwargs):
        return None

    def Select(self, sid: str):
        encoder = PandasGraphEncoder()
        config = {
            'random_seed': 0,
            'learning_rate': 0.001,
            'clamp_gradient_norm': 1.0,
            'node_dimension': 100,
            'classifier_hidden_dims': [100],
            'batch_size': 30000,
            'use_propagation_attention': True,
            'edge_msg_aggregation': 'avg',
            'residual_connections': {},
            'layer_timesteps': [1, 1, 1],
            'graph_rnn_cell': 'gru',
            'graph_rnn_activation': 'tanh',
            'edge_weight_dropout': 0.1,
            'num_node_features': encoder.get_num_node_features(),
            'num_edge_types': encoder.get_num_edge_types()
        }

        return PandasSelect(config, sid)
