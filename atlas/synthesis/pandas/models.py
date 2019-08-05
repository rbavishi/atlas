import os
from abc import ABC
from typing import Optional, Collection, Any, Dict

from atlas.models.imitation import DefaultIndependentOpModel, DefaultOpModel
from atlas.models.tensorflow.graphs.gnn import GGNN
from atlas.models.tensorflow.graphs.operators import SelectGGNN
from atlas.synthesis.pandas.encoders import PandasGraphEncoder
from atlas.tracing import OpTrace
from atlas.utils.ioutils import IndexedFileReader, IndexedFileWriter


class BasePandasOperatorModel(DefaultOpModel, ABC):
    def __init__(self, sid: str, encoder: PandasGraphEncoder, model: GGNN):
        super().__init__(sid)
        self.encoder = encoder
        self.encoder_func = encoder.get_encoder(self.sid)
        self.model = model

    def dump_encodings(self, data: Collection[OpTrace], path: str = None):
        if isinstance(data, IndexedFileReader):
            if path is None:
                path = f"{data.path}.encoded"

            if os.path.exists(path):
                return IndexedFileReader(path)

        if path is None:
            path = 'train.pkl'

        encoding_file = IndexedFileWriter(path)
        for op in data:
            encoding_file.append(self.encoder_func(
                domain=op.domain,
                context=op.context,
                choice=op.choice,
                sid=op.sid
            ))

        encoding_file.close()
        return IndexedFileReader(path)

    def train(self, train_op_traces: Collection[OpTrace],
              validation_op_traces: Optional[Collection[OpTrace]] = None,
              num_epochs: int = 10, **kwargs):

        encoded_train = self.dump_encodings(train_op_traces)
        if validation_op_traces is not None:
            encoded_valid = self.dump_encodings(validation_op_traces)
        else:
            encoded_valid = None

        self.model.train(encoded_train, encoded_valid, num_epochs=num_epochs)

    def save(self, path: str):
        self.model.save(path)


class PandasSelect(BasePandasOperatorModel):
    def __init__(self, sid: str, encoder: PandasGraphEncoder, config: Dict):
        super().__init__(sid, encoder, SelectGGNN(config))


class PandasModelBasic(DefaultIndependentOpModel):
    def __init__(self, path: str = None):
        super().__init__(path)

    def infer(self, domain: Any, context: Any = None, sid: str = ''):
        pass

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
            'edge_weight_dropout': 0.8,
            'num_node_features': encoder.get_num_node_features(),
            'num_edge_types': encoder.get_num_edge_types()
        }

        return PandasSelect(sid, encoder, config)
