import pickle
import time
from typing import Optional

from atlas.models.imitation import DefaultIndependentOpModel
from atlas.models.tensorflow.graphs.operators import SelectGGNN
from atlas.synthesis.pandas.encoders import PandasGraphEncoder
from atlas.utils.ioutils import IndexedFileReader


class PandasModelBasic(DefaultIndependentOpModel):
    def __init__(self, encoder: PandasGraphEncoder):
        super().__init__(encoder)

    def get_specific_op_model(self, op_name: str, oid: Optional[str], **kwargs):
        resolution_order = []
        if oid is not None:
            resolution_order.append(f"{op_name}_{oid}")

        resolution_order.append(op_name)
        for label in resolution_order:
            if hasattr(self, label):
                handler = getattr(self, label)
                return handler(op_name=op_name, oid=oid, **kwargs)

        return None

    def train_op(self, data_dir: str,
                 op_name: str, oid: Optional[str], sid: str, gen_name: str, gen_group: Optional[str],
                 num_epochs: int = 10):

        model = self.get_specific_op_model(op_name=op_name, oid=oid, sid=sid, gen_name=gen_name, gen_group=gen_group)
        if model is None:
            print(f"Skipping training for {sid}")
            return

        print(f"Training model for {sid}")
        path = f"{data_dir}/{sid}"
        model.train(IndexedFileReader(f"{path}/training_encoded.pkl"),
                    IndexedFileReader(f"{path}/validation_encoded.pkl"), num_epochs)

    def Select_input_selection(self, *args, **kwargs):
        return None

    def Select(self, op_name: str, oid: Optional[str], sid: str, **kwargs):
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
            'num_edge_types': self.encoder.get_num_edge_types(),
            'num_node_features': self.encoder.get_num_node_features(),
        }

        return SelectGGNN(config)
