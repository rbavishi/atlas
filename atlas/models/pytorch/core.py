from abc import ABC

from atlas.operators import OpInfo


class PyTorchGeneratorModel(ABC):
    def __init__(self, shared_state: bool = False):
        self.shared_state: bool = shared_state

    def get_op_model(self, op_info: OpInfo):
        pass
