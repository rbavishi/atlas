from atlas.semantics import DfsSemantics, op_def
import torch
import torch.nn as nn


class WTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

    def add_op_module(self, op):
        self.layers.append(op)

    def forward(self, x):
        print(len(self.layers))
        for i in self.layers:
            x = i(x)

        return x


class SmartSemantics(DfsSemantics):
    def __init__(self):
        super().__init__()
        self.model = None

    def init_run(self):
        super().init_run()
        self.model = WTF()

    def finish_run(self):
        super().finish_run()
        print(self.model)
        print(self.model(torch.FloatTensor([0, 0, 0, 0, 1])))

    def Select_special(self, domain, *args, **kwargs):
        self.model.add_op_module(nn.Linear(5, 5))
        yield from domain

