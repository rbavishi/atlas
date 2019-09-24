import torch
import torch.nn
from abc import ABC, abstractmethod

from atlas.models import GeneratorModel
from atlas.operators import OpInfo, OpResolvable, find_known_operators, resolve_operator
from typing import Dict, Iterable, Any

from atlas.tracing import GeneratorTrace


class PyTorchGeneratorSharedStateModel(GeneratorModel, ABC, OpResolvable):
    def __init__(self):
        self.model_definitions = find_known_operators(self)
        self.model_map: Dict[OpInfo, PyTorchOpModel] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim_parameters = []
        self.cur_state = None

    def init_run(self):
        self.cur_state = self.get_initial_state()

    def get_op_model(self, op_info: OpInfo):
        if op_info not in self.model_map:
            try:
                model = self.model_map[op_info] = resolve_operator(self.model_definitions,
                                                                   op_info)(op_info).to(self.device)
                self.optim_parameters.append({'params': model.parameters()})

            except ValueError:
                return None

        return self.model_map[op_info]

    @abstractmethod
    def get_initial_state(self):
        pass

    def create_loader(self, dataset: Iterable[GeneratorTrace], batch_size: int = 1, eager: bool = True, **kwargs):
        if eager:
            dataset = list(dataset)
            processed = []
            for b in range(0, len(dataset), batch_size):
                batch = dataset[b: b + batch_size]
                if not all(len(i.op_traces) == len(batch[0].op_traces) for i in batch):
                    raise NotImplementedError("Cannot handle variable sized traces in a single batch")

                ops = [j.op_info for j in batch[0].op_traces]
                if not all([j.op_info for j in i.op_traces] == ops for i in batch):
                    raise NotImplementedError("Cannot handle traces with different set of operator calls")

                processed_batch = []
                for idx, op in enumerate(ops):
                    model = self.get_op_model(op)
                    if model is None:
                        continue

                    batch_data = []
                    for datapoint in batch:
                        t = datapoint.op_traces[idx]
                        batch_data.append((t.domain, t.context, t.choice))

                    processed_batch.append((model, model.encode(batch_data)))

                processed.append(processed_batch)

            return processed

    def train(self,
              training_data: Iterable[GeneratorTrace],
              validation_data: Iterable[GeneratorTrace] = None,
              num_epochs: int = 10,
              batch_size: int = 1,
              **kwargs):

        encoded_train = self.create_loader(training_data, batch_size=batch_size, **kwargs)
        encoded_valid = None
        if validation_data is not None:
            encoded_valid = self.create_loader(validation_data, batch_size=batch_size, **kwargs)

        optimizer = torch.optim.Adam(self.optim_parameters, lr=0.001)
        for epoch in range(num_epochs):
            for model in self.model_map.values():
                model.train()

            loss_all = 0
            correct_all = 0

            for batch in encoded_train:
                optimizer.zero_grad()
                state = self.get_initial_state()[[0] * len(batch)]
                cum_loss = 0
                correct = None
                for model, data in batch:
                    data = data.to(self.device)
                    loss, c, state = model(data, state=state)
                    cum_loss += loss
                    correct = c if correct is None else c * correct

                cum_loss.backward()
                loss_all += cum_loss.item() * len(batch)
                correct_all += correct.sum().item()
                optimizer.step()

            total = len(encoded_train)
            print(f"[Epoch {epoch}] Training Loss: {loss_all/total:.4f} Training Acc: {correct_all/total:.2f}")

            if encoded_valid is None:
                continue

            for model in self.model_map.values():
                model.eval()

            loss_all = 0
            correct_all = 0

            for batch in encoded_train:
                state = self.get_initial_state()
                cum_loss = 0
                correct = None
                for model, data in batch:
                    data = data.to(self.device)
                    loss, c, state = model(data, state=state)
                    cum_loss += loss
                    correct = c if correct is None else c * correct

                loss_all += cum_loss.item() * batch_size
                correct_all += correct.sum().item()

            total = len(encoded_train)
            print(f"[Epoch {epoch}] Validation Loss: {loss_all/total:.4f} Validation Acc: {correct_all/total:.2f}")

    def infer(self, domain: Any, context: Any = None, op_info: OpInfo = None, **kwargs):
        model = self.get_op_model(op_info)
        if model is None:
            return None

        res, state = model.infer(domain, context, state=self.cur_state)
        self.cur_state = state
        return res

    def save(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        pass


class PyTorchOpModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, data):
        pass

    @abstractmethod
    def infer(self, domain, context, state=None):
        pass
