import datetime
import json
import os
import pickle
import shutil

import tqdm
from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, Set, Any

from atlas.models.core import GeneratorModel
from atlas.tracing import GeneratorTrace, OpTrace
from atlas.utils.genutils import unpack_sid
from atlas.utils.ioutils import IndexedFileWriter, IndexedFileReader


class TraceImitationModel(GeneratorModel, ABC):
    @abstractmethod
    def train(self, traces: Collection[GeneratorTrace], *args, **kwargs):
        pass


class DefaultOpModel(ABC):
    def __init__(self, sid: str):
        self.sid = sid

    @abstractmethod
    def train(self, train_op_traces: Collection[OpTrace],
              validation_op_traces: Optional[Collection[OpTrace]] = None, **kwargs):
        pass

    @abstractmethod
    def infer(self, domain, context: Any = None, sid: str = ''):
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save model in an inference-suitable format
        Args:
            path: The path to store the model in.

        """

        os.makedirs(path, exist_ok=True)
        with open(f"{path}/loader.pkl", 'wb') as f:
            pickle.dump(self.load, f)

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'DefaultOpModel':
        with open(f"{path}/loader.pkl", 'rb') as f:
            loader = pickle.load(f)

        return loader(path)


class DefaultIndependentOpModel(TraceImitationModel, ABC):
    def __init__(self, path: str = None):
        if path is None:
            path = f"generator-model-{datetime.datetime.today():%d-%m-%Y-%H-%M-%S}"

        self.path = path
        os.makedirs(path, exist_ok=True)
        self.model_map: Dict[str, DefaultOpModel] = {}
        self.modeled_sids: Dict[str, str] = {}

    def train(self, training_traces: Collection[GeneratorTrace],
              validation_traces: Collection[GeneratorTrace] = None,
              **kwargs):
        #  First, go over all the traces and create separate data-sets for each operator
        training_datasets: Dict[str, Collection[OpTrace]] = self.create_operator_datasets(training_traces)
        if validation_traces is not None:
            validation_datasets: Dict[str, Collection[OpTrace]] = self.create_operator_datasets(validation_traces)
        else:
            validation_datasets = {}

        #  Now, train each operator model separately
        self.modeled_sids: Dict[str, str] = {}
        for sid, dataset in training_datasets.items():
            print(f"[+] Training model for {sid}")
            model: DefaultOpModel = self.get_op_model(sid)
            if model is not None:
                self.model_map[sid] = model
                model.train(dataset, validation_datasets.get(sid, None), **kwargs)
                model_dir = f"{self.path}/models/{sid}"
                os.makedirs(model_dir, exist_ok=True)
                model.save(f"{model_dir}")
                self.modeled_sids[sid] = model_dir

        with open(f"{self.path}/model_list.json", "w") as f:
            json.dump(self.modeled_sids, f)

    def infer(self, domain: Any, context: Any = None, sid: str = ''):
        if sid not in self.model_map:
            return None

        return self.model_map[sid].infer(domain, context, sid)

    def create_operator_datasets(self, traces: Collection[GeneratorTrace]) -> Dict[str, Collection[OpTrace]]:
        file_maps: Dict[str, IndexedFileWriter] = {}
        path_maps: Dict[str, str] = {}
        for trace in tqdm.tqdm(traces):
            for op in trace.op_traces:
                if op.sid not in file_maps:
                    path = f"{self.path}/data/{op.sid}"
                    os.makedirs(path, exist_ok=True)
                    file_maps[op.sid] = IndexedFileWriter(f"{path}/op_data.pkl")
                    path_maps[op.sid] = f"{path}/op_data.pkl"

                file_maps[op.sid].append(op)

        for v in file_maps.values():
            v.close()

        return {k: IndexedFileReader(v) for k, v in path_maps.items()}

    def get_op_model(self, sid: str) -> Optional[DefaultOpModel]:
        unpacked = unpack_sid(sid)
        op_type, oid = unpacked.op_type, unpacked.oid
        if oid is not None and hasattr(self, f"{op_type}_{oid}"):
            return getattr(self, f"{op_type}_{oid}")(sid)

        if hasattr(self, op_type):
            return getattr(self, op_type)(sid)

        return None

    def save(self, path: str):
        super().save(path)

        if path != self.path:
            with open(f"{path}/model_list.json", "w") as f:
                json.dump(self.modeled_sids, f)

            shutil.rmtree(f"{path}/models", ignore_errors=True)
            shutil.copytree(f"{self.path}/models", f"{path}/models")

    @classmethod
    def load(cls, path: str):
        model = cls(path)
        with open(f"{path}/model_list.json", "r") as f:
            model.modeled_sids = json.load(f)

        model.load_models()

        return model

    def load_models(self):
        for sid, model_dir in self.modeled_sids.items():
            if sid in self.model_map:
                continue

            self.model_map[sid] = DefaultOpModel.load(model_dir)
