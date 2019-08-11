import datetime
import json
import os
import pickle
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, Any

import tqdm

from atlas.models import GeneratorModel, TrainableModel
from atlas.tracing import GeneratorTrace, OpTrace
from atlas.utils.genutils import unpack_sid
from atlas.utils.ioutils import IndexedFileWriter, IndexedFileReader


class TraceImitationModel(GeneratorModel, ABC):
    @abstractmethod
    def train(self, traces: Collection[GeneratorTrace], *args, **kwargs):
        pass


class IndependentOperatorsModel(TraceImitationModel, ABC):
    def __init__(self, work_dir: str = None):
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"generator-model-{datetime.datetime.today():%d-%m-%Y-%H-%M-%S}")

        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        self._safe_del = False

        self.model_map: Dict[str, TrainableModel] = {}
        self.modeled_sids: Dict[str, str] = {}

    def __del__(self):
        if self._safe_del:
            shutil.rmtree(self.work_dir, ignore_errors=True)

    def train(self,
              train_traces: Collection[GeneratorTrace],
              val_traces: Collection[GeneratorTrace] = None,
              **kwargs):

        #  First, go over all the traces and create separate data-sets for each operator
        train_datasets: Dict[str, Collection[OpTrace]] = self.create_operator_datasets(train_traces)
        val_datasets: Dict[str, Collection[OpTrace]] = {}
        if val_traces is not None:
            val_datasets: Dict[str, Collection[OpTrace]] = self.create_operator_datasets(val_traces, mode='validation')

        for sid, dataset in train_datasets.items():
            print(f"[+] Training model for {sid}")
            model: TrainableModel = self.get_op_model(sid)
            if model is not None:
                model_dir = f"{self.work_dir}/models/{sid}"
                os.makedirs(model_dir, exist_ok=True)
                self.model_map[sid] = model
                self.modeled_sids[sid] = model_dir

                model.train(dataset, val_datasets.get(sid, None), **kwargs)
                model.save(f"{model_dir}")

    def infer(self, domain: Any, context: Any = None, sid: str = ''):
        if sid not in self.model_map:
            return None

        return self.model_map[sid].infer(domain, context, sid)

    def create_operator_datasets(self, traces: Collection[GeneratorTrace],
                                 mode: str = 'training') -> Dict[str, Collection[OpTrace]]:
        file_maps: Dict[str, IndexedFileWriter] = {}
        path_maps: Dict[str, str] = {}
        for trace in tqdm.tqdm(traces):
            for op in trace.op_traces:
                if op.sid not in file_maps:
                    path = f"{self.work_dir}/data/{op.sid}"
                    os.makedirs(path, exist_ok=True)
                    file_maps[op.sid] = IndexedFileWriter(f"{path}/{mode}_op_data.pkl")
                    path_maps[op.sid] = f"{path}/{mode}_op_data.pkl"

                file_maps[op.sid].append(op)

        for v in file_maps.values():
            v.close()

        return {k: IndexedFileReader(v) for k, v in path_maps.items()}

    def get_op_model(self, sid: str) -> Optional[TrainableModel]:
        unpacked = unpack_sid(sid)
        op_type, oid = unpacked.op_type, unpacked.oid
        if oid is not None and hasattr(self, f"{op_type}_{oid}"):
            return getattr(self, f"{op_type}_{oid}")(sid)

        if hasattr(self, op_type):
            return getattr(self, op_type)(sid)

        return None

    def save(self, path: str):
        super().save(path)
        with open(f"{path}/model_list.json", "w") as f:
            json.dump({k: os.path.relpath(v, self.work_dir) for k, v in self.modeled_sids.items()}, f)

        if path != self.work_dir:
            self._safe_del = True
            shutil.rmtree(f"{path}/models", ignore_errors=True)
            shutil.copytree(f"{self.work_dir}/models", f"{path}/models")

        else:
            self._safe_del = False

    @classmethod
    def load(cls, path: str):
        model = cls(path)
        with open(f"{path}/model_list.json", "r") as f:
            model.modeled_sids = {k: f"{path}/{v}" for k, v in json.load(f).items()}

        model.load_models()

        return model

    def load_models(self):
        for sid, model_dir in self.modeled_sids.items():
            if sid in self.model_map:
                continue

            self.model_map[sid] = TrainableModel.load(model_dir)
