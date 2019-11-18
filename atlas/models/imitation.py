import collections
import datetime
import os
import pickle
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Collection, Dict, Optional, Any, List, Callable

import tqdm
from atlas.models.core import GeneratorModel, TrainableModel, SerializableModel, TrainableSerializableModel
from atlas.models.utils import save_model, restore_model
from atlas.models.tensorflow.graphs.earlystoppers import EarlyStopper
from atlas.operators import OpInfo, OpResolvable, find_known_operators, resolve_operator
from atlas.tracing import GeneratorTrace, OpTrace
from atlas.utils.ioutils import IndexedFileWriter, IndexedFileReader


class TraceImitationModel(GeneratorModel, ABC):
    @abstractmethod
    def train(self, traces: Collection[GeneratorTrace], *args, **kwargs):
        pass


class IndependentOperatorsModel(TraceImitationModel, SerializableModel, OpResolvable, ABC):
    USE_DISK = True

    def __init__(self, work_dir: str = None):
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"generator-model-{datetime.datetime.today():%d-%m-%Y-%H-%M-%S}")

        self.work_dir = work_dir
        self.model_map: Dict[OpInfo, TrainableSerializableModel] = {}
        self.model_paths: Dict[OpInfo, str] = {}

        self.op_info_mapping: Dict[OpInfo, OpInfo] = {}

        self.model_definitions = find_known_operators(self)

    def get_op_model(self, op_info: OpInfo, dataset: Collection[OpTrace]) -> Optional[TrainableSerializableModel]:
        try:
            return resolve_operator(self.model_definitions, op_info)(self, op_info.sid)
        except ValueError:
            #  None implies that no model is defined for this operator
            return None

    def infer(self, domain: Any, context: Any = None, op_info: OpInfo = None, **kwargs):
        if op_info not in self.model_map:
            return None

        return self.model_map[op_info].infer(domain, context=context, op_info=op_info, **kwargs)

    def train(self,
              traces: Collection[GeneratorTrace],
              val_traces: Collection[GeneratorTrace] = None,
              skip_sid: Callable = None,
              **kwargs):
        #  First, go over all the traces and create separate data-sets for each operator
        train_data: Dict[OpInfo, Collection[OpTrace]] = self.create_operator_datasets(traces)
        val_data: Dict[OpInfo, Collection[OpTrace]] = {}

        if val_traces is not None:
            val_data = self.create_operator_datasets(val_traces, mode='validation')

        self.train_with_datasets(train_data, val_data, skip_sid, **kwargs)

    def train_with_datasets(self,
                            train_datasets: Dict[OpInfo, Collection[OpTrace]],
                            valid_datasets: Dict[OpInfo, Collection[OpTrace]],
                            skip_sid: Callable = None,
                            early_stopper: EarlyStopper = None,
                            **kwargs):
        tbegin = datetime.datetime.now()
        for op_info, dataset in train_datasets.items():
            if skip_sid and skip_sid(op_info.sid):
                print(f"Skip {op_info.sid}")
                continue

            if op_info in self.model_map:
                print(f"[+] Training the existing model for {op_info.sid}")
                model = self.model_map[op_info]
                model_dir = self.model_paths[op_info]
            else:
                model: TrainableSerializableModel = self.get_op_model(op_info, dataset)
                if model is None:
                    continue
                print(f"[+] Training a new model for {op_info.sid}")
                model_dir = f"{self.work_dir}/models/{op_info.sid}"
                os.makedirs(model_dir, exist_ok=True)
                self.model_map[op_info] = model
                self.model_paths[op_info] = model_dir

            early_stopper.reset()
            model.train(dataset, valid_datasets.get(op_info, None),
                        early_stopper=early_stopper, **kwargs)
            save_model(model, model_dir, no_zip=True)

            print(f"Done. Time elapsed: {datetime.datetime.now() - tbegin}")

    def create_operator_datasets(self, traces: Collection[GeneratorTrace],
                                 mode: str = 'training') -> Dict[OpInfo, Collection[OpTrace]]:
        if self.USE_DISK:
            file_maps: Dict[str, IndexedFileWriter] = {}
            path_maps: Dict[str, str] = {}
            for trace in tqdm.tqdm(traces):
                for op in trace.op_traces:
                    op_info = op.op_info
                    if op_info not in file_maps:
                        path = f"{self.work_dir}/data/{op_info.sid}"
                        os.makedirs(path, exist_ok=True)
                        file_maps[op_info] = IndexedFileWriter(f"{path}/{mode}_op_data.pkl")
                        path_maps[op_info] = f"{path}/{mode}_op_data.pkl"

                    file_maps[op_info].append(op)

            for v in file_maps.values():
                v.close()

            return {k: IndexedFileReader(v) for k, v in path_maps.items()}

        else:
            data: Dict[OpInfo, List[OpTrace]] = collections.defaultdict(list)
            for trace in tqdm.tqdm(traces):
                for op in trace.op_traces:
                    data[op.op_info].append(op)

            return data

    def load_operator_datasets(self, path_maps: Dict[str, str]):
        return {k: IndexedFileReader(v) for k, v in path_maps.items()}

    def serialize(self, path: str):
        with open(f"{path}/model_list.pkl", "wb") as f:
            pickle.dump({k: os.path.relpath(v, self.work_dir) for k, v in self.model_paths.items()}, f)

        if path != self.work_dir:
            shutil.rmtree(f"{path}/models", ignore_errors=True)
            shutil.copytree(f"{self.work_dir}/models", f"{path}/models")

    def deserialize(self, path: str):
        with open(f"{path}/model_list.pkl", "rb") as f:
            self.model_paths = {k: f"{path}/{v}" for k, v in pickle.load(f).items()}

        self.load_models()

    def load_models(self):
        for op_info, model_dir in self.model_paths.items():
            if op_info not in self.model_map:
                self.model_map[op_info] = restore_model(model_dir)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('model_map')
        state.pop('model_paths')
        state.pop('model_definitions')

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model_map: Dict[OpInfo, TrainableModel] = {}
        self.model_paths: Dict[OpInfo, str] = {}

        self.model_definitions = find_known_operators(self)

