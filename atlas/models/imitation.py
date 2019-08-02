import json
import os
import datetime
import pickle
import shutil
from abc import ABC, abstractmethod
from typing import Collection, Dict, Set, Tuple, Optional

import ray
import tqdm

from atlas.models.encoding import OpEncoder, ParallelizedEncoder, StatefulEncoder
from atlas.models.models import OpModel
from atlas.tracing import GeneratorTrace
from atlas.utils.ioutils import IndexedFileWriter


class TraceImitationOpModel(OpModel, ABC):
    @abstractmethod
    def train(self, gen: 'Generator', training_traces: Collection[GeneratorTrace],
              validation_traces: Collection[GeneratorTrace] = None, **kwargs):
        pass


@ray.remote
class ParallelFileWriter(IndexedFileWriter):
    def close(self):
        super().close()
        return True


@ray.remote(num_cpus=1)
class FileWriterCollection:
    def __init__(self, path_prefix: str):
        self.path_prefix = path_prefix
        self.mapping: Dict[str, ParallelFileWriter] = {}

    def append(self, rel_path, obj):
        if rel_path not in self.mapping:
            path = f"{self.path_prefix}/{rel_path}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.mapping[rel_path] = ParallelFileWriter.remote(path)

        handle = self.mapping[rel_path]
        handle.append.remote(obj)

    def join(self):
        for f in self.mapping.values():
            ray.get(f.close.remote())


class DefaultIndependentOpModel(TraceImitationOpModel, ABC):

    def encode_traces(self, traces: Collection[GeneratorTrace], out_dir: str,
                      prefix: str = 'training', parallel_encoding: bool = False):

        @ray.remote(num_cpus=1)
        class ParallelEncoder:
            def __init__(self, encoder: OpEncoder, file_map: FileWriterCollection):
                self.encoder = encoder
                self.file_map = file_map

            def encode_trace(self, trace: GeneratorTrace):
                for op in trace.op_traces:
                    op_encoder = self.encoder.get_encoder(op.op_name, op.sid, op.oid)
                    encoding = op_encoder(op.domain, op.context, choice=op.choice, mode='training',
                                          op_name=op.op_name, oid=op.oid, sid=op.sid)
                    self.file_map.append.remote(f"{op.sid}/{prefix}_encoded.pkl", encoding)

        ray.init()
        num_workers = int(ray.available_resources()['CPU'] // 2)
        encoding_file_map = FileWriterCollection.remote(out_dir)
        op_list: Set[Tuple[str, str, str, str, str]] = set()

        if parallel_encoding:
            workers = [ParallelEncoder.remote(self.encoder, encoding_file_map) for _ in range(num_workers)]
            batch = []
            for trace in tqdm.tqdm(traces):
                op_list.update([(op.op_name, op.oid, op.sid, op.gen_group, op.gen_name) for op in trace.op_traces])
                if len(batch) < num_workers:
                    batch.append(trace)
                else:
                    ray.get([worker.encode_trace.remote(trace) for worker, trace in zip(workers, batch)])
                    batch.clear()

            if len(batch) > 0:
                ray.get([worker.encode_trace.remote(trace) for worker, trace in zip(workers, batch)])

        else:
            for trace in tqdm.tqdm(traces):
                for op in trace.op_traces:
                    op_list.update([(op.op_name, op.oid, op.sid, op.gen_group, op.gen_name) for op in trace.op_traces])
                    op_encoder = self.encoder.get_encoder(op.op_name, op.sid, op.oid)
                    encoding = op_encoder(op.domain, op.context, choice=op.choice, mode='training',
                                          op_name=op.op_name, oid=op.oid, sid=op.sid)
                    encoding_file_map.append.remote(f"{op.sid}/{prefix}_encoded.pkl", encoding)

        ray.get(encoding_file_map.join.remote())
        ray.shutdown()
        with open(f"{out_dir}/trainable_ops.json", 'w') as f:
            trainable_ops = [{
                'op_name': o[0],
                'oid': o[1],
                'sid': o[2],
                'gen_name': o[3],
                'gen_group': o[4]
            } for o in op_list]

            json.dump(trainable_ops, f)

    def train(self, gen: 'Generator', training_traces: Collection[GeneratorTrace],
              validation_traces: Collection[GeneratorTrace] = None,
              wdir: str = None,
              force_reencode: bool = False,
              **kwargs):

        if wdir is None:
            wdir = f"generator-model-{datetime.datetime.today():%d-%m-%Y-%H-%M-%S}"

        os.makedirs(wdir, exist_ok=True)
        data_dir = f"{wdir}/data"
        model_dir = f"{wdir}/model"
        os.makedirs(model_dir, exist_ok=True)

        if isinstance(self.encoder, StatefulEncoder) and os.path.exists(f"{model_dir}/encoder.pkl"):
            with open(f"{model_dir}/encoder.pkl", 'rb') as f:
                self.encoder = pickle.load(f)

        if force_reencode or not os.path.exists(data_dir):
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir, ignore_errors=True)

            os.makedirs(data_dir, exist_ok=True)
            self.encode_traces(training_traces, data_dir,
                               parallel_encoding=isinstance(self.encoder, ParallelizedEncoder))
            if validation_traces is not None:
                self.encode_traces(validation_traces, data_dir, prefix='validation',
                                   parallel_encoding=isinstance(self.encoder, ParallelizedEncoder))

            if isinstance(self.encoder, StatefulEncoder):
                with open(f"{model_dir}/encoder.pkl", 'wb') as f:
                    pickle.dump(self.encoder, f)

        with open(f"{data_dir}/trainable_ops.json", 'r') as f:
            trainable_ops = json.load(f)

        for trainable_op in trainable_ops:
            self.train_op(data_dir=data_dir, **trainable_op)

    @abstractmethod
    def train_op(self, data_dir: str,
                 op_name: str, oid: Optional[str], sid: str, gen_name: str, gen_group: Optional[str]):
        pass
