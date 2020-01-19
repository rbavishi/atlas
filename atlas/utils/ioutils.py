import multiprocessing
import cloudpickle as pickle
import random
import struct
from typing import BinaryIO, Callable, Any, Collection


class IndexedFileWriter:
    def __init__(self, path, mode='w', writer: Callable[[Any], bytes] = pickle.dumps):
        if mode in ['w', 'a']:
            mode += 'b'
        else:
            raise ValueError(f"Unrecognized mode {mode}")

        self.f: BinaryIO = open(path, mode)
        self.index_f: BinaryIO = open(path + '.index', mode)
        self.writer = writer

    def append(self, record):
        offset = self.f.tell()
        self.f.write(self.writer(record))
        self.index_f.write(struct.pack('<Q', offset))

    def close(self):
        self.f.close()
        self.index_f.close()


class IndexedFileReader(Collection):
    def __init__(self, path, index_path=None, loader=pickle.load):
        self.path = path
        self.f = open(path, 'rb')
        if index_path is None:
            self.index_f = open(path + '.index', 'rb')
        else:
            self.index_f = open(index_path, 'rb')

        self.indices = self.read_indices()
        self.loader = loader

    def read_indices(self):
        indices = []
        while True:
            offset = self.index_f.read(8)
            if not offset:
                break
            offset, = struct.unpack('<Q', offset)
            indices.append(offset)

        return indices

    def close(self):
        self.f.close()
        self.index_f.close()

    def shuffle(self):
        random.shuffle(self.indices)

    def set_loader(self, fn):
        self.loader = fn

    def __getitem__(self, idx):
        self.f.seek(self.indices[idx])
        return self.loader(self.f)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for i in range(len(self.indices)):
            yield self.__getitem__(i)

    def parallel_iter(self, fn, num_processes=4, batch_size=100):
        total = len(self.indices)
        with multiprocessing.Pool(num_processes) as p:
            for i in range(0, total, batch_size):
                batch = [self.__getitem__(j) for j in range(i, min(total, i + batch_size))]
                for res in p.imap(fn, batch):
                    yield res

    def __contains__(self, x) -> bool:
        for i in iter(self):
            if i is x:
                return True

        return False

