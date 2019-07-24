import collections
from abc import abstractmethod, ABC
from typing import Callable, Tuple, Dict, Any, Optional, Set


def operator(func):
    setattr(func, "_is_generator_op", True)
    return func


def is_operator(func):
    return getattr(func, "_is_generator_op", False)


class Strategy(ABC):
    def __init__(self):
        self.op_cnt: Dict[str, int] = collections.defaultdict(int)
        self.known_ops: Set[str] = {k for k in dir(self) if is_operator(getattr(self, k))}

    def init(self):
        pass

    def finish(self):
        pass

    def init_run(self):
        pass

    def finish_run(self):
        pass

    @abstractmethod
    def is_finished(self):
        pass

    def get_known_ops(self):
        return self.known_ops

    def process_op(self, kind: str, sid: Optional[str] = None) -> Tuple[str, str, Callable]:
        if sid is None:
            self.op_cnt[kind] += 1
            sid = str(self.op_cnt[kind])

        op_name: str = kind + "_" + str(sid)
        return op_name, sid, self.make_op(kind, sid)

    @abstractmethod
    def make_op(self, kind: str, sid: str) -> Callable:
        pass
