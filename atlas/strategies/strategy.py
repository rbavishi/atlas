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

    def process_op(self, op_name: str, oid: Optional[str] = None) -> Tuple[str, str, Callable]:
        if oid is None:
            self.op_cnt[op_name] += 1
            sid = str(self.op_cnt[op_name])

        else:
            sid = oid

        label: str = op_name + "_" + str(sid)
        return label, sid, self.make_op(op_name, sid, oid)

    @abstractmethod
    def make_op(self, op_name: str, sid: str, oid: Optional[str]) -> Callable:
        pass
