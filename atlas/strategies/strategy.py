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
        self.sid_cnt: Dict[Tuple[str, str], int] = collections.defaultdict(int)
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

    def process_op(self, gen: 'Generator', op_name: str, oid: Optional[str] = None) -> Tuple[str, str, Callable]:
        path = "/".join(filter(None, [gen.group, gen.name]))
        label = f"{op_name}"
        if oid is not None:
            label += f"_{oid}"

        self.sid_cnt[path, label] += 1
        sid = f"{path}/{label}_{self.sid_cnt[path, label]}"
        op_label = f"{label}_{self.sid_cnt[path, label]}"

        return op_label, sid, self.make_op(gen, op_name, sid, oid)

    @abstractmethod
    def make_op(self, gen: 'Generator', op_name: str, sid: str, oid: Optional[str]) -> Callable:
        pass
