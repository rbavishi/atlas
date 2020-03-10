from typing import Any, Iterator, List


class PeekableGenerator:
    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self._finished: bool = False
        self._cur_val: Any = None
        self._next_val: Any = next(iterator)
        self.step()

    def is_finished(self):
        return self._finished

    def peek(self):
        return self._cur_val

    def step(self):
        self._cur_val = self._next_val

        try:
            self._next_val = next(self.iterator)
        except StopIteration:
            self._finished = True


class IndexableGenerator:
    def __init__(self, iterator: Iterator):
        self.iterator = iterator
        self.saved_values: List[Any] = []

    def __getitem__(self, item):
        if len(self.saved_values) > item:
            return self.saved_values[item]

        while len(self.saved_values) <= item:
            self.saved_values.append(next(self.iterator))

        return self.saved_values[item]
