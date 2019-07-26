from typing import Iterator, Any, Mapping


class Parameters(Mapping[str, Any]):
    def __init__(self):
        self.__dict__['mapping'] = {}

    def __getitem__(self, key: str) -> Any:
        return self.mapping.__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        return self.mapping.__setitem__(key, value)

    def __getattr__(self, item: str):
        return self.mapping.__getitem__(item)

    def __setattr__(self, key: str, value: Any):
        return self.mapping.__setitem__(key, value)

    def get(self, key: str, default: Any):
        if key in self.mapping:
            return self.mapping[key]

        return default

    def __len__(self) -> int:
        return len(self.mapping)

    def __iter__(self) -> Iterator[str]:
        return self.mapping.__iter__()
