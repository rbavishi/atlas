from abc import ABC, abstractmethod


def hook(func):
    setattr(func, "_is_operator_hook", True)
    return func


def is_hook(func):
    return getattr(func, "_is_operator_hook", False)


class Hook(ABC):
    def resolve_handler(self, op_kind: str, op_id: str):
        return getattr(self, op_kind + "_" + op_id,
                       getattr(self, op_kind, None))

    def create_hook(self, op_kind: str, op_id: str):
        handler = self.resolve_handler(op_kind, op_id)

        def hook_func(*args, oid=op_id, **kwargs):
            return handler(*args, **kwargs, oid=op_id)

        return hook_func


class PreHook(Hook, ABC):
    pass


class PostHook(Hook, ABC):
    pass
