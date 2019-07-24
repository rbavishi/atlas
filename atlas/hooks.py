from abc import ABC, abstractmethod


def hook(func):
    setattr(func, "_is_operator_hook", True)
    return func


def is_hook(func):
    return getattr(func, "_is_operator_hook", False)


class Hook(ABC):
    def resolve_handler(self, op_name: str, sid: str):
        return getattr(self, op_name + "_" + sid,
                       getattr(self, op_name, None))

    def create_hook(self, op_name: str, sid: str):
        handler = self.resolve_handler(op_name, sid)

        def hook_func(*args, **kwargs):
            kwargs['sid'] = sid
            return handler(*args, **kwargs)

        return hook_func


class PreHook(Hook, ABC):
    pass


class PostHook(Hook, ABC):
    pass
