from abc import ABC, abstractmethod


def hook(func):
    setattr(func, "_is_operator_hook", True)
    return func


def is_hook(func):
    return getattr(func, "_is_operator_hook", False)


class Hook(ABC):
    def resolve_handler(self, kind: str, sid: str):
        return getattr(self, kind + "_" + sid,
                       getattr(self, kind, None))

    def create_hook(self, kind: str, sid: str):
        handler = self.resolve_handler(kind, sid)

        def hook_func(*args, **kwargs):
            kwargs['sid'] = sid
            return handler(*args, **kwargs)

        return hook_func


class PreHook(Hook, ABC):
    pass


class PostHook(Hook, ABC):
    pass
