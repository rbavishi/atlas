class LambdaWrapper:
    def __init__(self, fn: str = None):
        self.fn = fn

    def __str__(self):
        return self.fn

    def __repr__(self):
        return self.fn

    def __call__(self, *args, **kwargs):
        return eval(self.fn)(*args, **kwargs)