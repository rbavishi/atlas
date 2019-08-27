def raise_unrecognized(op_name):
    raise Exception(f"This call to {op_name} was not recognized by Atlas. "
                    f"Please make sure you are not using aliases.")


def stub(func):
    def stub_wrapper(*args, **kwargs):
        raise_unrecognized(func.__name__)

    return stub_wrapper
