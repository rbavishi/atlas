def raise_unrecognized(op_name):
    raise Exception(f"This call to {op_name} was not recognized by Atlas. "
                    f"Please make sure you are not using aliases.")


def Select(*args, **kwargs):
    raise_unrecognized('Select')


def Subset(*args, **kwargs):
    raise_unrecognized('Subset')


def OrderedSubset(*args, **kwargs):
    raise_unrecognized('OrderedSubset')


def Product(*args, **kwargs):
    raise_unrecognized('Product')


def Sequence(*args, **kwargs):
    raise_unrecognized('Sequence')
