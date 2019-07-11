def raise_unrecognized(op_name):
    raise Exception("This call to {op_name} was not recognized by Atlas. "
                    "Please make sure you are not using aliases.".format(op_name=op_name))


def Select(*args, **kwargs):
    raise_unrecognized('Select')


def Subsets(*args, **kwargs):
    raise_unrecognized('Subsets')


def OrderedSubsets(*args, **kwargs):
    raise_unrecognized('OrderedSubsets')


def Sequences(*args, **kwargs):
    raise_unrecognized('Sequences')
