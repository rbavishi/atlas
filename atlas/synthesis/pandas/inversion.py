import warnings
from abc import ABC, abstractmethod

import numpy as np
from typing import Dict, Any

from atlas.operators import operator
from atlas.strategies import DfsStrategy
from atlas.synthesis.pandas.checker import Checker
from atlas.synthesis.pandas.utils import Program, check_nan


#  ---------------------------------------------------------------------
#  NOTE:
#  Template generated using the function `create_inversion_template` in
#  atlas.synthesis.pandas.utils
#  ---------------------------------------------------------------------
class GeneratorInversionStrategy(DfsStrategy, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_args(self, state: Dict) -> Dict[str, Any]:
        pass

    def checked_select(self, domain, key, **kwargs):
        try:
            if 'default' in kwargs and (kwargs['default'] == key or kwargs['default'] is key or
                                        (str(key) in ['nan', 'None'] and str(kwargs['default']) == str(key))):
                yield key
                return
        except Exception as e:
            # logging.exception(e)
            pass

        try:
            if key in domain:
                yield key

        except Exception as e:
            checker = Checker.get_checker(key)
            if any(checker(key, i) for i in domain):
                yield key
            else:
                warnings.warn("Failed Comparison", domain, key)

    def checked_ordered_subset(self, domain, key):
        if all(i in domain for i in key):
            yield key

    def checked_subset(self, domain, key):
        if all(i in domain for i in key):
            yield key

    def checked_product(self, domain, key):
        if (not isinstance(key, (tuple, list))) or len(key) != len(domain):
            return

        if all(k in d for k, d in zip(key, domain)):
            yield key

    @operator(name="SelectExternal", gen_name="df.index", uid="1")
    def Inv0(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.columns", uid="1")
    def Inv1(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.dtypes", uid="1")
    def Inv2(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.ftypes", uid="1")
    def Inv3(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.values", uid="1")
    def Inv4(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.axes", uid="1")
    def Inv5(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.ndim", uid="1")
    def Inv6(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.size", uid="1")
    def Inv7(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.shape", uid="1")
    def Inv8(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.T", uid="1")
    def Inv9(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.as_matrix", uid="1")
    def Inv10(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.as_matrix", uid="2")
    def Inv11(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)

        if args['columns'] is None:
            yield True
        else:
            yield False

    @operator(name="OrderedSubset", gen_name="df.as_matrix", uid="3")
    def Inv12(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['columns']))

    @operator(name="SelectExternal", gen_name="df.get_dtype_counts", uid="1")
    def Inv13(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.get_ftype_counts", uid="1")
    def Inv14(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.select_dtypes", uid="1")
    def Inv15(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.select_dtypes", uid="2")
    def Inv16(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['exclude'] is None:
            yield True
        else:
            yield False

    @operator(name="Subset", gen_name="df.select_dtypes", uid="3")
    def Inv17(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, args['include'])

    @operator(name="Subset", gen_name="df.select_dtypes", uid="4")
    def Inv18(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, args['exclude'])

    @operator(name="SelectExternal", gen_name="df.astype", uid="1")
    def Inv19(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.astype", uid="2")
    def Inv20(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if isinstance(args['dtype'], str):
            yield from self.checked_select(domain, np.dtype(args['dtype']))
        elif isinstance(args['dtype'], dict):
            yield from set(domain) & {np.dtype(i) for i in args['dtype'].values()}
        else:
            yield from self.checked_select(domain, args['dtype'])

    @operator(name="SelectExternal", gen_name="df.isna", uid="1")
    def Inv21(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.notna", uid="1")
    def Inv22(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.head", uid="1")
    def Inv23(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.head", uid="2")
    def Inv24(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['n'])

    @operator(name="SelectExternal", gen_name="df.tail", uid="1")
    def Inv25(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.tail", uid="2")
    def Inv26(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['n'])

    @operator(name="SelectExternal", gen_name="df.at.__getitem__", uid="1")
    def Inv27(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.at.__getitem__", uid="2")
    def Inv28(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['key'][0])

    @operator(name="Select", gen_name="df.at.__getitem__", uid="3")
    def Inv29(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['key'][1])

    @operator(name="SelectExternal", gen_name="df.iat.__getitem__", uid="1")
    def Inv30(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.iat.__getitem__", uid="2")
    def Inv31(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['key'][0])

    @operator(name="Select", gen_name="df.iat.__getitem__", uid="3")
    def Inv32(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['key'][1])

    @operator(name="SelectExternal", gen_name="df.loc.__getitem__", uid="1")
    def Inv33(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.loc.__getitem__", uid="2")
    def Inv34(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if not isinstance(key[0], slice):
            return
        if key[0].step in [None, 1]:
            yield False
        else:
            yield True

    @operator(name="SelectFixed", gen_name="df.loc.__getitem__", uid="3")
    def Inv35(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if not isinstance(key[1], slice):
            return
        if key[1].step in [None, 1]:
            yield False
        else:
            yield True

    @operator(name="Subset", gen_name="df.loc.__getitem__", uid="4")
    def Inv36(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if key[0].step in [None, 1]:
            yield (key[0].start, key[0].stop)
        else:
            yield (key[0].stop, key[0].start)

    @operator(name="Subset", gen_name="df.loc.__getitem__", uid="5")
    def Inv37(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if key[1].step in [None, 1]:
            yield (key[1].start, key[1].stop)
        else:
            yield (key[1].stop, key[1].start)

    @operator(name="SelectExternal", gen_name="df.iloc.__getitem__", uid="1")
    def Inv38(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.iloc.__getitem__", uid="2")
    def Inv39(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if not isinstance(key[0], slice):
            return
        if key[0].step in [None, 1]:
            yield False
        else:
            yield True

    @operator(name="SelectFixed", gen_name="df.iloc.__getitem__", uid="3")
    def Inv40(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if not isinstance(key[1], slice):
            return
        if key[1].step in [None, 1]:
            yield False
        else:
            yield True

    @operator(name="Subset", gen_name="df.iloc.__getitem__", uid="4")
    def Inv41(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if key[0].step in [None, 1]:
            yield (key[0].start, key[0].stop)
        else:
            yield (key[0].stop, key[0].start)

    @operator(name="Subset", gen_name="df.iloc.__getitem__", uid="5")
    def Inv42(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        key = args['key']
        if key[1].step in [None, 1]:
            yield (key[1].start, key[1].stop)
        else:
            yield (key[1].stop, key[1].start)

    @operator(name="SelectExternal", gen_name="df.lookup", uid="1")
    def Inv43(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="OrderedSubset", gen_name="df.lookup", uid="2")
    def Inv44(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['row_labels']))

    @operator(name="OrderedSubset", gen_name="df.lookup", uid="3")
    def Inv45(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['col_labels']))

    @operator(name="SelectExternal", gen_name="df.xs", uid="1")
    def Inv46(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.xs", uid="2")
    def Inv47(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['drop_level'])

    @operator(name="SelectFixed", gen_name="df.xs", uid="3")
    def Inv48(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="Select", gen_name="df.xs", uid="4")
    def Inv49(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield from self.checked_select(domain, args['key'])

    @operator(name="Subset", gen_name="df.xs", uid="5")
    def Inv50(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, args['level'])

    @operator(name="Product", gen_name="df.xs", uid="6")
    def Inv51(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is not None:
            yield from self.checked_product(domain, args['key'])

    @operator(name="SelectExternal", gen_name="df.isin", uid="1")
    def Inv52(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.isin", uid="2")
    def Inv53(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['values'])

    @operator(name="SelectExternal", gen_name="df.where", uid="1")
    def Inv54(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.where", uid="2")
    def Inv55(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['cond'])

    @operator(name="SelectExternal", gen_name="df.where", uid="3")
    def Inv56(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectExternal", gen_name="df.mask", uid="1")
    def Inv57(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.mask", uid="2")
    def Inv58(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['cond'])

    @operator(name="SelectExternal", gen_name="df.mask", uid="3")
    def Inv59(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectExternal", gen_name="df.query", uid="1")
    def Inv60(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.query", uid="2")
    def Inv61(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['expr'])

    @operator(name="SelectExternal", gen_name="df.__getitem__", uid="1")
    def Inv62(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.__getitem__", uid="2")
    def Inv63(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if isinstance(args['key'], list):
            yield False
        else:
            yield True

    @operator(name="Select", gen_name="df.__getitem__", uid="3")
    def Inv64(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['key'])

    @operator(name="OrderedSubset", gen_name="df.__getitem__", uid="4")
    def Inv65(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['key']))

    @operator(name="SelectExternal", gen_name="df.add", uid="1")
    def Inv66(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.add", uid="2")
    def Inv67(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.add", uid="3")
    def Inv68(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.add", uid="4")
    def Inv69(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.add", uid="5")
    def Inv70(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.add", uid="6")
    def Inv71(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.sub", uid="1")
    def Inv72(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.sub", uid="2")
    def Inv73(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.sub", uid="3")
    def Inv74(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.sub", uid="4")
    def Inv75(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.sub", uid="5")
    def Inv76(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.sub", uid="6")
    def Inv77(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.mul", uid="1")
    def Inv78(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.mul", uid="2")
    def Inv79(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.mul", uid="3")
    def Inv80(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.mul", uid="4")
    def Inv81(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.mul", uid="5")
    def Inv82(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.mul", uid="6")
    def Inv83(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.div", uid="1")
    def Inv84(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.div", uid="2")
    def Inv85(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.div", uid="3")
    def Inv86(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.div", uid="4")
    def Inv87(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.div", uid="5")
    def Inv88(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.div", uid="6")
    def Inv89(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.truediv", uid="1")
    def Inv90(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.truediv", uid="2")
    def Inv91(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.truediv", uid="3")
    def Inv92(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.truediv", uid="4")
    def Inv93(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.truediv", uid="5")
    def Inv94(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.truediv", uid="6")
    def Inv95(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.floordiv", uid="1")
    def Inv96(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.floordiv", uid="2")
    def Inv97(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.floordiv", uid="3")
    def Inv98(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.floordiv", uid="4")
    def Inv99(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.floordiv", uid="5")
    def Inv100(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.floordiv", uid="6")
    def Inv101(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.mod", uid="1")
    def Inv102(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.mod", uid="2")
    def Inv103(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.mod", uid="3")
    def Inv104(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.mod", uid="4")
    def Inv105(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.mod", uid="5")
    def Inv106(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.mod", uid="6")
    def Inv107(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.pow", uid="1")
    def Inv108(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.pow", uid="2")
    def Inv109(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.pow", uid="3")
    def Inv110(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.pow", uid="4")
    def Inv111(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.pow", uid="5")
    def Inv112(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.pow", uid="6")
    def Inv113(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.radd", uid="1")
    def Inv114(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.radd", uid="2")
    def Inv115(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.radd", uid="3")
    def Inv116(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.radd", uid="4")
    def Inv117(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.radd", uid="5")
    def Inv118(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.radd", uid="6")
    def Inv119(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rsub", uid="1")
    def Inv120(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rsub", uid="2")
    def Inv121(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rsub", uid="3")
    def Inv122(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rsub", uid="4")
    def Inv123(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rsub", uid="5")
    def Inv124(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rsub", uid="6")
    def Inv125(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rmul", uid="1")
    def Inv126(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rmul", uid="2")
    def Inv127(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rmul", uid="3")
    def Inv128(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rmul", uid="4")
    def Inv129(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rmul", uid="5")
    def Inv130(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rmul", uid="6")
    def Inv131(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rdiv", uid="1")
    def Inv132(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rdiv", uid="2")
    def Inv133(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rdiv", uid="3")
    def Inv134(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rdiv", uid="4")
    def Inv135(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rdiv", uid="5")
    def Inv136(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rdiv", uid="6")
    def Inv137(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rtruediv", uid="1")
    def Inv138(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rtruediv", uid="2")
    def Inv139(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rtruediv", uid="3")
    def Inv140(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rtruediv", uid="4")
    def Inv141(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rtruediv", uid="5")
    def Inv142(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rtruediv", uid="6")
    def Inv143(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rfloordiv", uid="1")
    def Inv144(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rfloordiv", uid="2")
    def Inv145(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rfloordiv", uid="3")
    def Inv146(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rfloordiv", uid="4")
    def Inv147(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rfloordiv", uid="5")
    def Inv148(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rfloordiv", uid="6")
    def Inv149(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rmod", uid="1")
    def Inv150(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rmod", uid="2")
    def Inv151(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rmod", uid="3")
    def Inv152(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rmod", uid="4")
    def Inv153(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rmod", uid="5")
    def Inv154(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rmod", uid="6")
    def Inv155(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.rpow", uid="1")
    def Inv156(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.rpow", uid="2")
    def Inv157(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.rpow", uid="3")
    def Inv158(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rpow", uid="4")
    def Inv159(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.rpow", uid="5")
    def Inv160(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.rpow", uid="6")
    def Inv161(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['fill_value'] is None:
            yield None
        else:
            yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.lt", uid="1")
    def Inv162(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.lt", uid="2")
    def Inv163(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.lt", uid="3")
    def Inv164(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.lt", uid="4")
    def Inv165(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.lt", uid="5")
    def Inv166(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.gt", uid="1")
    def Inv167(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.gt", uid="2")
    def Inv168(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.gt", uid="3")
    def Inv169(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.gt", uid="4")
    def Inv170(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.gt", uid="5")
    def Inv171(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.le", uid="1")
    def Inv172(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.le", uid="2")
    def Inv173(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.le", uid="3")
    def Inv174(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.le", uid="4")
    def Inv175(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.le", uid="5")
    def Inv176(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.ge", uid="1")
    def Inv177(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.ge", uid="2")
    def Inv178(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.ge", uid="3")
    def Inv179(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.ge", uid="4")
    def Inv180(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.ge", uid="5")
    def Inv181(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.ne", uid="1")
    def Inv182(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.ne", uid="2")
    def Inv183(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.ne", uid="3")
    def Inv184(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.ne", uid="4")
    def Inv185(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.ne", uid="5")
    def Inv186(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.eq", uid="1")
    def Inv187(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.eq", uid="2")
    def Inv188(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.eq", uid="3")
    def Inv189(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.eq", uid="4")
    def Inv190(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['level'] is None:
            yield True
        else:
            yield False

    @operator(name="Select", gen_name="df.eq", uid="5")
    def Inv191(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.combine", uid="1")
    def Inv192(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.combine", uid="2")
    def Inv193(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectExternal", gen_name="df.combine", uid="3")
    def Inv194(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['func'])

    @operator(name="SelectFixed", gen_name="df.combine", uid="4")
    def Inv195(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['overwrite'])

    @operator(name="Select", gen_name="df.combine", uid="5")
    def Inv196(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.combine_first", uid="1")
    def Inv197(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.combine_first", uid="2")
    def Inv198(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectExternal", gen_name="df.apply", uid="1")
    def Inv199(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.apply", uid="2")
    def Inv200(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['func'])

    @operator(name="SelectFixed", gen_name="df.apply", uid="3")
    def Inv201(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.apply", uid="4")
    def Inv202(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['broadcast'])

    @operator(name="SelectFixed", gen_name="df.apply", uid="5")
    def Inv203(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['raw'])

    @operator(name="SelectExternal", gen_name="df.applymap", uid="1")
    def Inv204(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.applymap", uid="2")
    def Inv205(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['func'])

    @operator(name="SelectExternal", gen_name="df.agg", uid="1")
    def Inv206(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.agg", uid="2")
    def Inv207(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['func'])

    @operator(name="SelectFixed", gen_name="df.agg", uid="3")
    def Inv208(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.transform", uid="1")
    def Inv209(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.transform", uid="2")
    def Inv210(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['func'])

    @operator(name="SelectExternal", gen_name="df.groupby", uid="1")
    def Inv211(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.groupby", uid="2")
    def Inv212(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.groupby", uid="3")
    def Inv213(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['sort'])

    @operator(name="SelectFixed", gen_name="df.groupby", uid="4")
    def Inv214(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['as_index'])

    @operator(name="SelectFixed", gen_name="df.groupby", uid="5")
    def Inv215(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield (args['level'] is None)

    @operator(name="SelectFixed", gen_name="df.groupby", uid="6")
    def Inv216(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield not isinstance(args['level'], (list, tuple))

    @operator(name="Select", gen_name="df.groupby", uid="7")
    def Inv217(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="OrderedSubset", gen_name="df.groupby", uid="8")
    def Inv218(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectFixed", gen_name="df.groupby", uid="9")
    def Inv219(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from [True, False]  # Not sure what to do here

    @operator(name="Select", gen_name="df.groupby", uid="10")
    def Inv220(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['by'])

    @operator(name="Subset", gen_name="df.groupby", uid="11")
    def Inv221(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['by']))

    @operator(name="SelectExternal", gen_name="df.abs", uid="1")
    def Inv222(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.all", uid="1")
    def Inv223(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.all", uid="2")
    def Inv224(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.all", uid="3")
    def Inv225(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['bool_only'])

    @operator(name="SelectFixed", gen_name="df.all", uid="4")
    def Inv226(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.all", uid="5")
    def Inv227(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.all", uid="6")
    def Inv228(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.any", uid="1")
    def Inv229(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.any", uid="2")
    def Inv230(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.any", uid="3")
    def Inv231(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['bool_only'])

    @operator(name="SelectFixed", gen_name="df.any", uid="4")
    def Inv232(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.any", uid="5")
    def Inv233(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.any", uid="6")
    def Inv234(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.clip", uid="1")
    def Inv235(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.clip", uid="2")
    def Inv236(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['lower'], default=extra_kwargs['default'])
        yield from self.checked_select(domain, args['lower'], default=None)

    @operator(name="SelectExternal", gen_name="df.clip", uid="3")
    def Inv237(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['upper'], default=extra_kwargs['default'])
        yield from self.checked_select(domain, args['upper'], default=None)

    @operator(name="SelectExternal", gen_name="df.clip_lower", uid="1")
    def Inv238(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.clip_lower", uid="2")
    def Inv239(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['threshold'], default=extra_kwargs['default'])

    @operator(name="SelectExternal", gen_name="df.clip_upper", uid="1")
    def Inv240(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.clip_upper", uid="2")
    def Inv241(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['threshold'], default=extra_kwargs['default'])

    @operator(name="SelectExternal", gen_name="df.corr", uid="1")
    def Inv242(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.corr", uid="2")
    def Inv243(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['min_periods'])

    @operator(name="SelectFixed", gen_name="df.corr", uid="3")
    def Inv244(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['method'])

    @operator(name="SelectExternal", gen_name="df.corrwith", uid="1")
    def Inv245(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.corrwith", uid="2")
    def Inv246(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.corrwith", uid="3")
    def Inv247(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['drop'])

    @operator(name="SelectFixed", gen_name="df.corrwith", uid="4")
    def Inv248(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.count", uid="1")
    def Inv249(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.count", uid="2")
    def Inv250(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.count", uid="3")
    def Inv251(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.count", uid="4")
    def Inv252(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="Select", gen_name="df.count", uid="5")
    def Inv253(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.cov", uid="1")
    def Inv254(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.cov", uid="2")
    def Inv255(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['min_periods'], default=None)

    @operator(name="SelectExternal", gen_name="df.cummax", uid="1")
    def Inv256(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.cummax", uid="2")
    def Inv257(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.cummax", uid="3")
    def Inv258(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectExternal", gen_name="df.cummin", uid="1")
    def Inv259(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.cummin", uid="2")
    def Inv260(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.cummin", uid="3")
    def Inv261(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectExternal", gen_name="df.cumprod", uid="1")
    def Inv262(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.cumprod", uid="2")
    def Inv263(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.cumprod", uid="3")
    def Inv264(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectExternal", gen_name="df.cumsum", uid="1")
    def Inv265(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.cumsum", uid="2")
    def Inv266(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.cumsum", uid="3")
    def Inv267(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectExternal", gen_name="df.diff", uid="1")
    def Inv268(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.diff", uid="2")
    def Inv269(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['periods'], default=1)

    @operator(name="SelectFixed", gen_name="df.diff", uid="3")
    def Inv270(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.eval", uid="1")
    def Inv271(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.eval", uid="2")
    def Inv272(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['expr'])

    @operator(name="SelectExternal", gen_name="df.kurt", uid="1")
    def Inv273(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.kurt", uid="2")
    def Inv274(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.kurt", uid="3")
    def Inv275(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.kurt", uid="4")
    def Inv276(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.kurt", uid="5")
    def Inv277(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.kurt", uid="6")
    def Inv278(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.mad", uid="1")
    def Inv279(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.mad", uid="2")
    def Inv280(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.mad", uid="3")
    def Inv281(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.mad", uid="4")
    def Inv282(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="SelectFixed", gen_name="df.mad", uid="5")
    def Inv283(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.max", uid="1")
    def Inv284(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.max", uid="2")
    def Inv285(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.max", uid="3")
    def Inv286(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.max", uid="4")
    def Inv287(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.max", uid="5")
    def Inv288(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.max", uid="6")
    def Inv289(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.mean", uid="1")
    def Inv290(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.mean", uid="2")
    def Inv291(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.mean", uid="3")
    def Inv292(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.mean", uid="4")
    def Inv293(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.mean", uid="5")
    def Inv294(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.mean", uid="6")
    def Inv295(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.median", uid="1")
    def Inv296(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.median", uid="2")
    def Inv297(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.median", uid="3")
    def Inv298(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.median", uid="4")
    def Inv299(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.median", uid="5")
    def Inv300(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.median", uid="6")
    def Inv301(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.min", uid="1")
    def Inv302(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.min", uid="2")
    def Inv303(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.min", uid="3")
    def Inv304(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.min", uid="4")
    def Inv305(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.min", uid="5")
    def Inv306(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.min", uid="6")
    def Inv307(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.mode", uid="1")
    def Inv308(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.mode", uid="2")
    def Inv309(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.mode", uid="3")
    def Inv310(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectExternal", gen_name="df.pct_change", uid="1")
    def Inv311(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.pct_change", uid="2")
    def Inv312(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['periods'])

    @operator(name="Select", gen_name="df.pct_change", uid="3")
    def Inv313(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['limit'])

    @operator(name="SelectExternal", gen_name="df.prod", uid="1")
    def Inv314(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.prod", uid="2")
    def Inv315(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.prod", uid="3")
    def Inv316(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.prod", uid="4")
    def Inv317(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="Select", gen_name="df.prod", uid="5")
    def Inv318(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['min_count'])

    @operator(name="SelectFixed", gen_name="df.prod", uid="6")
    def Inv319(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.prod", uid="7")
    def Inv320(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.quantile", uid="1")
    def Inv321(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.quantile", uid="2")
    def Inv322(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="Select", gen_name="df.quantile", uid="3")
    def Inv323(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['q'])

    @operator(name="SelectFixed", gen_name="df.quantile", uid="4")
    def Inv324(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="Select", gen_name="df.quantile", uid="5")
    def Inv325(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['interpolation'])

    @operator(name="SelectExternal", gen_name="df.rank", uid="1")
    def Inv326(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.rank", uid="2")
    def Inv327(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.rank", uid="3")
    def Inv328(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['method'])

    @operator(name="SelectFixed", gen_name="df.rank", uid="4")
    def Inv329(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['na_option'])

    @operator(name="SelectFixed", gen_name="df.rank", uid="5")
    def Inv330(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.rank", uid="6")
    def Inv331(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ascending'])

    @operator(name="SelectFixed", gen_name="df.rank", uid="7")
    def Inv332(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['pct'])

    @operator(name="SelectExternal", gen_name="df.round", uid="1")
    def Inv333(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.round", uid="2")
    def Inv334(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['decimals'])

    @operator(name="SelectExternal", gen_name="df.sem", uid="1")
    def Inv335(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.sem", uid="2")
    def Inv336(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.sem", uid="3")
    def Inv337(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.sem", uid="4")
    def Inv338(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.sem", uid="5")
    def Inv339(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.sem", uid="6")
    def Inv340(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="Select", gen_name="df.sem", uid="7")
    def Inv341(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ddof'])

    @operator(name="Select", gen_name="df.sem", uid="8")
    def Inv342(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ddof'])

    @operator(name="SelectExternal", gen_name="df.skew", uid="1")
    def Inv343(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.skew", uid="2")
    def Inv344(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.skew", uid="3")
    def Inv345(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.skew", uid="4")
    def Inv346(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.skew", uid="5")
    def Inv347(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.skew", uid="6")
    def Inv348(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.sum", uid="1")
    def Inv349(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.sum", uid="2")
    def Inv350(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.sum", uid="3")
    def Inv351(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.sum", uid="4")
    def Inv352(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="Select", gen_name="df.sum", uid="5")
    def Inv353(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['min_count'])

    @operator(name="SelectFixed", gen_name="df.sum", uid="6")
    def Inv354(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.sum", uid="7")
    def Inv355(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.std", uid="1")
    def Inv356(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.std", uid="2")
    def Inv357(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.std", uid="3")
    def Inv358(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.std", uid="4")
    def Inv359(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.std", uid="5")
    def Inv360(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.std", uid="6")
    def Inv361(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="Select", gen_name="df.std", uid="7")
    def Inv362(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ddof'])

    @operator(name="Select", gen_name="df.std", uid="8")
    def Inv363(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ddof'])

    @operator(name="SelectExternal", gen_name="df.var", uid="1")
    def Inv364(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.var", uid="2")
    def Inv365(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.var", uid="3")
    def Inv366(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['numeric_only'])

    @operator(name="SelectFixed", gen_name="df.var", uid="4")
    def Inv367(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectFixed", gen_name="df.var", uid="5")
    def Inv368(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.var", uid="6")
    def Inv369(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="Select", gen_name="df.var", uid="7")
    def Inv370(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ddof'])

    @operator(name="Select", gen_name="df.var", uid="8")
    def Inv371(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ddof'])

    @operator(name="SelectExternal", gen_name="df.add_prefix", uid="1")
    def Inv372(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.add_prefix", uid="2")
    def Inv373(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['prefix'])

    @operator(name="SelectExternal", gen_name="df.add_suffix", uid="1")
    def Inv374(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.add_suffix", uid="2")
    def Inv375(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['suffix'])

    @operator(name="SelectExternal", gen_name="df.align", uid="1")
    def Inv376(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.align", uid="2")
    def Inv377(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.align", uid="3")
    def Inv378(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.align", uid="4")
    def Inv379(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['broadcast_axis'])

    @operator(name="SelectFixed", gen_name="df.align", uid="5")
    def Inv380(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['join'])

    @operator(name="SelectFixed", gen_name="df.align", uid="6")
    def Inv381(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="OrderedSubset", gen_name="df.align", uid="7")
    def Inv382(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.drop", uid="1")
    def Inv383(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.drop", uid="2")
    def Inv384(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.drop", uid="3")
    def Inv385(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is None

    @operator(name="Select", gen_name="df.drop", uid="4")
    def Inv386(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="Subset", gen_name="df.drop", uid="5")
    def Inv387(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['labels']))

    @operator(name="SelectExternal", gen_name="df.drop_duplicates", uid="1")
    def Inv388(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Subset", gen_name="df.drop_duplicates", uid="2")
    def Inv389(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['subset']))

    @operator(name="SelectFixed", gen_name="df.drop_duplicates", uid="3")
    def Inv390(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['keep'])

    @operator(name="SelectExternal", gen_name="df.duplicated", uid="1")
    def Inv391(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Subset", gen_name="df.duplicated", uid="2")
    def Inv392(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['subset']))

    @operator(name="SelectFixed", gen_name="df.duplicated", uid="3")
    def Inv393(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['keep'])

    @operator(name="SelectExternal", gen_name="df.equals", uid="1")
    def Inv394(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.equals", uid="2")
    def Inv395(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectExternal", gen_name="df.filter", uid="1")
    def Inv396(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.filter", uid="2")
    def Inv397(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if args['items'] is not None:
            yield 'use_items'
        elif args['like'] is not None:
            yield 'use_like'
        else:
            yield 'use_regex'

    @operator(name="Subset", gen_name="df.filter", uid="3")
    def Inv398(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['items']))

    @operator(name="SelectFixed", gen_name="df.filter", uid="4")
    def Inv399(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.filter", uid="5")
    def Inv400(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['like'])

    @operator(name="SelectFixed", gen_name="df.filter", uid="6")
    def Inv401(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.filter", uid="7")
    def Inv402(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['regex'])

    @operator(name="SelectExternal", gen_name="df.first", uid="1")
    def Inv403(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.first", uid="2")
    def Inv404(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['offset'])

    @operator(name="SelectExternal", gen_name="df.idxmax", uid="1")
    def Inv405(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.idxmax", uid="2")
    def Inv406(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.idxmax", uid="3")
    def Inv407(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectExternal", gen_name="df.idxmin", uid="1")
    def Inv408(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.idxmin", uid="2")
    def Inv409(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.idxmin", uid="3")
    def Inv410(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['skipna'])

    @operator(name="SelectExternal", gen_name="df.last", uid="1")
    def Inv411(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.last", uid="2")
    def Inv412(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['offset'])

    @operator(name="SelectExternal", gen_name="df.reindex", uid="1")
    def Inv413(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.reindex", uid="2")
    def Inv413_2(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['labels'], default=None)

    @operator(name="SelectExternal", gen_name="df.reindex", uid="3")
    def Inv414(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['fill_value'], default=np.nan)

    @operator(name="SelectExternal", gen_name="df.reindex", uid="4")
    def Inv415(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['limit'], default=None)

    @operator(name="SelectFixed", gen_name="df.reindex", uid="5")
    def Inv416(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args.get('index', None) is not None

    @operator(name="SelectFixed", gen_name="df.reindex", uid="6")
    def Inv417(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="Select", gen_name="df.reindex", uid="7")
    def Inv418(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.reindex_like", uid="1")
    def Inv419(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.reindex_like", uid="2")
    def Inv420(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['other'])

    @operator(name="SelectFixed", gen_name="df.reindex_like", uid="3")
    def Inv421(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['method'])

    @operator(name="SelectExternal", gen_name="df.rename", uid="1")
    def Inv422(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.rename", uid="2")
    def Inv423(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args.get('index', None) is not None

    @operator(name="SelectExternal", gen_name="df.rename", uid="3")
    def Inv424(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['index'])

    @operator(name="SelectExternal", gen_name="df.rename", uid="4")
    def Inv425(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['columns'])

    @operator(name="SelectFixed", gen_name="df.rename", uid="5")
    def Inv426(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.rename", uid="6")
    def Inv427(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['mapper'])

    @operator(name="Select", gen_name="df.rename", uid="7")
    def Inv428(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['level'])

    @operator(name="SelectExternal", gen_name="df.reset_index", uid="1")
    def Inv429(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.reset_index", uid="2")
    def Inv430(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['drop'])

    @operator(name="SelectFixed", gen_name="df.reset_index", uid="3")
    def Inv431(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['level'] is not None

    @operator(name="Subset", gen_name="df.reset_index", uid="4")
    def Inv432(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['level']))

    @operator(name="SelectFixed", gen_name="df.reset_index", uid="5")
    def Inv433(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['col_level'] == 0

    @operator(name="Select", gen_name="df.reset_index", uid="6")
    def Inv434(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['col_level'])

    @operator(name="Select", gen_name="df.reset_index", uid="7")
    def Inv435(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['col_fill'])

    @operator(name="SelectExternal", gen_name="df.set_index", uid="1")
    def Inv436(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.set_index", uid="2")
    def Inv437(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['drop'])

    @operator(name="SelectFixed", gen_name="df.set_index", uid="3")
    def Inv438(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['append'])

    @operator(name="OrderedSubset", gen_name="df.set_index", uid="4")
    def Inv439(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['keys']))

    @operator(name="SelectExternal", gen_name="df.take", uid="1")
    def Inv440(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.take", uid="2")
    def Inv441(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['indices'])

    @operator(name="SelectFixed", gen_name="df.take", uid="3")
    def Inv442(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectExternal", gen_name="df.dropna", uid="1")
    def Inv443(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.dropna", uid="2")
    def Inv444(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.dropna", uid="3")
    def Inv445(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['how'])

    @operator(name="SelectFixed", gen_name="df.dropna", uid="4")
    def Inv446(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['subset'] is None

    @operator(name="Subset", gen_name="df.dropna", uid="5")
    def Inv447(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['subset']))

    @operator(name="Select", gen_name="df.dropna", uid="6")
    def Inv448(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['thresh'])

    @operator(name="SelectExternal", gen_name="df.fillna", uid="1")
    def Inv449(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.fillna", uid="2")
    def Inv450(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.fillna", uid="3")
    def Inv451(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['method'])

    @operator(name="Select", gen_name="df.fillna", uid="4")
    def Inv452(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['limit'])
        yield max(domain)

    @operator(name="SelectFixed", gen_name="df.fillna", uid="5")
    def Inv453(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['value'] is None

    @operator(name="SelectExternal", gen_name="df.fillna", uid="6")
    def Inv454(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['value'])

    @operator(name="SelectExternal", gen_name="df.pivot_table", uid="1")
    def Inv455(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.pivot_table", uid="2")
    def Inv456(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['margins'])

    @operator(name="Select", gen_name="df.pivot_table", uid="3")
    def Inv457(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        if isinstance(args['aggfunc'], str):
            yield from self.checked_select(domain, args['aggfunc'])
        elif args['aggfunc'] is np.amax or args['aggfunc'] is np.max:
            yield 'max'
        elif args['aggfunc'] is np.amin or args['aggfunc'] is np.min:
            yield 'min'
        elif args['aggfunc'] is np.sum:
            yield 'sum'
        elif args['aggfunc'] is np.median:
            yield 'median'
        elif args['aggfunc'] is np.mean:
            yield 'mean'

    @operator(name="Select", gen_name="df.pivot_table", uid="4")
    def Inv458(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectFixed", gen_name="df.pivot_table", uid="5")
    def Inv459(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['dropna'])

    @operator(name="Select", gen_name="df.pivot_table", uid="6")
    def Inv460(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['margins_name'])

    @operator(name="SelectFixed", gen_name="df.pivot_table", uid="7")
    def Inv461(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['columns'] == []

    @operator(name="OrderedSubset", gen_name="df.pivot_table", uid="8")
    def Inv462(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['columns']))

    @operator(name="SelectFixed", gen_name="df.pivot_table", uid="9")
    def Inv463(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['index'] == []

    @operator(name="OrderedSubset", gen_name="df.pivot_table", uid="10")
    def Inv464(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['index']))

    @operator(name="SelectFixed", gen_name="df.pivot_table", uid="11")
    def Inv465(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield not isinstance(args['values'], (list, tuple))

    @operator(name="Select", gen_name="df.pivot_table", uid="12")
    def Inv466(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['values'])

    @operator(name="OrderedSubset", gen_name="df.pivot_table", uid="13")
    def Inv467(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['values']))

    @operator(name="SelectExternal", gen_name="df.pivot", uid="1")
    def Inv468(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.pivot", uid="2")
    def Inv469(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['columns'])

    @operator(name="Select", gen_name="df.pivot", uid="3")
    def Inv470(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['index'])

    @operator(name="Select", gen_name="df.pivot", uid="4")
    def Inv471(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['values'])

    @operator(name="SelectExternal", gen_name="df.reorder_levels", uid="1")
    def Inv472(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.reorder_levels", uid="2")
    def Inv473(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="OrderedSubset", gen_name="df.reorder_levels", uid="3")
    def Inv474(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['order']))

    @operator(name="SelectExternal", gen_name="df.sort_values", uid="1")
    def Inv475(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.sort_values", uid="2")
    def Inv476(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['axis'])

    @operator(name="SelectFixed", gen_name="df.sort_values", uid="3")
    def Inv477(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['na_position'])

    @operator(name="OrderedSubset", gen_name="df.sort_values", uid="4")
    def Inv478(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['by']))

    @operator(name="OrderedSubset", gen_name="df.sort_values", uid="5")
    def Inv479(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['by']))

    @operator(name="SelectFixed", gen_name="df.sort_values", uid="6")
    def Inv480(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['ascending'])

    @operator(name="SelectExternal", gen_name="df.stack", uid="1")
    def Inv481(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.stack", uid="2")
    def Inv482(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['dropna'])

    @operator(name="SelectFixed", gen_name="df.stack", uid="3")
    def Inv483(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        try:
            yield args['level'] != -1
        except:
            pass

    @operator(name="OrderedSubset", gen_name="df.stack", uid="4")
    def Inv484(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectExternal", gen_name="df.unstack", uid="1")
    def Inv485(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectFixed", gen_name="df.unstack", uid="2")
    def Inv486(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        try:
            yield args['level'] != -1
        except:
            pass

    @operator(name="OrderedSubset", gen_name="df.unstack", uid="3")
    def Inv487(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['level']))

    @operator(name="SelectFixed", gen_name="df.unstack", uid="4")
    def Inv488(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['fill_value'] is not None

    @operator(name="Select", gen_name="df.unstack", uid="5")
    def Inv489(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['fill_value'])

    @operator(name="SelectExternal", gen_name="df.melt", uid="1")
    def Inv490(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="Select", gen_name="df.melt", uid="2")
    def Inv491(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['var_name'])

    @operator(name="Select", gen_name="df.melt", uid="3")
    def Inv492(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['value_name'])

    @operator(name="SelectFixed", gen_name="df.melt", uid="4")
    def Inv493(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['id_vars'] is None

    @operator(name="SelectFixed", gen_name="df.melt", uid="5")
    def Inv494(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['value_vars'] is None

    @operator(name="OrderedSubset", gen_name="df.melt", uid="6")
    def Inv495(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['id_vars']))

    @operator(name="OrderedSubset", gen_name="df.melt", uid="7")
    def Inv496(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['value_vars']))

    @operator(name="SelectFixed", gen_name="df.melt", uid="8")
    def Inv497(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args['col_level'] is not None

    @operator(name="Select", gen_name="df.melt", uid="9")
    def Inv498(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['col_level'])

    @operator(name="SelectExternal", gen_name="df.merge", uid="1")
    def Inv499(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="df.merge", uid="2")
    def Inv500(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['right'])

    @operator(name="SelectFixed", gen_name="df.merge", uid="3")
    def Inv501(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['how'])

    @operator(name="SelectFixed", gen_name="df.merge", uid="4")
    def Inv502(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['sort'])

    @operator(name="SelectFixed", gen_name="df.merge", uid="5")
    def Inv503(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args.get('on', None) is not None

    @operator(name="Subset", gen_name="df.merge", uid="6")
    def Inv504(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['on']))

    @operator(name="SelectFixed", gen_name="df.merge", uid="7")
    def Inv505(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args.get('left_on', None) is None

    @operator(name="SelectFixed", gen_name="df.merge", uid="8")
    def Inv506(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield args.get('right_on', None) is None

    @operator(name="Subset", gen_name="df.merge", uid="9")
    def Inv507(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_subset(domain, tuple(args['left_on']))

    @operator(name="OrderedSubset", gen_name="df.merge", uid="10")
    def Inv508(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_ordered_subset(domain, tuple(args['right_on']))

    @operator(name="SelectExternal", gen_name="dfgroupby.count", uid="1")
    def Inv509(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.first", uid="1")
    def Inv510(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.last", uid="1")
    def Inv511(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.max", uid="1")
    def Inv512(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.mean", uid="1")
    def Inv513(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.median", uid="1")
    def Inv514(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.min", uid="1")
    def Inv515(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.idxmin", uid="1")
    def Inv516(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.idxmax", uid="1")
    def Inv517(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.prod", uid="1")
    def Inv518(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.size", uid="1")
    def Inv519(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.sum", uid="1")
    def Inv520(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.transform", uid="1")
    def Inv521(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])

    @operator(name="SelectExternal", gen_name="dfgroupby.transform", uid="2")
    def Inv522(self, domain, kwargs, **extra_kwargs):
        args = self.get_args(state=kwargs)
        yield from self.checked_select(domain, args['self'])


class SequenceFirstInversionStrategy(GeneratorInversionStrategy):
    """
    Inversion for the OOPSLA '19 generator which predicts entire sequences at once.
    """
    def __init__(self, program: Program):
        super().__init__()
        self.program: Program = program

    def get_args(self, state: Dict) -> Dict[str, Any]:
        return self.program.arguments[state['idx'] - 1]

    @operator(name='Sequence', tags=['function_sequence_prediction'])
    def FuncSeqInverter(self, **kwargs):
        yield self.program.functions
