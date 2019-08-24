import itertools
import random

import numpy as np

from typing import Optional, List, NamedTuple
from atlas import generator
from atlas.strategies import RandStrategy, operator


class ValueBag:
    def __init__(self, values: List, name: Optional[str] = None):
        self.values = values
        self.name = name

    def __iter__(self):
        yield from self.values


class Bags:
    #  Strings
    #      Entities
    names = ValueBag(["Amy", "Joseph", "Anne", "Kennedy", "Kira", "Brian", "Christie"], "name")
    baz = ValueBag(["foo", "bar", "baz", "fizz", "buzz"], "funcs")
    fruits = ValueBag(["kiwi", "apple", "bananer", "pear", "date", "cherimoya"], "Fruits")
    countries = ValueBag(["Canada", "India", "Germany", "Brazil", "US", "AlienLand"], "country")

    #      Compositions
    things_1 = ValueBag(['_'.join(x)
                         for x in itertools.product(['3', '7', '24', '1', '2', '8', '9', '7'], baz.values)], "stuff1")
    things_2 = ValueBag(['_'.join(x)
                         for x in itertools.product(baz.values, ['32', '71', '24', '1', '2', '8', '9', '7'])], "stuff2")
    things_3 = ValueBag([".".join(x)
                         for x in itertools.product(fruits.values, baz.values)], "stats1")
    uber_things = ValueBag(["_".join(x)
                            for x in itertools.product(fruits.values, things_2.values)], "stats2")

    #  Ints
    small_ints = ValueBag(list(range(0, 24)), "points")
    five_ints = ValueBag(list(range(15, 55, 5)), "how_much")
    more_ints = ValueBag([3, 123, 532, 391, 53, 483, 85, 584, 48, 68, 49], "more_ints")
    big_ints = ValueBag(list(range(400, 1000, 50)), "stocks")

    #  Floats
    small_floats = ValueBag([0.1, 0.234, 0.7, 0.23411, 0.54327, 0.834953, 0.4, 0.81231, 0.9, np.NaN], "prob")
    even_floats = ValueBag([x / 2 for x in range(-10, 10, 1)], "div_by_twos")
    big_floats = ValueBag([132141.124, 132186.432, 3024234.234, 4234.4, 894324.5, 23894243.7, 123.4, np.NaN], "wats")
    no_nans_floats = ValueBag([71.3, 123.4, 32.4, 85.5, 23.7, 23.8, 83.7], "no_nans")
    more_nans_floats = ValueBag([123.4, 2324.2, 213.789, 12.54, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], "more_nans")

    bool_bags = [ValueBag([True, False], "bools")]
    string_bags = [names, baz, fruits, countries, things_2, things_1, things_3, uber_things]
    int_bags = [small_ints, five_ints, more_ints, big_ints]
    float_bags = [small_floats, big_floats, no_nans_floats, even_floats]


class DfConfig(NamedTuple):
    #  Basic DataFrame structure
    num_rows: Optional[int] = None
    num_cols: Optional[int] = None
    min_width: int = 1
    min_height: int = 1
    max_width: int = 7
    max_height: int = 7

    #  Handling multi-indices
    index_levels: Optional[int] = None
    column_levels: Optional[int] = None
    max_index_levels: int = 3
    max_column_levels: int = 3
    multi_index_prob: float = 0.2
    multi_col_index_prob: float = 0.2

    #  Various knobs
    int_col_prob: float = 0.2  # Columns are integers
    idx_mutation_prob: float = 0.2  # Have index other than range(0 num_rows). Only applicable if not multi-index
    col_prefix: str = ''  # A common prefix for all columns
    col_feeding_prob: float = 0.2  # Feed from previously generated columns. Controls spurious equality edges
    nan_prob: float = 0.0  # Probability of having nans (apart from the values in the float/nan bags being picked)


class RandDfStrategy(RandStrategy):
    @operator
    def SelectRange(self, low: int, high: int, **kwargs):
        return random.randint(low, high)

    @operator
    def CoinToss(self, bias: float = 0.5, **kwargs):
        return np.random.choice([0, 1], p=[1 - bias, bias])


@generator(strategy=RandDfStrategy())
def generate_random_dataframe(cfg: DfConfig):
    num_rows = SelectRange(low=cfg.min_height, high=cfg.max_height) if cfg.num_rows is None else cfg.num_rows
    num_cols = SelectRange(low=cfg.min_width, high=cfg.max_width) if cfg.num_cols is None else cfg.num_cols

    index_levels = cfg.index_levels
    if index_levels is None and CoinToss(cfg.multi_index_prob) == 1:
        index_levels = SelectRange(low=2, high=cfg.max_index_levels)

    column_levels = cfg.column_levels
    if column_levels is None and CoinToss(cfg.multi_col_index_prob) == 1:
        column_levels = SelectRange(low=2, high=cfg.max_column_levels)

    return num_rows, num_cols, index_levels, column_levels
