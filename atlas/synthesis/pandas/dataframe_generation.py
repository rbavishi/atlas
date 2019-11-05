import itertools
import math
import random
from typing import List, Optional, NamedTuple, Tuple

import numpy as np
import pandas as pd

from atlas import generator
from atlas.operators import operator
from atlas.strategies import RandStrategy
from atlas.stubs import Select, Subset
from atlas.synthesis.pandas.stubs import SelectRange, Shuffle, CoinToss


class ValueBag:
    def __init__(self, values: List, name: Optional[str] = None):
        self.values = values
        self.name = name

    def __iter__(self):
        yield from self.values

    def __len__(self):
        return len(self.values)


class Bags:
    #  Strings
    #      Entities
    names = ValueBag(["Amy", "Joseph", "Anne", "Kennedy", "Kira", "Brian", "Christie", "Michael", "Riya", "Nikita",
                      "Thomas", "Ganesh", "Paul", "Allen"],
                     "name")
    baz = ValueBag(["foo", "bar", "baz", "fizz", "buzz"], "funcs")
    fruits = ValueBag(["kiwi", "apple", "banana", "pear", "date", "cherimoya", "watermelon", "papaya"], "Fruits")
    countries = ValueBag(["Canada", "India", "Germany", "Brazil", "US", "AlienLand", "France", "Zimbabwe"], "country")

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
    small_ints = ValueBag(list(range(0, 24, 39)), "points")
    five_ints = ValueBag(list(range(15, 55, 5)), "how_much")
    more_ints = ValueBag([3, 123, 532, 391, 53, 483, 85, 584, 48, 68, 49], "more_ints")
    big_ints = ValueBag(list(range(400, 1000, 50)), "stocks")

    #  Floats
    small_floats = ValueBag([0.1, 0.234, 0.7, 0.23411, 0.54327, 0.834953, 0.4, 0.81231, 0.9, np.NaN], "prob")
    even_floats = ValueBag([x / 2 for x in range(-10, 10, 1)], "div_by_twos")
    big_floats = ValueBag([132141.124, 132186.432, 3024234.234, 4234.4, 894324.5, 23894243.7, 123.4, np.NaN], "wats")
    no_nans_floats = ValueBag([71.3, 123.4, 32.4, 85.5, 23.7, 23.8, 83.7], "no_nans")
    more_nans_floats = ValueBag([123.4, 2324.2, 213.789, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], "more_nans")

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
    value_bags: List[ValueBag] = [*Bags.string_bags, *Bags.int_bags, *Bags.float_bags]

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
    index_like_columns_prob: float = 0.35
    col_prefix: str = ''  # A common prefix for all columns
    nan_prob: float = 0.0  # Probability of having nans (apart from the values in the float/nan bags being picked)


class RandDfStrategy(RandStrategy):
    @operator
    def SelectRange(self, low: int, high: int, **kwargs):
        return random.randint(low, high)

    @operator
    def CoinToss(self, bias: float = 0.5, **kwargs):
        return np.random.choice([0, 1], p=[1 - bias, bias])

    @operator
    def Shuffle(self, domain, **kwargs):
        res = list(domain)
        random.shuffle(res)
        return res


@generator(strategy=RandDfStrategy())
def find_approximate_factoring(number: int, num_factors: int) -> List[int]:
    """
    Find a list of ints of size `num_factors` such that their product is *close* to `number`.
    For example, `find_approximate_factoring(number=3, num_factors=2) can return `[2, 2]` or `[2, 1]`.
    Args:
        number: The number the product of factors should be close to
        num_factors: Number of factors desired

    Returns:
        A list of integers of size `num_factors` such that their product is close to `number`.
    """

    if num_factors == 1:
        return [number]

    if number == 1:
        return [1] * num_factors

    picked = SelectRange(low=2, high=math.ceil(number ** (1 / num_factors)))
    remaining = math.ceil(number / picked)
    return [picked] + find_approximate_factoring(remaining, num_factors - 1)


@generator(strategy=RandDfStrategy())
def generate_index(length: int, num_levels: int) -> List[Tuple]:
    #  Number of values to generate for each level
    num_level_values = Shuffle(find_approximate_factoring(length, num_levels))
    level_values = []
    for num in num_level_values:
        #  Giving more weight to strings
        bag_collection = Select([Bags.string_bags, Bags.int_bags, Bags.string_bags])
        bag = Select([i for i in bag_collection if len(i) >= num])
        level_values.append(list(Subset(bag.values, lengths=[num])))

    return Subset(list(itertools.product(*level_values)), lengths=[length])


@generator(strategy=RandDfStrategy())
def generate_random_dataframe(cfg: DfConfig = None):
    if cfg is None:
        cfg = DfConfig()

    num_rows = SelectRange(low=cfg.min_height, high=cfg.max_height) if cfg.num_rows is None else cfg.num_rows
    num_cols = SelectRange(low=cfg.min_width, high=cfg.max_width) if cfg.num_cols is None else cfg.num_cols

    value_bags = cfg.value_bags[:]
    if cfg.nan_prob > 0:
        value_bags.extend([Bags.more_nans_floats] * int(cfg.nan_prob * 10))

    index_levels = cfg.index_levels
    if index_levels is None and CoinToss(bias=cfg.multi_index_prob) == 1:
        index_levels = SelectRange(low=2, high=cfg.max_index_levels)

    column_levels = cfg.column_levels
    if column_levels is None and CoinToss(bias=cfg.multi_col_index_prob) == 1:
        column_levels = SelectRange(low=2, high=cfg.max_column_levels)

    df_dict = {}
    if CoinToss(bias=cfg.index_like_columns_prob) == 1:
        index_tuples = generate_index(num_rows, num_levels=SelectRange(low=1, high=min(3, num_cols)))
        values = [list(c) for c in zip(*index_tuples)]
        df_dict = {f"NAME{idx}": val for idx, val in enumerate(values)}

    for _ in range(num_cols - len(df_dict)):
        bag: ValueBag = Select(value_bags)
        col_name = f"{cfg.col_prefix}{bag.name}"
        col_values = [Select(bag.values) for _ in range(num_rows)]

        #  Avoid repetition of column names
        while col_name in df_dict:
            col_name = f"{col_name}{SelectRange(low=0, high=10)}"

        df_dict[col_name] = col_values

    df = pd.DataFrame(df_dict)

    if CoinToss(bias=cfg.int_col_prob) == 1:
        df.columns = pd.Index(range(len(df.columns)))

    if index_levels is not None and index_levels > 1:
        index_tuples = generate_index(num_rows, num_levels=index_levels)
        df.index = pd.MultiIndex.from_tuples(index_tuples)

    if column_levels is not None and column_levels > 1:
        column_index_tuples = generate_index(num_cols, num_levels=column_levels)
        df.columns = pd.MultiIndex.from_tuples(column_index_tuples)

    return df
