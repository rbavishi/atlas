import unittest

import atlas.synthesis.pandas.api
from atlas.synthesis.pandas.data_generation import generate_sequential_data
from atlas.utils import get_group_by_name

api_gens = {
    gen.name: gen for gen in get_group_by_name('pandas') if gen.metadata.get('data-generation', True)
}
print(api_gens.keys())


class TestSequentialDataGenerationBasic(unittest.TestCase):
    """
    The sequential data generator should be successful at least once when asked for
    a (input, output, program) tuple for just one function
    """

    def check(self, func: str):
        num_trials = 15
        successes = 0
        for _ in range(num_trials):
            try:
                generate_sequential_data([func], max_attempts=1)
                successes += 1

            except:
                pass

        self.assertGreater(successes/num_trials, 0.0)

    def test_df_index(self):
        self.check('df.index')

    def test_df_columns(self):
        self.check('df.columns')

    def test_df_dtypes(self):
        self.check('df.dtypes')

    def test_df_ftypes(self):
        self.check('df.ftypes')

    def test_df_values(self):
        self.check('df.values')

    def test_df_axes(self):
        self.check('df.axes')

    def test_df_ndim(self):
        self.check('df.ndim')

    def test_df_size(self):
        self.check('df.size')

    def test_df_shape(self):
        self.check('df.shape')

    def test_df_T(self):
        self.check('df.T')

    def test_df_as_matrix(self):
        self.check('df.as_matrix')

    def test_df_get_dtype_counts(self):
        self.check('df.get_dtype_counts')

    def test_df_get_ftype_counts(self):
        self.check('df.get_ftype_counts')

    def test_df_select_dtypes(self):
        self.check('df.select_dtypes')

    def test_df_astype(self):
        self.check('df.astype')

    def test_df_isna(self):
        self.check('df.isna')

    def test_df_notna(self):
        self.check('df.notna')

    def test_df_head(self):
        self.check('df.head')

    def test_df_tail(self):
        self.check('df.tail')

    def test_df_at___getitem__(self):
        self.check('df.at.__getitem__')

    def test_df_iat___getitem__(self):
        self.check('df.iat.__getitem__')

    def test_df_loc___getitem__(self):
        self.check('df.loc.__getitem__')

    def test_df_iloc___getitem__(self):
        self.check('df.iloc.__getitem__')

    def test_df_lookup(self):
        self.check('df.lookup')

    def test_df_xs(self):
        self.check('df.xs')

    def test_df_isin(self):
        self.check('df.isin')

    def test_df_where(self):
        self.check('df.where')

    def test_df_mask(self):
        self.check('df.mask')

    def test_df_query(self):
        self.check('df.query')

    def test_df___getitem__(self):
        self.check('df.__getitem__')

    def test_df_add(self):
        self.check('df.add')

    def test_df_sub(self):
        self.check('df.sub')

    def test_df_mul(self):
        self.check('df.mul')

    def test_df_div(self):
        self.check('df.div')

    def test_df_truediv(self):
        self.check('df.truediv')

    def test_df_floordiv(self):
        self.check('df.floordiv')

    def test_df_mod(self):
        self.check('df.mod')

    def test_df_pow(self):
        self.check('df.pow')

    def test_df_radd(self):
        self.check('df.radd')

    def test_df_rsub(self):
        self.check('df.rsub')

    def test_df_rmul(self):
        self.check('df.rmul')

    def test_df_rdiv(self):
        self.check('df.rdiv')

    def test_df_rtruediv(self):
        self.check('df.rtruediv')

    def test_df_rfloordiv(self):
        self.check('df.rfloordiv')

    def test_df_rmod(self):
        self.check('df.rmod')

    def test_df_rpow(self):
        self.check('df.rpow')

    def test_df_lt(self):
        self.check('df.lt')

    def test_df_gt(self):
        self.check('df.gt')

    def test_df_le(self):
        self.check('df.le')

    def test_df_ge(self):
        self.check('df.ge')

    def test_df_ne(self):
        self.check('df.ne')

    def test_df_eq(self):
        self.check('df.eq')

    def test_df_combine(self):
        self.check('df.combine')

    def test_df_combine_first(self):
        self.check('df.combine_first')

    def test_df_apply(self):
        self.check('df.apply')

    def test_df_groupby(self):
        self.check('df.groupby')

    def test_df_abs(self):
        self.check('df.abs')

    def test_df_all(self):
        self.check('df.all')

    def test_df_any(self):
        self.check('df.any')

    def test_df_clip(self):
        self.check('df.clip')

    def test_df_clip_lower(self):
        self.check('df.clip_lower')

    def test_df_clip_upper(self):
        self.check('df.clip_upper')

    def test_df_corr(self):
        self.check('df.corr')

    def test_df_corrwith(self):
        self.check('df.corrwith')

    def test_df_count(self):
        self.check('df.count')

    def test_df_cov(self):
        self.check('df.cov')

    def test_df_cummax(self):
        self.check('df.cummax')

    def test_df_cummin(self):
        self.check('df.cummin')

    def test_df_cumprod(self):
        self.check('df.cumprod')

    def test_df_cumsum(self):
        self.check('df.cumsum')

    def test_df_diff(self):
        self.check('df.diff')

    def test_df_kurt(self):
        self.check('df.kurt')

    def test_df_mad(self):
        self.check('df.mad')

    def test_df_max(self):
        self.check('df.max')

    def test_df_mean(self):
        self.check('df.mean')

    def test_df_median(self):
        self.check('df.median')

    def test_df_min(self):
        self.check('df.min')

    def test_df_mode(self):
        self.check('df.mode')

    def test_df_pct_change(self):
        self.check('df.pct_change')

    def test_df_prod(self):
        self.check('df.prod')

    def test_df_quantile(self):
        self.check('df.quantile')

    def test_df_rank(self):
        self.check('df.rank')

    def test_df_round(self):
        self.check('df.round')

    def test_df_sem(self):
        self.check('df.sem')

    def test_df_skew(self):
        self.check('df.skew')

    def test_df_sum(self):
        self.check('df.sum')

    def test_df_std(self):
        self.check('df.std')

    def test_df_var(self):
        self.check('df.var')

    def test_df_add_prefix(self):
        self.check('df.add_prefix')

    def test_df_add_suffix(self):
        self.check('df.add_suffix')

    def test_df_align(self):
        self.check('df.align')

    def test_df_drop(self):
        self.check('df.drop')

    def test_df_drop_duplicates(self):
        self.check('df.drop_duplicates')

    def test_df_duplicated(self):
        self.check('df.duplicated')

    def test_df_equals(self):
        self.check('df.equals')

    def test_df_filter(self):
        self.check('df.filter')

    def test_df_idxmax(self):
        self.check('df.idxmax')

    def test_df_idxmin(self):
        self.check('df.idxmin')

    def test_df_reindex(self):
        self.check('df.reindex')

    def test_df_reindex_like(self):
        self.check('df.reindex_like')

    def test_df_reset_index(self):
        self.check('df.reset_index')

    def test_df_set_index(self):
        self.check('df.set_index')

    def test_df_take(self):
        self.check('df.take')

    def test_df_dropna(self):
        self.check('df.dropna')

    def test_df_fillna(self):
        self.check('df.fillna')

    def test_df_pivot_table(self):
        self.check('df.pivot_table')

    def test_df_pivot(self):
        self.check('df.pivot')

    def test_df_reorder_levels(self):
        self.check('df.reorder_levels')

    def test_df_sort_values(self):
        self.check('df.sort_values')

    def test_df_stack(self):
        self.check('df.stack')

    def test_df_unstack(self):
        self.check('df.unstack')

    def test_df_melt(self):
        self.check('df.melt')

    def test_df_merge(self):
        self.check('df.merge')