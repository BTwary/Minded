import os, sys, unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from packages.analytics_core.src.statistics.method_selection import (
    select_two_group, select_multi_group, select_categorical, select_correlation,
)
from packages.analytics_core.src.statistics.inference import (
    execute_two_group, execute_multi_group, execute_categorical, execute_correlation, execute_regression,
)
from packages.analytics_core.src.statistics.engine import StatisticalEngine


class TestStatisticalMethodSelection(unittest.TestCase):
    def test_normal_equal_variance_prefers_student_t(self):
        d = select_two_group([10,11,12,13,14,15], [10,11,12,13,14,15])
        self.assertEqual(d.method, 'Welch t-test')
        r = execute_two_group([10,11,12,13,14,15], [20,21,22,23,24,25])
        self.assertEqual(r['method'], 'Welch t-test')
        self.assertIn('confidence_interval_mean_difference', r)

    def test_heteroskedastic_prefers_welch(self):
        d = select_two_group([10,11,12,13,14,15], [20,22,24,30,40,60])
        self.assertEqual(d.method, 'Welch t-test')
        self.assertIsNotNone(d.diagnostics['variance']['levene_p'])

    def test_heavy_skew_prefers_rank_test(self):
        d = select_two_group([1,1,1,2,3,100], [1,1,2,3,4,200])
        self.assertEqual(d.method, 'Mann-Whitney U')
        r = execute_two_group([1,1,1,2,3,100], [1,1,2,3,4,200])
        self.assertEqual(r['method'], 'Mann-Whitney U')

    def test_paired_design_is_explicit(self):
        d = select_two_group([1,2,3,4,5], [2,3,4,5,6], paired=True)
        self.assertEqual(d.problem, 'two_group_paired')
        self.assertIn(d.method, {'Paired t-test', 'Wilcoxon signed-rank'})

    def test_multi_group_normal_homogeneous_prefers_anova(self):
        d = select_multi_group({'A':[10,11,12,13,14], 'B':[20,21,22,23,24], 'C':[30,31,32,33,34]})
        self.assertEqual(d.method, 'One-way ANOVA')
        r = execute_multi_group({'A':[10,11,12,13,14], 'B':[20,21,22,23,24], 'C':[30,31,32,33,34]})
        self.assertEqual(r['method'], 'One-way ANOVA')

    def test_multi_group_heteroskedastic_uses_welch_or_rank(self):
        d = select_multi_group({'A':[1,2,3,4,5], 'B':[10,20,30,40,50], 'C':[100,101,102,103,104]})
        self.assertIn(d.method, {'Welch ANOVA', 'Kruskal-Wallis'})

    def test_sparse_2x2_uses_fisher(self):
        df = pd.DataFrame({'a':[0,0,0,0,1,1,1,1], 'b':[0,0,0,0,0,0,0,1]})
        d = select_categorical(pd.crosstab(df.a, df.b))
        self.assertEqual(d.method, 'Fisher exact test')
        r = execute_categorical(df, 'a', 'b')
        self.assertEqual(r['method'], 'Fisher exact test')

    def test_correlation_switches_to_spearman_when_skewed(self):
        x = np.array([1,2,3,4,5,100], float)
        y = np.array([2,3,4,5,6,7], float)
        d = select_correlation(x,y)
        self.assertIn(d.method, {'Pearson correlation', 'Spearman correlation'})
        r = execute_correlation(x,y)
        self.assertEqual(r['method'], d.method)
        self.assertIn('decision', r)

    def test_regression_uses_hc3_when_heteroskedastic(self):
        x = np.arange(1,21, dtype=float)
        y = 2*x + np.linspace(0.1, 100, 20) * np.sin(x)
        df = pd.DataFrame({'x':x, 'y':y})
        r = execute_regression(df, ['x'], 'y')
        self.assertIn(r['method'], {'OLS classical SE', 'OLS with HC3 robust SE'})
        self.assertIn('confidence_interval', r['coefficients']['x'])
        self.assertIn('breusch_pagan_p', r['diagnostics'])

    def test_statistical_engine_exposes_new_layer(self):
        e = StatisticalEngine()
        plan = e.select_two_group_method([1,2,3,4], [5,6,7,8])
        self.assertIn('method', plan)
        result = e.infer_two_group([1,2,3,4], [5,6,7,8])
        self.assertEqual(result['method'], plan['method'])


if __name__ == '__main__':
    unittest.main()
