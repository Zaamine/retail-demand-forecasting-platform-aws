"""Small, no-S3 tests for the deterministic CT training policy."""
import unittest

import pandas as pd

from pipelines.processing.build_ct_dataset import (
    choose_rotating_weekly_origins,
    started_store_dates,
    eligible_origin_state,
)


class CTTrainingDataTests(unittest.TestCase):
    def test_weekday_rotation_uses_global_history(self):
        all_days = pd.date_range('2017-01-29', '2017-03-18', freq='D')
        a = choose_rotating_weekly_origins(
            all_days, '2017-01-01', '2017-03-19'
        )
        self.assertEqual(len(a), 8)
        self.assertEqual(a[0].date().isoformat(), '2017-01-29')
        self.assertEqual(a[1].date().isoformat(), '2017-01-31')
        self.assertEqual(a[2].date().isoformat(), '2017-02-08')
        self.assertEqual(a[3].date().isoformat(), '2017-02-16')
        # Last origin remains strictly before training cutoff.
        self.assertTrue((a < pd.Timestamp('2017-03-19')).all())

    def test_missing_day_is_never_selected(self):
        days = pd.date_range('2017-01-29','2017-02-20',freq='D')
        days = days[days != pd.Timestamp('2017-02-07')]
        selected = choose_rotating_weekly_origins(days, '2017-01-01','2017-02-21')
        self.assertNotIn(pd.Timestamp('2017-02-07'), selected)

    def test_store_scope_excludes_pre_activity_origins(self):
        sales = pd.DataFrame({
            'date': pd.to_datetime(['2017-01-10','2017-02-01']),
            'store_nbr': [1,2],
            'family': ['FOOD','FOOD'],
            'sales': [1.0,3.0],
        })
        starts = started_store_dates(sales)
        origins = pd.DataFrame({
            'store_nbr': [1,2],
            'family': ['FOOD','FOOD'],
            'forecast_origin': pd.to_datetime(['2017-01-15']*2),
        })
        filtered = eligible_origin_state(origins,'2017-01-15',starts)
        self.assertEqual(filtered['store_nbr'].tolist(),[1])

    def test_zero_sales_does_not_start_a_store(self):
        sales = pd.DataFrame({
            'date': pd.to_datetime(['2017-01-10','2017-02-01']),
            'store_nbr': [1,1],
            'sales': [0,10.0],
        })
        starts = started_store_dates(sales)
        self.assertEqual(starts[1],pd.Timestamp('2017-02-01'))


if __name__=='__main__':
    unittest.main()
