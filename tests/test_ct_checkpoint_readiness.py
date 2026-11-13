import json
import unittest
from datetime import date, timedelta
import pandas as pd

from scripts.ct_checkpoint_readiness import decide_readiness, parse_checkpoint, parse_s3_uri

CFG = {
    'champion_package_arn': 'arn:champion/1',
    'last_challenger_package_arn': 'arn:challenger/2'
}


def fake_state():
    return {
        'schema_version': 'ct_checkpoint_v1',
        'last_successful_ct_cutoff': '2017-08-15',
        'last_successful_cycle_phase': 'CHALLENGER_REGISTERED_PENDING',
        'last_cycle_promoted': False,
        'last_cycle_new_out_of_sample_wape_available': False,
        'registered_champion_reference_arn': 'arn:champion/1',
        'last_registered_challenger_arn': 'arn:challenger/2',
        'last_challenger_sha256': 'a'*64
    }


def full_window(n=3):
    first = date(2017,8,16)
    return pd.DataFrame([{'date': first+timedelta(days=day),'store_nbr': store,'family':'F', 'sales':1.5}
                         for day in range(14) for store in range(n)])


class CheckpointReadinessTests(unittest.TestCase):
    def test_skipped_same_last_date(self):
        result = decide_readiness(date(2017,8,15),date(2017,8,15), None)
        self.assertEqual(result['status'], 'SKIPPED')
        self.assertEqual(result['next_required_labeled_end'], '2017-08-29')
        self.assertEqual(result['observed_days_after_checkpoint'], 0)
        self.assertFalse(result['training_started'])

    def test_skipped_partial_window(self):
        result = decide_readiness(date(2017,8,15),date(2017,8,28),None)
        self.assertEqual(result['status'],'SKIPPED')
        self.assertEqual(result['observed_days_after_checkpoint'], 13)

    def test_ready_when_complete(self):
        result = decide_readiness(date(2017,8,15),date(2017,8,30),full_window(),expected_series=3)
        self.assertEqual(result['status'], 'READY')
        self.assertEqual(result['next_training_cutoff'],'2017-08-29')
        self.assertEqual(result['new_labeled_rows'],42)
        self.assertFalse(result['launch_permitted'])

    def test_block_duplicate_keys(self):
        frame = full_window()
        frame = pd.concat([frame,frame.iloc[[0]]],ignore_index=True)
        result=decide_readiness(date(2017,8,15),date(2017,8,29),frame,expected_series=3)
        self.assertEqual(result['reason'],'DUPLICATE_LABEL_KEYS')

    def test_block_missing_series(self):
        frame=full_window().iloc[1:]
        result=decide_readiness(date(2017,8,15),date(2017,8,29),frame,expected_series=3)
        self.assertEqual(result['reason'],'INCOMPLETE_14_DAY_SERIES_PANEL')

    def test_block_nonfinite_label(self):
        frame=full_window()
        frame.loc[0,'sales']=float('inf')
        result=decide_readiness(date(2017,8,15),date(2017,8,29),frame,expected_series=3)
        self.assertEqual(result['reason'],'NONFINITE_OR_MISSING_LABELS')

    def test_block_series_drift_across_days(self):
        frame=full_window()
        frame.loc[0,'store_nbr']=99
        result=decide_readiness(date(2017,8,15),date(2017,8,29),frame,expected_series=3)
        self.assertEqual(result['reason'],'SERIES_SET_DRIFT_IN_NEW_WINDOW')

    def test_fails_closed_on_checkpoint_mismatch(self):
        bad=fake_state()
        bad['last_successful_ct_cutoff']='2017-08-29'
        with self.assertRaisesRegex(ValueError,'pinned'):
            parse_checkpoint(json.dumps(bad).encode(),CFG)

    def test_checkpoint_valid(self):
        state=fake_state()
        got, sha=parse_checkpoint(json.dumps(state).encode(),CFG)
        self.assertEqual(got,state)
        self.assertEqual(len(sha),64)

    def test_bad_source_uri(self):
        with self.assertRaises(ValueError):
            parse_s3_uri('https://example.com/data/')

if __name__=='__main__':
    unittest.main()
