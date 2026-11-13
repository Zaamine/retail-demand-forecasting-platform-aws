"""CT-6C cloud readiness for a Step Functions gate.

No model changes; no training; reads checkpoint + Glue sales and writes only
a small readiness report to the Processing Job output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/sagemaker/ct_readiness_checkpoint_v1.json'


def as_day(value):
    """Converts Arrow/Pandas/Python date scalar to an ISO date; rejects nulls."""
    import pandas as pd
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError('Curated labels contain a null date')
    if parsed.tz is not None:
        raise ValueError('Curated dates must be timezone-naive')
    return parsed.date()


def parse_checkpoint(blob: bytes, config: dict) -> tuple[dict, str]:
    """Read any valid *future* checkpoint; never pin the cutoff to V2."""
    sha = hashlib.sha256(blob).hexdigest()
    state = json.loads(blob)
    if state.get('schema_version') != 'ct_checkpoint_v1':
        raise ValueError('Unsupported CT checkpoint schema')
    if state.get('last_successful_cycle_phase') != 'CHALLENGER_REGISTERED_PENDING':
        raise ValueError('Checkpoint has not completed pending registration')
    if state.get('last_cycle_promoted') is not False:
        raise ValueError('CT must not auto-promote')
    if state.get('last_cycle_new_out_of_sample_wape_available') is not False:
        raise ValueError('Unknown future-label evaluation: fail closed')
    if state.get('last_challenger_approval_status') != 'PendingManualApproval':
        raise ValueError('Unexpected checkpoint approval status')
    if state.get('registered_champion_reference_arn') != config['champion_package_arn']:
        raise ValueError('Champion reference changed')
    prefix = config['registry_group_arn_prefix']
    candidate_arn = state.get('last_registered_challenger_arn', '')
    if not candidate_arn.startswith(prefix):
        raise ValueError('Unexpected CT Model Registry group')
    version = candidate_arn.removeprefix(prefix)
    if not version.isdecimal() or int(version) < 2:
        raise ValueError('Invalid registered challenger version')
    try:
        cutoff = date.fromisoformat(state['last_successful_ct_cutoff'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Invalid cutoff') from exc
    if cutoff < date.fromisoformat(config['first_completed_ct_cutoff']):
        raise ValueError('Checkpoint regressed before first validated CT')
    checksum = state.get('last_challenger_sha256', '')
    if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Missing/invalid challenger SHA-256')
    return state, sha


def decide_readiness(checkpoint_date, latest_date, new_rows, expected_series=1782, horizon=14):
    """Pure decision: fail closed for partial/unreliable new labels."""
    start = checkpoint_date + timedelta(days=1)
    required_end = checkpoint_date + timedelta(days=horizon)
    base = {
        'checkpoint_cutoff': checkpoint_date.isoformat(),
        'next_window_start': start.isoformat(),
        'next_required_labeled_end': required_end.isoformat(),
        'latest_observed_date': latest_date.isoformat() if latest_date else None,
        'expected_series': expected_series,
        'horizon_days': horizon,
        'expected_new_labeled_rows': expected_series * horizon,
        'training_started': False,
        'registry_mutated': False,
        'promotion_performed': False,
        'next_stage_configured': False,
        'launch_permitted': False,
    }
    if latest_date is None:
        return {**base, 'status': 'BLOCKED', 'reason': 'CURATED_SALES_EMPTY'}
    if latest_date < checkpoint_date:
        return {**base, 'status': 'BLOCKED', 'reason': 'SOURCE_BEHIND_CHECKPOINT'}
    if latest_date < required_end:
        return {**base, 'status': 'SKIPPED', 'reason': 'INSUFFICIENT_NEW_LABELED_DAYS',
                'observed_days_after_checkpoint': (latest_date - checkpoint_date).days}

    # Validate date, store-family uniqueness, and finite labels for exactly 14 days.
    # Perform this even if the source contains later observations.
    if new_rows is None:
        return {**base, 'status': 'BLOCKED', 'reason': 'WINDOW_ROWS_MISSING'}

    import pandas as pd
    frame = new_rows.copy()
    required_cols = {'date', 'store_nbr', 'family', 'sales'}
    if not required_cols.issubset(frame.columns):
        raise ValueError(f'Curated window schema missing: {required_cols - set(frame.columns)}')
    if frame.empty:
        return {**base, 'status': 'BLOCKED', 'reason': 'WINDOW_ROWS_MISSING'}
    frame['date'] = pd.to_datetime(frame['date'], errors='raise').dt.date
    if frame['date'].isna().any() or frame['store_nbr'].isna().any() or frame['family'].isna().any():
        return {**base, 'status': 'BLOCKED', 'reason': 'NULL_LABEL_KEYS'}
    if frame['date'].lt(start).any() or frame['date'].gt(required_end).any():
        raise ValueError('Incorrect date filter for readiness window')
    if frame.duplicated(['date','store_nbr','family']).any():
        return {**base, 'status': 'BLOCKED', 'reason': 'DUPLICATE_LABEL_KEYS'}
    if pd.to_numeric(frame['sales'], errors='coerce').isna().any():
        return {**base, 'status': 'BLOCKED', 'reason': 'NONFINITE_OR_MISSING_LABELS'}
    if not all(math.isfinite(float(v)) for v in frame['sales']):
        return {**base, 'status': 'BLOCKED', 'reason': 'NONFINITE_OR_MISSING_LABELS'}

    expected_days = [start + timedelta(days=i) for i in range(horizon)]
    counts = frame.groupby('date').size().to_dict()
    if any(counts.get(day) != expected_series for day in expected_days):
        return {**base, 'status': 'BLOCKED', 'reason': 'INCOMPLETE_14_DAY_SERIES_PANEL',
                'new_labeled_rows': len(frame),
                'complete_days': sum(counts.get(day) == expected_series for day in expected_days)}
    first = None
    for day in expected_days:
        subset = frame.loc[frame['date'].eq(day), ['store_nbr','family']]
        series = {(int(store), str(family)) for store,family in subset.itertuples(index=False, name=None)}
        if len(series) != expected_series:
            return {**base, 'status': 'BLOCKED', 'reason': 'INCOMPLETE_14_DAY_SERIES_PANEL'}
        if first is None:
            first = series
        elif series != first:
            return {**base, 'status': 'BLOCKED', 'reason': 'SERIES_SET_DRIFT_IN_NEW_WINDOW'}

    return {**base, 'status': 'READY', 'reason': 'COMPLETE_NEXT_LABELED_WINDOW',
            'new_labeled_rows': len(frame), 'next_training_cutoff': required_end.isoformat(),
            'notice': 'Data ready only: CT-6C training/orchestration must dynamically target this cutoff before any job is allowed.'}


def parse_s3_uri(uri):
    result = urlparse(uri)
    if result.scheme != 's3' or not result.netloc or not result.path.strip('/'):
        raise ValueError(f'Invalid curated Glue S3 URI: {uri}')
    return result.netloc, result.path.lstrip('/')


def read_curated_dates_and_window(local_dir, region, cutoff, end):
    """Read already SageMaker-mounted Parquet; no cloud SDK/credential dependency."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.dataset as ds

    data = ds.dataset(str(local_dir), format='parquet')
    typ = data.schema.field('date').type
    if pa.types.is_date(typ):
        arrow_cutoff, arrow_end = cutoff, end
    elif pa.types.is_timestamp(typ):
        arrow_cutoff = datetime.combine(cutoff, datetime.min.time())
        arrow_end = datetime.combine(end, datetime.min.time())
    else:
        raise TypeError(f'Unsupported sales date type: {typ}')
    dates = data.to_table(columns=['date']).column('date')
    if dates.null_count:
        raise ValueError('Curated source has null dates')
    if len(dates) == 0:
        return None, None
    latest = as_day(dates.to_pandas().max())
    if latest < end:
        return latest, None
    window = data.to_table(columns=['date','store_nbr','family','sales'],
                           filter=(ds.field('date') > arrow_cutoff) &
                                  (ds.field('date') <= arrow_end))
    return latest, window.to_pandas()


def run(config, checkpoint_path, sales_dir):
    """Read mounted checkpoint and sales; workflow separately verifies live SHA."""
    original_bytes = Path(checkpoint_path).read_bytes()
    state, checksum = parse_checkpoint(original_bytes, config)
    cutoff = date.fromisoformat(state['last_successful_ct_cutoff'])
    end = cutoff + timedelta(days=config['horizon_days'])
    latest, frame = read_curated_dates_and_window(sales_dir, config['region'], cutoff, end)
    decision = decide_readiness(cutoff, latest, frame, config['expected_series'], config['horizon_days'])
    if hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest() != checksum:
        raise RuntimeError('Mounted checkpoint changed during readiness scan')
    decision.update({'checkpoint_sha256': checksum,
                     'source': f'glue:{config["glue_database"]}.{config["glue_table"]}',
                     'checkpoint_s3_uri': f's3://{config["bucket"]}/{config["checkpoint_key"]}'})
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('/opt/ml/processing/output'))
    parser.add_argument('--checkpoint', type=Path, default=Path('/opt/ml/processing/input/checkpoint/last_successful_ct.json'))
    parser.add_argument('--sales-dir', type=Path, default=Path('/opt/ml/processing/input/sales'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['expected_series'] != 1782 or config['horizon_days'] != 14:
        raise ValueError('Unsupported panel/horizon; fail closed')
    result = run(config, args.checkpoint, args.sales_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'readiness_report.json').write_text(
        json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print('CT6C READINESS: '+result['status'], flush=True)


if __name__ == '__main__':
    main()
