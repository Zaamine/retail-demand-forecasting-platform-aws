"""Rebuild frozen Notebook 05 supervised training data at a newer label cutoff.

Run in a SageMaker Processing container with curated sales mounted under
/opt/ml/processing/input/sales and the source bundle mounted as /code.
No V1 artifact is modified. All predictions/features at each historical T
use only observations through that T; labels may be read through the cutoff.
"""
import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path('/opt/ml/processing/input/code')
if CODE.is_dir():
    sys.path.insert(0, str(CODE))
else:
    # Also supports PYTHONPATH=src local testing.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


HORIZON = 14
EXPECTED_SERIES = 1782
KEYS = ['store_nbr', 'family', 'target_date']


def choose_rotating_weekly_origins(complete_dates, first_sales_date, cutoff):
    """Same absolute-calendar anchored rotating-weekday rule as Notebook 05.

    Sample across ALL history before applying the 2-year training window;
    restarting the weekday rotation at the rolling-window start is incorrect.
    """
    dates = pd.DatetimeIndex(complete_dates).sort_values().unique()
    cutoff = pd.Timestamp(cutoff)
    dates = dates[(dates >= pd.Timestamp(first_sales_date) + pd.Timedelta(days=28))
                  & (dates < cutoff)]
    if dates.empty:
        raise ValueError('No eligible weekly training origins')

    frame = pd.DataFrame({'forecast_origin': dates})
    frame['week_start'] = frame['forecast_origin'] - pd.to_timedelta(
        frame['forecast_origin'].dt.dayofweek, unit='D'
    )
    selected = []
    for number, (_, week) in enumerate(frame.groupby('week_start', sort=True)):
        desired = number % 7
        week = week.copy()
        week['weekday_distance'] = (
            week['forecast_origin'].dt.dayofweek - desired
        ).abs()
        origin = week.sort_values(['weekday_distance', 'forecast_origin']).iloc[0]
        selected.append(pd.Timestamp(origin['forecast_origin']))
    return pd.DatetimeIndex(selected)


def started_store_dates(sales):
    """Historical first positive store-wide activity; safe for any past origin."""
    observed_positive = sales.loc[sales['sales'] > 0, ['store_nbr', 'date']]
    return observed_positive.groupby('store_nbr')['date'].min().to_dict()


def eligible_origin_state(origin_features, date, first_positive):
    date = pd.Timestamp(date)
    state = origin_features.loc[
        origin_features['forecast_origin'].eq(date)
    ].copy()
    started_at = state['store_nbr'].map(first_positive)
    return state.loc[started_at.notna() & started_at.le(date)].copy()


def build_labeled_origin(date, cutoff, regular, origin_features, sales_lookup,
                         first_positive, schema):
    from features.horizon import expand_forecast_origin_v1
    from features.target_aligned import build_target_aligned_features_v1
    from forecasting.model_inputs import prepare_xgboost_xy
    """Construct one exact point-in-time 14-day deployment with known labels."""
    date = pd.Timestamp(date)
    cutoff = pd.Timestamp(cutoff)
    state = eligible_origin_state(origin_features, date, first_positive)
    if state.empty:
        return None
    expanded = expand_forecast_origin_v1(state)
    expanded = expanded.loc[expanded['target_date'] <= cutoff].copy()
    if expanded.empty:
        return None

    # The frozen V1 target-aligned builder uses ONLY the historical dates
    # implied by T+h: last-year sales, 364/728-day lags and up to 4 seasonal
    # cycles. Index those specific dates instead of repeatedly merging the
    # entire multi-year sales panel for each weekly training origin.
    targets = pd.DatetimeIndex(expanded['target_date'].unique())
    sales_dates = (
        targets - pd.Timedelta(days=364)
    ).union(targets - pd.Timedelta(days=728)).union(
        targets - pd.DateOffset(years=1)
    )
    seasonal_dates = pd.DatetimeIndex([])
    for cycle in (1, 2, 3, 4):
        seasonal_dates = seasonal_dates.union(
            targets - pd.Timedelta(days=364 * cycle)
        )
    historical_regular = regular.loc[regular['date'].isin(sales_dates)]
    historical_origin = origin_features.loc[
        origin_features['forecast_origin'].isin(seasonal_dates)
    ]
    # All the required historical lookups must be dated <= forecast origin.
    if (historical_regular['date'].gt(date).any()
            or historical_origin['forecast_origin'].gt(date).any()):
        raise ValueError('Target-aligned lookups cross the origin cutoff')
    seasonal = build_target_aligned_features_v1(
        expanded[KEYS], historical_regular, historical_origin, date
    )
    expanded = expanded.merge(
        seasonal, on=KEYS, how='left', validate='one_to_one', sort=False
    )
    level = expanded['historical_seasonal_level']
    expanded['current_vs_seasonal_ratio'] = np.where(
        level.notna() & (level.abs() > 1e-8),
        expanded['rolling_mean_28d'] / level, np.nan
    ).astype('float32')

    expanded = expanded.merge(
        sales_lookup, on=KEYS, how='left', validate='many_to_one', sort=False
    )
    # Christmas dates absent from source are NEVER imputed as zero labels.
    expanded = expanded.loc[expanded['target_sales'].notna()].copy()
    if expanded.empty:
        return None

    if expanded.duplicated(['store_nbr','family','forecast_origin','horizon']).any():
        raise ValueError('Duplicate supervised training keys')
    if expanded['target_date'].max() > cutoff:
        raise ValueError('Target label after training cutoff')
    if not expanded['forecast_origin'].eq(date).all():
        raise ValueError('Unexpected historical origin')
    if not expanded['horizon'].between(1, HORIZON).all():
        raise ValueError('Invalid forecast horizon')

    X, y = prepare_xgboost_xy(expanded, schema)
    if y.isna().any() or not np.isfinite(y).all():
        raise ValueError('Invalid target labels')
    # Keep only the frozen 31 features, label, and temporal audit columns.
    output = expanded[['forecast_origin', 'target_date', 'target_sales']].copy()
    for feature in schema.features:
        output[feature] = X[feature]
    output['family'] = output['family'].astype('string')
    for name in schema.categorical_levels:
        if name != 'family':
            output[name] = X[name].astype('int16')
    return output[['forecast_origin','target_date', *schema.features, 'target_sales']]


def read_sales(path, cutoff):
    import pyarrow as pa
    import pyarrow.dataset as ds
    data = ds.dataset(str(path), format='parquet')
    dtype = data.schema.field('date').type
    if pa.types.is_date(dtype):
        arrow_cutoff = cutoff.date()
    elif pa.types.is_timestamp(dtype):
        arrow_cutoff = cutoff.to_pydatetime()
    else:
        raise TypeError(f'Unsupported curated sales date type: {dtype}')
    table = data.to_table(
        columns=['date','store_nbr','family','sales'],
        filter=ds.field('date') <= arrow_cutoff
    )
    result = table.to_pandas()
    result['date'] = pd.to_datetime(result['date'])
    result['store_nbr'] = result['store_nbr'].astype('int16')
    result['family'] = result['family'].astype('string')
    result['sales'] = result['sales'].astype('float32')
    if result.empty or result['date'].max() != cutoff:
        raise ValueError(f'Curated labels must reach cutoff {cutoff.date()}')
    if result.duplicated(['date','store_nbr','family']).any():
        raise ValueError('Duplicate curated sales keys')
    if result['sales'].isna().any() or not np.isfinite(result['sales']).all():
        raise ValueError('Invalid historical sales labels')
    if result[['store_nbr','family']].drop_duplicates().shape[0] != EXPECTED_SERIES:
        raise ValueError('Expected 1782 store-family series')
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args):
    import pyarrow as pa
    from features.origin import regularize_sales_v1, build_origin_features_v1
    from forecasting.model_inputs import load_model_input_schema
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff != pd.Timestamp('2017-08-15'):
        raise ValueError('Week 4B rehearsal only permits 2017-08-15')
    sales = read_sales(args.sales, cutoff)
    regular = regularize_sales_v1(sales, cutoff)
    origin_features = build_origin_features_v1(regular)

    per_day = regular.groupby('date',sort=True)['sales_observed'].all()
    selected = choose_rotating_weekly_origins(
        per_day[per_day].index, sales['date'].min(), cutoff
    )
    oldest = cutoff - pd.DateOffset(years=2)
    selected = selected[selected >= oldest]
    if len(selected) < 100 or len(selected) > 110:
        raise ValueError(f'Unexpected 2-year training-origin count: {len(selected)}')
    first_positive = started_store_dates(sales)
    sales_lookup = regular[['store_nbr','family','date','sales']].rename(
        columns={'date':'target_date','sales':'target_sales'}
    )

    schema = load_model_input_schema(args.contract, args.categories)
    output_dir = Path(args.output_dir)
    data_dir = output_dir / 'data'
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output directory must be empty to avoid overwrites')
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = 0
    part_manifest = []
    arrow_schema = None
    for index, date in enumerate(selected):
        part = build_labeled_origin(
            date, cutoff, regular, origin_features, sales_lookup,
            first_positive, schema
        )
        if part is None:
            raise ValueError(f'No supervised rows at origin {date.date()}')
        table = pa.Table.from_pandas(part, preserve_index=False)
        current_schema = table.schema.remove_metadata()
        if arrow_schema is None:
            arrow_schema = current_schema
        elif current_schema != arrow_schema:
            raise ValueError(f'Parquet schema drift at {date.date()}')
        dest = data_dir / f'part-{index:04d}.parquet'
        import pyarrow.parquet as pq
        pq.write_table(table,dest,compression='zstd',use_dictionary=True)
        rows += len(part)
        part_manifest.append({
            'origin':date.date().isoformat(),
            'rows':len(part),
            'filename':dest.name,
            'sha256':sha256(dest),
        })
        print(f'[{index+1}/{len(selected)}] {date.date()}: {len(part):,} rows',flush=True)
        del part,table
        gc.collect()
    if rows < 2_000_000:
        raise ValueError('Unexpectedly low training row count')

    manifest = {
        'status':'PASSED',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'cutoff':cutoff.date().isoformat(),
        'training_policy':'last_2_years',
        'training_origin_sampling':'weekly_rotating_weekday_global_anchor',
        'eligible_store_scope':'positive_store_activity_at_origin',
        'xgboost_features':list(schema.features),
        'training_origin_count':len(selected),
        'training_rows':rows,
        'first_origin':selected.min().date().isoformat(),
        'last_origin':selected.max().date().isoformat(),
        'parts':part_manifest,
        'champion_unchanged':True,
    }
    (output_dir/'dataset_manifest.json').write_text(
        json.dumps(manifest,indent=2)+'\n',encoding='utf-8'
    )
    print(json.dumps({k:v for k,v in manifest.items() if k!='parts'},indent=2))
    print('CT DATASET BUILD: PASSED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cutoff',default='2017-08-15')
    p.add_argument('--sales',type=Path,default=Path('/opt/ml/processing/input/sales'))
    p.add_argument('--contract',type=Path,default=CODE/'contract/point_forecast_model_contract_v1.json')
    p.add_argument('--categories',type=Path,default=CODE/'categories/categories_v1.json')
    p.add_argument('--output-dir',type=Path,default=Path('/opt/ml/processing/output'))
    run(p.parse_args())

if __name__=='__main__':
    main()
