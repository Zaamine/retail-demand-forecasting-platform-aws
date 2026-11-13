"""Audited XGBoost CT challenger fit using the frozen Notebook 05 contract.

SageMaker Processing container; NO promotion, NO Model Registry mutation.
A `smoke` run produces only a technical report; `full` creates a V2 candidate
archive containing a V1-format JSON model for legacy batch inference tooling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path('/opt/ml/processing/input/code')
if CODE_ROOT.is_dir():
    sys.path.insert(0, str(CODE_ROOT))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

CUTOFF = pd.Timestamp('2017-08-15')
EXPECTED_ROWS = 2_543_640
EXPECTED_ORIGINS = 106
EXPECTED_FEATURES = 31
MODEL_JSON_NAME = 'xgboost_point_forecast_v1.json'  # frozen inference format


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(manifest: dict, contract: dict, expected_rows=EXPECTED_ROWS) -> None:
    """Fail closed if dataset provenance is not the validated CT-2 run."""
    required = {
        'status': 'PASSED',
        'cutoff': CUTOFF.date().isoformat(),
        'training_policy': 'last_2_years',
        'training_origin_sampling': 'weekly_rotating_weekday_global_anchor',
        'eligible_store_scope': 'positive_store_activity_at_origin',
        'training_origin_count': EXPECTED_ORIGINS,
        'training_rows': expected_rows,
    }
    for k, value in required.items():
        if manifest.get(k) != value:
            raise ValueError(f'CT-2 manifest mismatch {k}: {manifest.get(k)!r} != {value!r}')
    if manifest.get('xgboost_features') != contract['model_features']:
        raise ValueError('Predictor order differs from frozen V1 contract')
    if len(contract['model_features']) != EXPECTED_FEATURES:
        raise ValueError('Frozen model contract must have exactly 31 predictors')
    parts = manifest.get('parts', [])
    if len(parts) != EXPECTED_ORIGINS or len({p['filename'] for p in parts}) != len(parts):
        raise ValueError('Unexpected, missing, or duplicated CT-2 part inventory')
    if manifest['training_rows'] != sum(int(p['rows']) for p in parts):
        raise ValueError('CT-2 part row count mismatch')
    if manifest.get('first_origin') != '2015-08-16' or manifest.get('last_origin') != '2017-08-14':
        raise ValueError('Unexpected temporal training origin population')
    ordered_dates = [pd.Timestamp(p['origin']) for p in parts]
    if ordered_dates != sorted(ordered_dates) or len(set(ordered_dates)) != EXPECTED_ORIGINS:
        raise ValueError('CT-2 part origins are not uniquely chronological')


def read_and_audit_dataset(data_root: Path, manifest: dict, contract: dict) -> pd.DataFrame:
    import pyarrow.parquet as pq

    expected_paths = {p['filename'] for p in manifest['parts']}
    files = {p.name for p in (data_root / 'data').glob('*.parquet')}
    if files != expected_paths:
        raise ValueError('Physical Parquet files differ from manifest inventory')
    required = {'forecast_origin', 'target_date', 'target_sales', *contract['model_features']}
    parts = []
    for i, part in enumerate(manifest['parts']):
        f = data_root / 'data' / part['filename']
        if checksum(f) != part['sha256']:
            raise ValueError(f'Parquet checksum mismatch: {f.name}')
        df = pq.read_table(f).to_pandas()
        if len(df) != part['rows']:
            raise ValueError(f'Part row count mismatch: {f.name}')
        if set(df.columns) != required:
            raise ValueError(f'Parquet schema columns unexpected: {f.name}')
        if not pd.to_datetime(df['forecast_origin']).eq(pd.Timestamp(part['origin'])).all():
            raise ValueError(f'Wrong forecast origin: {f.name}')
        parts.append(df)
        if (i + 1) % 20 == 0 or i + 1 == len(manifest['parts']):
            print(f'Audited Parquet files: {i + 1}/{len(manifest["parts"])}', flush=True)
    data = pd.concat(parts, ignore_index=True)
    del parts
    if len(data) != manifest['training_rows']:
        raise ValueError('Total training row count mismatch')
    data['forecast_origin'] = pd.to_datetime(data['forecast_origin'])
    data['target_date'] = pd.to_datetime(data['target_date'])
    if (data['target_date'].gt(CUTOFF).any()
            or data['forecast_origin'].ge(data['target_date']).any()
            or data['forecast_origin'].lt(CUTOFF - pd.DateOffset(years=2)).any()):
        raise ValueError('Time leakage / invalid 2-year training policy detected')
    if data.duplicated(['forecast_origin', 'store_nbr', 'family', 'horizon']).any():
        raise ValueError('Duplicate supervised business key')
    if data['target_sales'].isna().any() or not np.isfinite(data['target_sales'].to_numpy()).all():
        raise ValueError('Missing or nonfinite target labels')
    if data['target_sales'].lt(0).any():
        raise ValueError('Negative sales labels')
    if not data['horizon'].between(1, 14).all():
        raise ValueError('Unexpected forecast horizon')
    if not ((data['target_date'] - data['forecast_origin']).dt.days == data['horizon']).all():
        raise ValueError('Target date / horizon mismatch')
    if data.loc[data['target_date'] > pd.Timestamp('2017-08-01')].empty:
        raise ValueError('No newly labeled data entered challenger training')
    return data


def make_callback(step=200):
    import xgboost as xgb
    class LogProgress(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if epoch == 0 or (epoch + 1) % step == 0:
                print(f'XGBoost boosting rounds completed: {epoch + 1}', flush=True)
            return False
    return LogProgress()


def model_config_from_contract(contract: dict, *, n_estimators: int, n_jobs: int) -> dict:
    allowed = ('objective', 'max_depth', 'learning_rate', 'min_child_weight',
               'subsample', 'colsample_bytree')
    result = {key: contract[key] for key in allowed}
    if contract.get('n_estimators') != 3200 or contract.get('training_policy') != 'last_2_years':
        raise ValueError('Frozen V1 training specification has changed')
    result.update(n_estimators=n_estimators, tree_method='hist', enable_categorical=True,
                  random_state=42, n_jobs=n_jobs, verbosity=0)
    return result


def run(args) -> None:
    import xgboost as xgb
    from xgboost import XGBRegressor
    from forecasting.model_inputs import load_model_input_schema, prepare_xgboost_xy

    if xgb.__version__ != '3.4.1':
        raise RuntimeError(f'Expected XGBoost 3.4.1; got {xgb.__version__}')
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    manifest_bytes = args.dataset.joinpath('dataset_manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    check_manifest(manifest, contract)
    schema = load_model_input_schema(args.contract, args.categories)
    data = read_and_audit_dataset(args.dataset, manifest, contract)
    rows_new = int(data['target_date'].gt(pd.Timestamp('2017-08-01')).sum())
    if rows_new != 44_550:
        raise ValueError(f'Unexpected new-label supervised rows: {rows_new}')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    training = data
    trees = 3200
    if mode == 'smoke':
        trees = 25
        # Audit entire input, then fit a bounded sample across all origins.
        training = data.groupby('forecast_origin', group_keys=False).head(500).copy()
    X, y = prepare_xgboost_xy(training, schema)
    if y.isna().any() or not np.isfinite(y.to_numpy()).all():
        raise ValueError('Model input labels invalid')
    print(json.dumps({
        'mode': mode, 'rows_fit': len(X), 'full_dataset_rows': len(data),
        'new_label_rows': rows_new, 'trees': trees, 'features': X.shape[1],
        'xgboost_version': xgb.__version__, 'cutoff': manifest['cutoff'],
    }, indent=2), flush=True)
    del data, training

    params = model_config_from_contract(contract, n_estimators=trees, n_jobs=args.n_jobs)
    params['callbacks'] = [make_callback()]
    model = XGBRegressor(**params)
    print('Fitting challenger (no validation_set, no early stopping)...', flush=True)
    model.fit(X, y)
    if model.get_booster().num_boosted_rounds() != trees:
        raise ValueError('Unexpected number of XGBoost boosting rounds')
    sample_X = X.iloc[:min(256, len(X))]
    pred = model.predict(sample_X)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite predictions on technical inference sample')

    report = {
        'status': 'PASSED',
        'mode': mode,
        'model_role': 'challenger_unapproved',
        'training_cutoff': CUTOFF.date().isoformat(),
        'dataset_rows_total': EXPECTED_ROWS,
        'fit_rows': len(X),
        'new_label_rows': rows_new,
        'sampled_forecast_origins': EXPECTED_ORIGINS,
        'features': EXPECTED_FEATURES,
        'n_estimators': trees,
        'xgboost_version': xgb.__version__,
        'python_version': platform.python_version(),
        'input_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'frozen_contract_sha256': checksum(args.contract),
        'training_policy': 'last_2_years',
        'configuration': {key: contract[key] for key in (
            'objective', 'max_depth', 'learning_rate', 'min_child_weight',
            'subsample', 'colsample_bytree')},
        'technical_inference_sample_rows': len(pred),
        'technical_inference_nonfinite_count': 0,
        'evaluation_type': 'training_integrity_only',
        'has_new_out_of_sample_metric': False,
        'promotion_eligible': False,
        'champion_v1_unchanged': True,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    if mode == 'full':
        raw_path = args.output_dir / MODEL_JSON_NAME
        model.save_model(str(raw_path))
        print('Saved raw XGBoost JSON; preparing compatible tar.gz...', flush=True)
        archive_path = args.output_dir / 'model.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as archive:
            archive.add(raw_path, arcname=MODEL_JSON_NAME)
        # Verify compatibility with V1 inference loader, without unpacking paths.
        with tarfile.open(archive_path, 'r:gz') as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            if len(members) != 1 or members[0].name != MODEL_JSON_NAME:
                raise ValueError('Model tarball violates frozen batch inference contract')
        report.update({
            'artifact_format': 'tar.gz',
            'model_archive_name': 'model.tar.gz',
            'model_archive_size_bytes': archive_path.stat().st_size,
            'model_archive_sha256': checksum(archive_path),
            'raw_model_json_sha256': checksum(raw_path),
        })
        raw_path.unlink()  # raw JSON is inside model.tar.gz to avoid duplicate S3 storage
    (args.output_dir/'training_report.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(report, indent=2), flush=True)
    print('CT CHALLENGER TRAINING: PASSED', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('/opt/ml/processing/input/dataset'))
    p.add_argument('--contract', type=Path,
                   default=CODE_ROOT/'contract/point_forecast_model_contract_v1.json')
    p.add_argument('--categories', type=Path, default=CODE_ROOT/'categories/categories_v1.json')
    p.add_argument('--output-dir', type=Path, default=Path('/opt/ml/processing/output'))
    p.add_argument('--mode', choices=('smoke', 'full'), default='smoke')
    p.add_argument('--n-jobs', type=int, choices=(1, 2, 3, 4), default=2)
    run(p.parse_args())

if __name__ == '__main__':
    main()
