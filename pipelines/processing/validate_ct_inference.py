"""CT-4: audited, label-free V1-vs-V2 inference on the SAME frozen snapshot.

Execution: a temporary SageMaker Processing Job. Never publishes forecasts,
changes a Model Registry version, trains a model, or promotes a challenger.
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

CODE = Path('/opt/ml/processing/input/code')
if CODE.is_dir():
    sys.path.insert(0, str(CODE))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

KEYS = ['store_nbr', 'family', 'forecast_origin', 'horizon', 'target_date']
CUTOFF = pd.Timestamp('2017-08-15')
FIRST_TARGET = pd.Timestamp('2017-08-16')
LAST_TARGET = pd.Timestamp('2017-08-29')
ROWS = 24_948
SERIES = 1_782
TREES = 3200
MODEL_MEMBER = 'xgboost_point_forecast_v1.json'   # legacy V1 loader contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metadata(feature_manifest: dict, training_report: dict, config: dict) -> None:
    expected_features = {
        'status': 'PASSED',
        'forecast_origin': str(CUTOFF.date()),
        'max_sales_date': str(CUTOFF.date()),
        'forecast_rows': ROWS,
        'feature_count': 31,
        'features_sha256': config['feature_sha256'],
    }
    for key, expected in expected_features.items():
        if feature_manifest.get(key) != expected:
            raise ValueError(f'Feature manifest mismatch {key}')
    expected_train = {
        'status': 'PASSED',
        'mode': 'full',
        'model_role': 'challenger_unapproved',
        'training_cutoff': str(CUTOFF.date()),
        'dataset_rows_total': 2_543_640,
        'fit_rows': 2_543_640,
        'new_label_rows': 44_550,
        'sampled_forecast_origins': 106,
        'features': 31,
        'n_estimators': TREES,
        'xgboost_version': '3.4.1',
        'evaluation_type': 'training_integrity_only',
        'has_new_out_of_sample_metric': False,
        'promotion_eligible': False,
        'model_archive_sha256': config['challenger_sha256'],
    }
    for key, expected in expected_train.items():
        if training_report.get(key) != expected:
            raise ValueError(f'Training report mismatch {key}')
    if training_report.get('input_manifest_sha256') != config['ct2_manifest_sha256']:
        raise ValueError('CT-2 input manifest provenance has changed')


def validate_features(frame: pd.DataFrame, contract: dict) -> None:
    if set(KEYS).difference(frame.columns):
        raise ValueError('Missing forecast keys')
    if len(frame) != ROWS or frame.duplicated(KEYS).any():
        raise ValueError('Unexpected forecast row population or duplicate keys')
    if set(frame.columns).intersection({'sales', 'target_sales', 'actual_sales', 'onpromotion'}):
        raise ValueError('Observed future sales or forbidden columns in prediction features')
    if set(contract['model_features']).difference(frame.columns):
        raise ValueError('Frozen 31-feature contract mismatch')
    if len(contract['model_features']) != 31:
        raise ValueError('Frozen model predictors must number 31')
    dates = pd.to_datetime(frame['target_date'])
    origins = pd.to_datetime(frame['forecast_origin'])
    if not origins.eq(CUTOFF).all() or dates.min() != FIRST_TARGET or dates.max() != LAST_TARGET:
        raise ValueError('Unexpected inference origin or target date window')
    horizons = pd.to_numeric(frame['horizon'], errors='raise')
    if set(horizons.astype(int)) != set(range(1, 15)):
        raise ValueError('Expected all 14 forecast horizons')
    if not ((dates - origins).dt.days == horizons).all():
        raise ValueError('Forecast horizon / target date mismatch')
    series = frame[['store_nbr', 'family']].drop_duplicates()
    if len(series) != SERIES:
        raise ValueError('Expected 1782 series')
    counts = frame.groupby(['store_nbr', 'family'], observed=True).size()
    if not counts.eq(14).all():
        raise ValueError('Each series must have exactly 14 forecast rows')


def load_xgb_archive(path: Path):
    from xgboost import XGBRegressor
    with tarfile.open(path, 'r:gz') as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        if len(members) != 1 or members[0].name != MODEL_MEMBER:
            raise ValueError('Unexpected XGBoost model.tar.gz format')
        content = archive.extractfile(members[0])
        if content is None:
            raise ValueError('Cannot load model JSON from archive')
        model_bytes = bytearray(content.read())
    model = XGBRegressor(enable_categorical=True, n_jobs=2)
    model.load_model(model_bytes)
    if model.get_booster().num_boosted_rounds() != TREES:
        raise ValueError('XGBoost model does not contain 3200 boosted rounds')
    return model


def make_report(frame: pd.DataFrame, champion_raw: np.ndarray,
                challenger_raw: np.ndarray, metadata: dict) -> tuple[pd.DataFrame, dict]:
    champion_raw = np.asarray(champion_raw, dtype=np.float64)
    challenger_raw = np.asarray(challenger_raw, dtype=np.float64)
    if champion_raw.shape != (ROWS,) or challenger_raw.shape != (ROWS,):
        raise ValueError('Prediction shape mismatch')
    if not (np.isfinite(champion_raw).all() and np.isfinite(challenger_raw).all()):
        raise ValueError('Non-finite predictions')
    champion = np.clip(champion_raw, 0, None)
    challenger = np.clip(challenger_raw, 0, None)
    output = frame[KEYS].copy()
    output['family'] = output['family'].astype('string')
    output['forecast_origin'] = pd.to_datetime(output['forecast_origin'])
    output['target_date'] = pd.to_datetime(output['target_date'])
    output['champion_v1_prediction'] = champion
    output['challenger_v2_prediction'] = challenger
    output['prediction_delta_v2_minus_v1'] = challenger - champion
    output = output.sort_values(KEYS).reset_index(drop=True)
    diff = challenger - champion
    report = {
        'status': 'PASSED',
        'validation_type': 'same_snapshot_unlabeled_model_comparison',
        'forecast_origin': str(CUTOFF.date()),
        'first_target_date': str(FIRST_TARGET.date()),
        'last_target_date': str(LAST_TARGET.date()),
        'forecast_rows': ROWS,
        'unique_series': SERIES,
        'feature_count': 31,
        'horizons': list(range(1, 15)),
        'model_champion_role': 'frozen_v1',
        'model_challenger_role': 'v2_unapproved',
        'evaluated_labels': 0,
        'has_new_out_of_sample_metric': False,
        'promotion_eligible': False,
        'v1_prediction_sum': float(champion.sum()),
        'v2_prediction_sum': float(challenger.sum()),
        'v1_prediction_min': float(champion.min()),
        'v2_prediction_min': float(challenger.min()),
        'v1_prediction_max': float(champion.max()),
        'v2_prediction_max': float(challenger.max()),
        'mean_absolute_prediction_delta': float(np.abs(diff).mean()),
        'mean_signed_prediction_delta': float(diff.mean()),
        'fraction_predictions_changed_gt_0_01': float(np.mean(np.abs(diff) > 0.01)),
        'explanation': 'Differences describe output changes, NOT forecast accuracy or WAPE.',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    report.update(metadata)
    return output, report


def main():
    from forecasting.model_inputs import load_model_input_schema, prepare_xgboost_X
    import xgboost as xgb

    p = argparse.ArgumentParser()
    root = Path('/opt/ml/processing/input')
    p.add_argument('--config', type=Path, default=root/'code/config/ct_inference_v1.json')
    p.add_argument('--contract', type=Path, default=root/'code/contract/point_forecast_model_contract_v1.json')
    p.add_argument('--categories', type=Path, default=root/'code/categories/categories_v1.json')
    p.add_argument('--features', type=Path, default=root/'features/features.parquet')
    p.add_argument('--manifest', type=Path, default=root/'feature-manifest/features.manifest.json')
    p.add_argument('--champion', type=Path, default=root/'champion/model.tar.gz')
    p.add_argument('--challenger', type=Path, default=root/'challenger/model.tar.gz')
    p.add_argument('--training-report', type=Path, default=root/'training-report/training_report.json')
    p.add_argument('--output-dir', type=Path, default=Path('/opt/ml/processing/output'))
    args = p.parse_args()
    if xgb.__version__ != '3.4.1':
        raise RuntimeError(f'Expected XGBoost 3.4.1; found {xgb.__version__}')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    contract = json.loads(args.contract.read_text(encoding='utf-8'))
    feature_manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    training_report = json.loads(args.training_report.read_text(encoding='utf-8'))
    validate_metadata(feature_manifest, training_report, config)
    expected_hashes = [
        ('features', args.features, config['feature_sha256']),
        ('champion', args.champion, config['champion_sha256']),
        ('challenger', args.challenger, config['challenger_sha256']),
    ]
    for name, path, wanted in expected_hashes:
        actual = sha256_file(path)
        if actual != wanted:
            raise ValueError(f'{name} SHA-256 mismatch: {actual}')
    if sha256_file(args.contract) != training_report['frozen_contract_sha256']:
        raise ValueError('Challenger trained using a different V1 contract')
    frame = pd.read_parquet(args.features)
    validate_features(frame, contract)
    schema = load_model_input_schema(args.contract, args.categories)
    X = prepare_xgboost_X(frame, schema)
    if X.shape != (ROWS, 31):
        raise ValueError('XGBoost model matrix invalid')
    del frame
    # Re-read just keys (and frame) to avoid retaining an unnecessary dataframe
    frame = pd.read_parquet(args.features, columns=KEYS)
    print('Validated frozen feature snapshot and both model archives.', flush=True)
    champion = load_xgb_archive(args.champion)
    champion_raw = champion.predict(X)
    del champion
    print('Champion V1 predictions completed.', flush=True)
    challenger = load_xgb_archive(args.challenger)
    challenger_raw = challenger.predict(X)
    del challenger
    print('Challenger V2 predictions completed.', flush=True)
    output, report = make_report(frame, champion_raw, challenger_raw, {
        'champion_model_sha256': config['champion_sha256'],
        'challenger_model_sha256': config['challenger_sha256'],
        'features_sha256': config['feature_sha256'],
        'training_report_sha256': sha256_file(args.training_report),
        'xgboost_version': xgb.__version__,
        'python_version': platform.python_version(),
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError('Output directory is not empty')
    output_path = args.output_dir/'challenger_comparison.parquet'
    output.to_parquet(output_path, index=False)
    report['comparison_parquet_sha256'] = sha256_file(output_path)
    report_path = args.output_dir/'inference_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('CT-4 INFERENCE AUDIT: PASSED', flush=True)


if __name__ == '__main__':
    main()
