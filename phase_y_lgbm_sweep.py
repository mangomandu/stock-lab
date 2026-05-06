"""
LGBM HP sweep — Phase Y v3 winner (d=4 n=200) 주변 + 더 깊이.

Sweep:
  depth: 3, 4, 5, 6
  n_est: 100, 200, 300, 500
  learning_rate: 0.05, 0.1
  min_child_samples: 20 (default, fixed)

Total: 4 × 4 × 2 = 32 cells
Time est: ~32 × 220s = ~120 min (2 hours)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import core
from phase1_compare import build_sp500_mask, load_spy_returns
from phase_y_pure_ml_v2 import (
    FEATURES, load_features_3d, make_target_arr,
    stack_train_arrays, predict_at_idx, make_model,
    TOP_N, HYST_EXIT, COST_ONEWAY, REBAL_DAYS, START_DATE, TARGET_HORIZONS,
)

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/results')


def make_lgbm(params):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        learning_rate=params.get('learning_rate', 0.1),
        num_leaves=2 ** params['max_depth'],
        min_child_samples=params.get('min_child_samples', 20),
        random_state=42,
        verbose=-1,
    )


def run_cell(feat_3d, target_arr, mask_arr, dates, tickers, close, spy_ret, params, label):
    print(f'\n=== {label} ===', flush=True)
    t0 = time.time()

    valid_dates = dates[(dates >= pd.Timestamp(START_DATE))]
    valid_idx = np.array([dates.get_loc(d) for d in valid_dates])
    n_dates = len(dates)
    score_arr = np.full((n_dates, len(tickers)), np.nan, dtype=np.float32)

    last_train_idx = -252
    model, scaler, medians = None, None, None
    n_retrains = 0

    for i in valid_idx:
        train_start = max(0, i - 10 * 252)
        if i - train_start < 252:
            continue
        if model is None or (i - last_train_idx) >= 252:
            buf = max(TARGET_HORIZONS)
            tr = np.arange(train_start, max(train_start, i - buf))
            X, y = stack_train_arrays(feat_3d, target_arr, mask_arr, tr)
            if len(X) < 1000:
                continue
            medians = np.median(X, axis=0)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = make_lgbm(params)
            try:
                model.fit(Xs, y)
            except Exception as e:
                print(f'  fit fail at idx {i}: {e}', flush=True)
                continue
            last_train_idx = i
            n_retrains += 1
        ti, sc = predict_at_idx(i, feat_3d, mask_arr, model, scaler, medians)
        if ti is not None:
            score_arr[i, ti] = sc

    score_panel = pd.DataFrame(score_arr, index=dates, columns=tickers)
    hp_bt = {'top_n': TOP_N, 'rebal_days': REBAL_DAYS,
             'hysteresis': HYST_EXIT, 'cost_oneway': COST_ONEWAY}
    in_top = core.build_holdings(score_panel, hp_bt)
    wd = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = wd.shift(1).fillna(0)
    sr = close.pct_change().fillna(0)
    pg_gross = (held * sr).sum(axis=1)
    daily_oneway = held.diff().abs().sum(axis=1).fillna(0)
    pg_net = pg_gross - daily_oneway * COST_ONEWAY

    pg_net = pg_net[pg_net.index >= pd.Timestamp(START_DATE)].dropna()
    pg_gross = pg_gross[pg_gross.index >= pd.Timestamp(START_DATE)].dropna()
    daily_oneway = daily_oneway[daily_oneway.index >= pd.Timestamp(START_DATE)]
    spy = spy_ret.reindex(pg_net.index).fillna(0)

    def cagr(s):
        return (1 + s).prod() ** (252 / len(s)) - 1 if len(s) > 0 else 0

    p_gross = float(cagr(pg_gross))
    p_net = float(cagr(pg_net))
    spy_c = float(cagr(spy))
    sharpe = float(pg_net.mean() / pg_net.std() * np.sqrt(252)) if pg_net.std() > 0 else 0
    cum = (1 + pg_net).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    annual_to = float(daily_oneway.sum() / (len(daily_oneway) / 252))

    fwd_21 = close.shift(-21) / close - 1
    ic_per_day = []
    for d in score_panel.dropna(how='all').index:
        s = score_panel.loc[d].dropna()
        r = fwd_21.loc[d].reindex(s.index).dropna() if d in fwd_21.index else pd.Series(dtype=float)
        common = s.index.intersection(r.index)
        if len(common) < 20:
            continue
        sv, rv = s[common].values, r[common].values
        if np.std(sv) == 0 or np.std(rv) == 0:
            continue
        ic, _ = spearmanr(sv, rv)
        if not np.isnan(ic):
            ic_per_day.append(ic)
    ic_mean = float(np.mean(ic_per_day)) if ic_per_day else 0
    ic_ir = ic_mean / np.std(ic_per_day) * np.sqrt(252) if ic_per_day and np.std(ic_per_day) > 0 else 0

    elapsed = time.time() - t0
    print(f'  α_g={(p_gross - spy_c)*100:+.2f}%p  α_n={(p_net - spy_c)*100:+.2f}%p  '
          f'Sharpe={sharpe:.2f}  IC={ic_mean:+.4f}  TO={annual_to:.1f}x  ({elapsed:.0f}s)',
          flush=True)

    return {
        'label': label, 'params': params,
        'gross_alpha': float(p_gross - spy_c),
        'net_alpha': float(p_net - spy_c),
        'sharpe': sharpe, 'mdd': mdd,
        'ic_mean': ic_mean, 'ic_ir': float(ic_ir),
        'annual_turnover': annual_to,
        'n_retrains': n_retrains,
        'elapsed_sec': elapsed,
    }


def main():
    print('=== LGBM HP Sweep ===', flush=True)
    print('Setup...', flush=True)
    close = pd.read_parquet(PANEL_DIR / 'sp500_close.parquet')
    sp500_changes = pd.read_parquet(SHARADAR_DIR / 'SP500.parquet')
    sp500_mask = build_sp500_mask(close, sp500_changes, list(close.columns))
    close = close.where(sp500_mask)
    dates = close.index
    tickers = list(close.columns)
    spy_ret = load_spy_returns(close.index.min(), close.index.max())
    feat_3d = load_features_3d(dates, tickers)
    target_arr = make_target_arr(close)
    mask_arr = sp500_mask.values

    cells = []
    for depth in [3, 4, 5, 6]:
        for n_est in [100, 200, 300, 500]:
            for lr in [0.05, 0.1]:
                params = {'max_depth': depth, 'n_estimators': n_est, 'learning_rate': lr}
                label = f'LGBM d={depth} n={n_est} lr={lr}'
                cells.append((params, label))

    print(f'\nTotal cells: {len(cells)}', flush=True)
    results = []
    for i, (params, label) in enumerate(cells, 1):
        print(f'\n[{i}/{len(cells)}]', flush=True)
        try:
            res = run_cell(feat_3d, target_arr, mask_arr, dates, tickers, close, spy_ret,
                           params, label)
            results.append(res)
            with open(OUT_DIR / 'lgbm_sweep_results.json', 'w') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            import traceback
            traceback.print_exc()

    print('\n' + '=' * 90, flush=True)
    print('LGBM SWEEP SUMMARY', flush=True)
    print('=' * 90, flush=True)
    df = pd.DataFrame(results).sort_values('net_alpha', ascending=False)
    print(df[['label', 'gross_alpha', 'net_alpha', 'sharpe', 'mdd', 'ic_mean']].to_string(),
          flush=True)
    print('\nTop 5:', flush=True)
    print(df.head(5)[['label', 'gross_alpha', 'net_alpha', 'sharpe', 'ic_mean']].to_string(),
          flush=True)


if __name__ == '__main__':
    main()
