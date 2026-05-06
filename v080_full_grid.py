"""
v0.8.0 Full Grid — 24 models × 27 portfolio HP = 648 cells.

Architecture:
  1. Build 24 score panels (model walk-forward) — ~70 min
  2. For each score panel, run 27 portfolio sims — ~11 min
  3. Save all 648 results

Total: ~1.5 hours.
"""
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import core
from phase1_compare import build_sp500_mask, load_spy_returns
from phase_y_pure_ml_v2 import (
    FEATURES, load_features_3d, make_target_arr,
    stack_train_arrays, predict_at_idx, make_model,
    COST_ONEWAY, START_DATE, TARGET_HORIZONS,
)

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/results')
SCORE_DIR = OUT_DIR / 'v080_score_panels'
SCORE_DIR.mkdir(exist_ok=True)


def build_score_panel(feat_3d, target_arr, mask_arr, dates, tickers,
                      model_type, params, cache_path=None):
    """Walk-forward 1번 돌려서 score_panel 생성. cache_path 있으면 disk에서 load/save."""
    if cache_path and Path(cache_path).exists():
        print(f'  [CACHE] {cache_path.name}', flush=True)
        return pd.read_parquet(cache_path)

    valid_dates = dates[(dates >= pd.Timestamp(START_DATE))]
    valid_idx = np.array([dates.get_loc(d) for d in valid_dates])
    n_dates = len(dates)
    score_arr = np.full((n_dates, len(tickers)), np.nan, dtype=np.float32)

    last_train_idx = -252
    model, scaler, medians = None, None, None

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
            model = make_model(model_type, params)
            try:
                model.fit(Xs, y)
            except Exception as e:
                print(f'    fit fail at {dates[i].date()}: {e}', flush=True)
                continue
            last_train_idx = i
        ti, sc = predict_at_idx(i, feat_3d, mask_arr, model, scaler, medians)
        if ti is not None:
            score_arr[i, ti] = sc

    score_panel = pd.DataFrame(score_arr, index=dates, columns=tickers)
    if cache_path:
        score_panel.to_parquet(cache_path)
    return score_panel


def run_portfolio(score_panel, close, spy_ret, top_n, hyst, rebal_days):
    hp_bt = {'top_n': top_n, 'rebal_days': rebal_days,
             'hysteresis': hyst, 'cost_oneway': COST_ONEWAY}
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

    p_g, p_n = float(cagr(pg_gross)), float(cagr(pg_net))
    spy_c = float(cagr(spy))
    sharpe = float(pg_net.mean() / pg_net.std() * np.sqrt(252)) if pg_net.std() > 0 else 0
    cum = (1 + pg_net).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    annual_to = float(daily_oneway.sum() / (len(daily_oneway) / 252))

    return {
        'gross_alpha': float(p_g - spy_c),
        'net_alpha': float(p_n - spy_c),
        'sharpe': sharpe,
        'mdd': mdd,
        'annual_turnover': annual_to,
    }


def main():
    print('=== v0.8.0 FULL GRID — 24 models × 27 portfolio = 648 cells ===', flush=True)
    print('Setup...', flush=True)
    t_setup = time.time()
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
    print(f'Setup: {(time.time()-t_setup):.0f}s', flush=True)

    # Define all 24 models
    models = []
    for alpha in [0.1, 1, 10, 100, 1000]:
        models.append(('Ridge', {'alpha': alpha}, f'Ridge_a{alpha}'))
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        models.append(('Lasso', {'alpha': alpha}, f'Lasso_a{alpha}'))
    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            models.append(('ElasticNet',
                           {'alpha': alpha, 'l1_ratio': l1_ratio},
                           f'EN_a{alpha}_l{l1_ratio}'))
    for max_depth in [3, 4]:
        for n_est in [50, 100, 200]:
            models.append(('LightGBM',
                           {'max_depth': max_depth, 'n_estimators': n_est},
                           f'LGBM_d{max_depth}_n{n_est}'))

    # Portfolio HP grid (27)
    portfolio_grid = []
    for rebal in [1, 5, 21]:
        for top in [5, 10, 20]:
            for hyst in [20, 40, 80]:
                portfolio_grid.append((top, hyst, rebal))

    print(f'\n{len(models)} models × {len(portfolio_grid)} portfolio HPs = '
          f'{len(models) * len(portfolio_grid)} cells', flush=True)

    all_results = []
    t_total = time.time()

    for mi, (model_type, params, slug) in enumerate(models, 1):
        t_model = time.time()
        cache_path = SCORE_DIR / f'{slug}.parquet'
        print(f'\n[Model {mi:2}/{len(models)}] {slug}', flush=True)
        score_panel = build_score_panel(feat_3d, target_arr, mask_arr,
                                         dates, tickers, model_type, params, cache_path)
        t_score = time.time() - t_model

        for top, hyst, rebal in portfolio_grid:
            res = run_portfolio(score_panel, close, spy_ret, top, hyst, rebal)
            res.update({
                'model_slug': slug,
                'model_type': model_type,
                'model_params': params,
                'top_n': top, 'hyst_exit': hyst, 'rebal_days': rebal,
            })
            all_results.append(res)
        del score_panel
        gc.collect()

        # Save incremental
        with open(OUT_DIR / 'v080_full_grid_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

        elapsed = time.time() - t_model
        rate = (time.time() - t_total) / mi
        eta = rate * (len(models) - mi) / 60
        print(f'  Done: score {t_score:.0f}s + 27 sims = {elapsed:.0f}s. '
              f'ETA: {eta:.0f} min remaining', flush=True)

    # Summary
    print('\n' + '=' * 100, flush=True)
    print('FULL GRID SUMMARY', flush=True)
    print('=' * 100, flush=True)
    df = pd.DataFrame(all_results).sort_values('net_alpha', ascending=False)

    print('\nTop 10 by net α:', flush=True)
    print(df.head(10)[['model_slug', 'top_n', 'hyst_exit', 'rebal_days',
                        'gross_alpha', 'net_alpha', 'sharpe', 'mdd',
                        'annual_turnover']].to_string(), flush=True)

    print('\nTop 10 by Sharpe:', flush=True)
    print(df.sort_values('sharpe', ascending=False).head(10)[
        ['model_slug', 'top_n', 'hyst_exit', 'rebal_days',
         'gross_alpha', 'net_alpha', 'sharpe']].to_string(), flush=True)

    # Per-model best
    print('\nBest portfolio HP per model (by net α):', flush=True)
    best_per = df.loc[df.groupby('model_slug')['net_alpha'].idxmax()]
    best_per = best_per.sort_values('net_alpha', ascending=False)
    print(best_per[['model_slug', 'top_n', 'hyst_exit', 'rebal_days',
                     'net_alpha', 'sharpe']].to_string(), flush=True)

    print(f'\nTotal time: {(time.time()-t_total)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
