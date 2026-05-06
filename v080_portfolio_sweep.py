"""
v0.8.0 portfolio HP sweep — TOP_N × HYST_EXIT × REBAL_DAYS.

Score panel은 model output. Rebal/top/hyst 변경해도 score 안 바뀜.
→ LGBM walk-forward 1번만 돌리고, 27 portfolio sim 벡터화.

Grid:
  REBAL: [1, 5, 21]   (daily, weekly, monthly)
  TOP_N: [5, 10, 20]
  HYST:  [20, 40, 80]
  = 27 cells
"""
import json
import sys
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
    COST_ONEWAY, START_DATE, TARGET_HORIZONS,
)

PANEL_DIR = Path('/home/dlfnek/stock_lab/data/panels')
SHARADAR_DIR = Path('/home/dlfnek/stock_lab/data/sharadar')
OUT_DIR = Path('/home/dlfnek/stock_lab/results')


def build_score_panel(feat_3d, target_arr, mask_arr, dates, tickers):
    """v0.8.0 winner LGBM walk-forward → score panel."""
    print('Building score panel (LGBM d=4 n=200 lr=0.1)...', flush=True)
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
            model = make_model('LightGBM', {
                'max_depth': 4, 'n_estimators': 200, 'learning_rate': 0.1,
            })
            try:
                model.fit(Xs, y)
            except Exception as e:
                print(f'  fit fail at {i}: {e}', flush=True)
                continue
            last_train_idx = i
            print(f'  {dates[i].date()}: trained', flush=True)
        ti, sc = predict_at_idx(i, feat_3d, mask_arr, model, scaler, medians)
        if ti is not None:
            score_arr[i, ti] = sc

    return pd.DataFrame(score_arr, index=dates, columns=tickers)


def run_portfolio(score_panel, close, spy_ret, top_n, hyst, rebal_days, label):
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

    p_g = float(cagr(pg_gross))
    p_n = float(cagr(pg_net))
    spy_c = float(cagr(spy))
    sharpe = float(pg_net.mean() / pg_net.std() * np.sqrt(252)) if pg_net.std() > 0 else 0
    cum = (1 + pg_net).cumprod()
    mdd = float((cum / cum.cummax() - 1).min())
    annual_to = float(daily_oneway.sum() / (len(daily_oneway) / 252))

    return {
        'label': label,
        'top_n': top_n, 'hyst_exit': hyst, 'rebal_days': rebal_days,
        'gross_alpha': float(p_g - spy_c),
        'net_alpha': float(p_n - spy_c),
        'sharpe': sharpe, 'mdd': mdd,
        'annual_turnover': annual_to,
    }


def main():
    print('=== v0.8.0 Portfolio HP Sweep (TOP × HYST × REBAL) ===', flush=True)
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

    # Build score panel ONCE
    t0 = time.time()
    score_panel = build_score_panel(feat_3d, target_arr, mask_arr, dates, tickers)
    print(f'Score panel built: {(time.time()-t0):.0f}s', flush=True)

    # 27-cell sweep
    cells = []
    for rebal in [1, 5, 21]:
        for top in [5, 10, 20]:
            for hyst in [20, 40, 80]:
                cells.append((top, hyst, rebal))

    print(f'\nRunning {len(cells)} portfolio sims...', flush=True)
    results = []
    for i, (top, hyst, rebal) in enumerate(cells, 1):
        rebal_str = {1: 'daily', 5: 'weekly', 21: 'monthly'}[rebal]
        label = f'TOP={top} HYST={hyst} REBAL={rebal_str}'
        t0 = time.time()
        res = run_portfolio(score_panel, close, spy_ret, top, hyst, rebal, label)
        elapsed = time.time() - t0
        print(f'  [{i:2}/{len(cells)}] {label:<35} '
              f'α_g={res["gross_alpha"]*100:+5.2f}%p α_n={res["net_alpha"]*100:+5.2f}%p '
              f'Sh={res["sharpe"]:.2f} MDD={res["mdd"]*100:.0f}% TO={res["annual_turnover"]:.1f}x '
              f'({elapsed:.0f}s)', flush=True)
        results.append(res)

    with open(OUT_DIR / 'v080_portfolio_sweep.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    print('\n' + '=' * 100, flush=True)
    print('PORTFOLIO HP SWEEP SUMMARY', flush=True)
    print('=' * 100, flush=True)
    df = pd.DataFrame(results).sort_values('net_alpha', ascending=False)
    print(df[['label', 'gross_alpha', 'net_alpha', 'sharpe', 'mdd', 'annual_turnover']].to_string(),
          flush=True)
    print('\nTop 5 by net α:', flush=True)
    print(df.head(5)[['label', 'gross_alpha', 'net_alpha', 'sharpe']].to_string(), flush=True)
    print('\nTop 5 by Sharpe:', flush=True)
    print(df.sort_values('sharpe', ascending=False).head(5)[
          ['label', 'gross_alpha', 'net_alpha', 'sharpe']].to_string(), flush=True)


if __name__ == '__main__':
    main()
