"""
Verify target leakage fix: rerun v5 baseline (3-feat Ridge + 7y + Weekly + Top-20)
and compare with prior result.

Prior (with leakage): alpha +34.56%p, Sharpe 1.79
Expected (after fix):  alpha slightly lower (-0.5~1.5%p)

Output: results/leakage_fix_verify.txt
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core
import ml_model
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
TOP_N = 20
REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
FEATURES = ['lowvol', 'rsi', 'volsurge']


def load_spy_returns():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last().pct_change()


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def backtest(close, score_wide):
    score = score_wide.where(close.notna())
    hp = {'top_n': TOP_N, 'rebal_days': REBAL_DAYS, 'hysteresis': 0,
          'cost_oneway': COST_ONEWAY}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * COST_ONEWAY
    return port_gross - daily_cost


def run_one_window(close, vol, test_year, hp, spy_ret):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid) < 100:
        return None
    close_sub = close[valid]
    vol_sub = vol[valid]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    train_long = train_long.dropna(subset=FEATURES)
    test_long = test_long.dropna(subset=FEATURES)
    if len(train_long) < 1000 or len(test_long) == 0:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(train_long[FEATURES].values)
    model = Ridge(alpha=1.0)
    model.fit(Xs, train_long['target'].values)
    preds = model.predict(scaler.transform(test_long[FEATURES].values))

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret = backtest(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'year': test_year,
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'excess': excess, 'train_rows': len(train_long),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Target leakage fix verification")
    w(f"  v5 baseline: 3-feat Ridge + 7y + Weekly + Top-20")
    w(f"  Prior (with leakage): alpha +34.56%p, Sharpe 1.79")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy_returns()
    hp = dict(ml_model.ML_HP_DEFAULT)
    hp['feature_names'] = FEATURES

    rows = []
    for y in range(1995, 2026):
        r = run_one_window(close, vol, y, hp, spy_ret)
        if r is not None:
            rows.append(r)
            print(f"  Year {y}: alpha {r['excess']:+.2f}%p, Sharpe {r['sharpe']:.2f}, "
                  f"train rows {r['train_rows']:,}", flush=True)

    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    avg_alpha = sum(excesses) / len(excesses) if excesses else 0
    avg_sharpe = sum(r['sharpe'] for r in rows) / len(rows) if rows else 0
    avg_mdd = sum(r['mdd'] for r in rows) / len(rows) if rows else 0
    avg_cagr = sum(r['cagr'] for r in rows) / len(rows) if rows else 0
    win = sum(1 for e in excesses if e > 0)
    t = t_stat(excesses)
    avg_train_rows = sum(r['train_rows'] for r in rows) / len(rows) if rows else 0

    w(f"\n{'='*100}")
    w(f"## After leakage fix (n={len(rows)} windows)")
    w(f"  Avg CAGR:     {avg_cagr*100:+.2f}%")
    w(f"  Avg Sharpe:   {avg_sharpe:.2f}")
    w(f"  Avg MDD:      {avg_mdd*100:.2f}%")
    w(f"  Avg alpha:    {avg_alpha:+.2f}%p")
    w(f"  Win rate:     {win}/{len(rows)} ({win/len(rows)*100:.1f}%)")
    w(f"  t-stat:       {t:.2f}")
    w(f"  Avg train rows: {avg_train_rows:,.0f}")
    w(f"\n## Comparison with prior (leakage)")
    w(f"  Alpha:  prior +34.56%p → current {avg_alpha:+.2f}%p (Δ {avg_alpha-34.56:+.2f}%p)")
    w(f"  Sharpe: prior 1.79     → current {avg_sharpe:.2f} (Δ {avg_sharpe-1.79:+.2f})")
    if abs(avg_alpha - 34.56) < 0.5:
        w(f"  → 차이 미미. Leakage 영향 ~0.5%p 미만 (예상보다 작음)")
    elif abs(avg_alpha - 34.56) < 2:
        w(f"  → 예상 범위 (0.5-1.5%p). Leakage 영향 확인됨")
    else:
        w(f"  → 큰 차이! Leakage 영향 컸음. README 정정 필수")

    out_path = os.path.join(OUTPUT_DIR, 'leakage_fix_verify.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
