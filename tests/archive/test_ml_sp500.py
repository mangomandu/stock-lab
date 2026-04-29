"""
ML walk-forward on S&P 500 universe.

Same model as before, but:
  - Universe: ~518 tickers (S&P 500 + 17 ETFs)
  - Walk-forward: extended to 31 windows (1995-2025)
  - Same Biweekly Top-20 rule, ML scoring

Output: results/ml_sp500_walkforward.txt
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import Counter

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 20
REBAL_DAYS = 10
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100  # require at least this many alive tickers


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


def load_spy_benchmark():
    """SPY data already in master_sp500."""
    spy_path = os.path.join(DATA_DIR, 'SPY.csv')
    df = pd.read_csv(spy_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    daily = df.groupby('Date')['Close'].last()
    return daily.pct_change()


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def backtest_with_score(close, score_wide, top_n=TOP_N, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    hp = {'top_n': top_n, 'rebal_days': rebal_days, 'hysteresis': 0,
          'cost_oneway': cost}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, daily_cost


def run_one_window(close, vol, test_year, hp, qqq_ret, spy_ret):
    train_start = pd.Timestamp(f'{test_year - 5}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    # Restrict to tickers with enough train data
    valid_tickers = close.columns[close[train_mask].notna().sum() >= 252]
    if len(valid_tickers) < MIN_UNIVERSE_SIZE:
        return None

    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    if len(train_long) < 1000:
        return None

    model = ml_model.train_model(train_long, hp)
    if model is None:
        return None

    score_long = ml_model.score_with_model(model, test_long, hp)
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret, dcost = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
    qqq_s = core.stats(qqq_t)
    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess_qqq = (s['cagr'] - qqq_s['cagr']) * 100 if qqq_s else None
    excess_spy = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    fi = pd.Series(model.feature_importance('gain'), index=hp['feature_names'])
    top_feat = fi.nlargest(2).index.tolist()

    return {
        'year': test_year,
        'universe_size': len(valid_tickers),
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'qqq_cagr': qqq_s['cagr'] if qqq_s else None,
        'spy_cagr': spy_s['cagr'] if spy_s else None,
        'excess_qqq': excess_qqq,
        'excess_spy': excess_spy,
        'top_features': top_feat,
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] ML walk-forward on S&P 500 universe")
    w(f"  Top-{TOP_N} biweekly, cost {COST_ONEWAY*200:.2f}% RT")
    w("=" * 110)

    # Load data with new master_dir
    print("Loading panel...")
    close, vol = core.load_panel(master_dir=DATA_DIR)
    print(f"  Loaded {close.shape[1]} tickers x {close.shape[0]} dates")
    print(f"  Range: {close.index.min().date()} ~ {close.index.max().date()}")

    qqq_ret = load_qqq()
    spy_ret = load_spy_benchmark()
    hp = dict(ml_model.ML_HP_DEFAULT)

    rows = []
    test_years = list(range(1995, 2026))
    for test_year in test_years:
        result = run_one_window(close, vol, test_year, hp, qqq_ret, spy_ret)
        if result is None:
            continue
        rows.append(result)
        e_qqq = result['excess_qqq'] if result['excess_qqq'] else 0
        e_spy = result['excess_spy'] if result['excess_spy'] else 0
        w(f"  [{result['year']}] U={result['universe_size']:>3} | "
          f"CAGR {result['cagr']*100:+7.2f}% Sh {result['sharpe']:>5.2f} "
          f"MDD {result['mdd']*100:>6.2f}% | "
          f"vs QQQ {e_qqq:+6.2f}%p vs SPY {e_spy:+6.2f}%p | "
          f"top: {result['top_features']}")

    if not rows:
        return

    # Aggregate
    excesses_qqq = [r['excess_qqq'] for r in rows if r['excess_qqq'] is not None]
    excesses_spy = [r['excess_spy'] for r in rows if r['excess_spy'] is not None]
    avg_cagr = sum(r['cagr'] for r in rows) / len(rows)
    avg_sharpe = sum(r['sharpe'] for r in rows) / len(rows)
    avg_mdd = sum(r['mdd'] for r in rows) / len(rows)

    w(f"\n{'='*110}")
    w(f"## {len(rows)}-window aggregate")
    w(f"  Avg Test CAGR: {avg_cagr*100:+.2f}%")
    w(f"  Avg Test Sharpe: {avg_sharpe:.2f}")
    w(f"  Avg Test MDD: {avg_mdd*100:.2f}%")

    w(f"\n## Statistical significance vs QQQ")
    if excesses_qqq:
        m_q = sum(excesses_qqq) / len(excesses_qqq)
        t_q = t_stat(excesses_qqq)
        win_q = sum(1 for e in excesses_qqq if e > 0)
        w(f"  Mean alpha: {m_q:+.2f}%p")
        w(f"  Win rate: {win_q}/{len(excesses_qqq)} ({win_q/len(excesses_qqq)*100:.0f}%)")
        w(f"  t-stat: {t_q:.2f}")

    w(f"\n## Statistical significance vs SPY")
    if excesses_spy:
        m_s = sum(excesses_spy) / len(excesses_spy)
        t_s = t_stat(excesses_spy)
        win_s = sum(1 for e in excesses_spy if e > 0)
        w(f"  Mean alpha: {m_s:+.2f}%p")
        w(f"  Win rate: {win_s}/{len(excesses_spy)} ({win_s/len(excesses_spy)*100:.0f}%)")
        w(f"  t-stat: {t_s:.2f}")

    # Feature importance
    feat_count = Counter()
    for r in rows:
        for f in r['top_features']:
            feat_count[f] += 1
    w(f"\n## Top-2 feature importance ({len(rows)} windows)")
    for f, c in feat_count.most_common():
        w(f"  {f}: {c}/{len(rows)}회 ({c/len(rows)*100:.0f}%)")

    # Save CSV
    df = pd.DataFrame(rows)
    df['top_features'] = df['top_features'].astype(str)
    csv_path = os.path.join(OUTPUT_DIR, 'ml_sp500_walkforward.csv')
    df.to_csv(csv_path, index=False)
    w(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'ml_sp500_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
