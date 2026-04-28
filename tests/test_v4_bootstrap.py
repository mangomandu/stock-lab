"""
Bootstrap robustness on v4 best config.

30 runs × universe 70% × 31 windows.
Verify alpha holds when universe is randomly subsetted.

Output: results/v4_bootstrap.txt
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 5
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100
N_BOOTSTRAP = 30
SUBSET_FRAC = 0.7
SEED = 42


def load_spy():
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


def backtest_topn(close, score_wide, top_n=TOP_N, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    hp = {'top_n': top_n, 'rebal_days': rebal_days, 'hysteresis': 0,
          'cost_oneway': cost}
    in_top = core.build_holdings(score, hp)
    weights_df = in_top.div(in_top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost


def run_one_window(close_sub, vol_sub, test_year, hp, spy_ret):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close_sub.index.max():
        return None

    train_mask = (close_sub.index >= train_start) & (close_sub.index < test_start)
    test_mask = (close_sub.index >= test_start) & (close_sub.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid_tickers = close_sub.columns[close_sub[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid_tickers) < MIN_UNIVERSE_SIZE:
        return None

    close_v = close_sub[valid_tickers]
    vol_v = vol_sub[valid_tickers]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_v, vol_v, train_mask, test_mask, hp)
    if len(train_long) < 1000:
        return None

    feat_cols = hp['feature_names']
    X_train = train_long[feat_cols].values
    y_train = train_long['target'].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    X_test = scaler.transform(test_long[feat_cols].values)
    preds = model.predict(X_test)

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_v.index[test_mask], close_v.columns)

    test_close = close_v[test_mask]
    port_ret = backtest_topn(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None
    return excess


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Bootstrap robustness on v4 best (Ridge + 7y + Weekly + Top-20)")
    w(f"  N runs: {N_BOOTSTRAP} | Subset: {int(SUBSET_FRAC*100)}% | Windows: 31")
    w("=" * 100)

    close_full, vol_full = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    all_tickers = list(close_full.columns)
    n_keep = int(len(all_tickers) * SUBSET_FRAC)

    w(f"\nFull universe: {len(all_tickers)} tickers")
    w(f"Each bootstrap: keep {n_keep} tickers")

    rng = random.Random(SEED)
    bootstrap_results = []

    for b in range(N_BOOTSTRAP):
        kept = rng.sample(all_tickers, n_keep)
        close_sub = close_full[kept]
        vol_sub = vol_full[kept]

        excesses = []
        for test_year in range(1995, 2026):
            e = run_one_window(close_sub, vol_sub, test_year, hp, spy_ret)
            if e is not None:
                excesses.append(e)

        if not excesses:
            continue
        mean_e = sum(excesses) / len(excesses)
        ts = t_stat(excesses)
        win = sum(1 for e in excesses if e > 0)
        bootstrap_results.append({
            'run': b + 1, 'n_windows': len(excesses),
            'mean_alpha': mean_e, 't_stat': ts,
            'win_rate': win / len(excesses) * 100,
        })
        w(f"  [Run {b+1:>2}] alpha {mean_e:+6.2f}%p, t={ts:>5.2f}, "
          f"win {win}/{len(excesses)} ({win/len(excesses)*100:.0f}%)")

    if not bootstrap_results:
        return

    means = [r['mean_alpha'] for r in bootstrap_results]
    ts_list = [r['t_stat'] for r in bootstrap_results]
    wins = [r['win_rate'] for r in bootstrap_results]

    w(f"\n{'='*100}")
    w(f"## Bootstrap aggregate ({len(bootstrap_results)} runs)")
    w(f"  Mean alpha:     {sum(means)/len(means):+.2f}%p")
    w(f"  Std of alphas:  {(sum((m - sum(means)/len(means))**2 for m in means)/len(means))**0.5:.2f}%p")
    w(f"  Min:            {min(means):+.2f}%p")
    w(f"  Max:            {max(means):+.2f}%p")
    w(f"  P25 / P50 / P75: {sorted(means)[len(means)//4]:+.2f}%p / "
      f"{sorted(means)[len(means)//2]:+.2f}%p / "
      f"{sorted(means)[3*len(means)//4]:+.2f}%p")
    n_sig = sum(1 for t in ts_list if t > 2)
    w(f"  Runs with t>2:  {n_sig}/{len(ts_list)} ({n_sig/len(ts_list)*100:.0f}%)")
    n_pos = sum(1 for m in means if m > 0)
    w(f"  Runs with α>0:  {n_pos}/{len(means)} ({n_pos/len(means)*100:.0f}%)")

    out_path = os.path.join(OUTPUT_DIR, 'v4_bootstrap.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
