"""
Portfolio rule comparison on v4 environment.

v4 setting: S&P 500 + Ridge + 7y train + Weekly + Top-20 equal weight.

Test alternative portfolio rules:
  - Top-N variants (10, 20, 30)
  - Linear weight + cutoff (1%, 3%, 5%, 7%)
  - Softmax T=1/2/3 + cutoff (2%, 5%)

Walk-forward 31 windows (1995-2025).

Output: results/v4_portfolio_rules.txt
"""
import core
import ml_model
import factors
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

REBAL_DAYS = 5  # Weekly
TRAIN_YEARS = 7
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100


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


# Portfolio rule functions
def softmax_weights(scores, temperature=1.0):
    z = (scores - scores.mean()) / (scores.std() + 1e-9)
    exp_z = np.exp(z * temperature)
    return exp_z / exp_z.sum()


def linear_weights(scores):
    shifted = scores - scores.min()
    s = shifted.sum()
    if s == 0:
        return pd.Series(1 / len(scores), index=scores.index)
    return shifted / s


def apply_cutoff(weights, min_weight):
    if min_weight <= 0:
        return weights
    kept = weights >= min_weight
    if not kept.any():
        kept = weights == weights.max()
    survivor_weights = weights.where(kept, 0)
    s = survivor_weights.sum()
    return survivor_weights / s if s > 0 else survivor_weights


def topn_weights(scores, top_n):
    """Equal weight on top N."""
    sorted_idx = scores.sort_values(ascending=False).index[:top_n]
    w = pd.Series(0.0, index=scores.index)
    w.loc[sorted_idx] = 1.0 / top_n
    return w


def build_portfolio(score_wide, rule, params, rebal_days=REBAL_DAYS):
    """For each rebalance day, compute portfolio weights."""
    n_days, n_stocks = score_wide.shape
    weights = np.zeros((n_days, n_stocks))
    last_w = np.zeros(n_stocks)
    last_rebal = -rebal_days

    for t in range(n_days):
        if (t - last_rebal) >= rebal_days:
            row = score_wide.iloc[t].dropna()
            if len(row) == 0:
                weights[t] = last_w
                continue

            if rule == 'topn':
                w = topn_weights(row, params['top_n'])
            elif rule == 'linear':
                w = linear_weights(row)
                w = apply_cutoff(w, params['min_weight'])
            elif rule == 'softmax':
                w = softmax_weights(row, params['temperature'])
                w = apply_cutoff(w, params['min_weight'])
            else:
                raise ValueError(rule)

            full = pd.Series(0.0, index=score_wide.columns)
            full.loc[w.index] = w
            last_w = full.values
            last_rebal = t

        weights[t] = last_w

    return pd.DataFrame(weights, index=score_wide.index, columns=score_wide.columns)


def backtest(close, score_wide, rule, params, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    score = score_wide.where(close.notna())
    weights_df = build_portfolio(score, rule, params, rebal_days)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, weights_df


def train_ridge(train_long, hp):
    feat_cols = hp['feature_names']
    X_train = train_long[feat_cols].values
    y_train = train_long['target'].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    return model, scaler


def run_one_window(close, vol, test_year, hp, spy_ret, rule, params):
    train_start = pd.Timestamp(f'{test_year - TRAIN_YEARS}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
    if len(valid_tickers) < MIN_UNIVERSE_SIZE:
        return None

    close_sub = close[valid_tickers]
    vol_sub = vol[valid_tickers]

    train_long, test_long, _, _ = ml_model.get_train_test_features(
        close_sub, vol_sub, train_mask, test_mask, hp)
    if len(train_long) < 1000:
        return None

    model, scaler = train_ridge(train_long, hp)
    feat_cols = hp['feature_names']
    X_test = test_long[feat_cols].values
    X_test_s = scaler.transform(X_test)
    preds = model.predict(X_test_s)

    score_long = pd.Series(preds, index=pd.MultiIndex.from_arrays(
        [test_long['date'], test_long['ticker']]))
    score_wide = ml_model.long_to_wide(
        score_long, close_sub.index[test_mask], close_sub.columns)

    test_close = close_sub[test_mask]
    port_ret, weights_df = backtest(test_close, score_wide, rule, params)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    avg_n_held = (weights_df > 0).sum(axis=1).mean()

    return {
        'year': test_year, 'cagr': s['cagr'], 'sharpe': s['sharpe'],
        'mdd': s['mdd'], 'excess': excess, 'avg_n_held': avg_n_held,
    }


def run_config(close, vol, hp, spy_ret, rule, params, label):
    rows = []
    for test_year in range(1995, 2026):
        r = run_one_window(close, vol, test_year, hp, spy_ret, rule, params)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'label': label, 'rule': rule, 'params': params,
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_mdd': sum(r['mdd'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
        'avg_n_held': sum(r['avg_n_held'] for r in rows) / len(rows),
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Portfolio rule comparison on v4 environment")
    w(f"  Ridge + 7y train + Weekly | 31 windows | cost {COST_ONEWAY*200:.2f}%")
    w("=" * 100)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers\n")

    configs = [
        # Top-N baselines
        ('Top-10 equal',           'topn',    {'top_n': 10}),
        ('Top-15 equal',           'topn',    {'top_n': 15}),
        ('Top-20 equal (current)', 'topn',    {'top_n': 20}),
        ('Top-30 equal',           'topn',    {'top_n': 30}),
        ('Top-50 equal',           'topn',    {'top_n': 50}),

        # Linear weighted
        ('Linear + cut 1%',        'linear',  {'min_weight': 0.01}),
        ('Linear + cut 2%',        'linear',  {'min_weight': 0.02}),
        ('Linear + cut 3%',        'linear',  {'min_weight': 0.03}),
        ('Linear + cut 5%',        'linear',  {'min_weight': 0.05}),

        # Softmax weighted
        ('Softmax T=1 + cut 2%',   'softmax', {'temperature': 1.0, 'min_weight': 0.02}),
        ('Softmax T=1 + cut 5%',   'softmax', {'temperature': 1.0, 'min_weight': 0.05}),
        ('Softmax T=2 + cut 5%',   'softmax', {'temperature': 2.0, 'min_weight': 0.05}),
        ('Softmax T=3 + cut 5%',   'softmax', {'temperature': 3.0, 'min_weight': 0.05}),
    ]

    results = []
    for label, rule, params in configs:
        w(f"\n[{label}] running...")
        r = run_config(close, vol, hp, spy_ret, rule, params, label)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f} | held {r['avg_n_held']:.1f}")

    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)

    w(f"\n{'='*120}")
    w(f"## Summary (sorted by Sharpe)")
    w(f"{'Config':<28} {'CAGR':>8} {'Sh':>5} {'MDD':>8} {'vs SPY':>9} {'Win':>8} {'t':>6} {'held':>6}")
    w("-" * 120)
    for r in results:
        w(f"{r['label']:<28} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p "
          f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f} {r['avg_n_held']:>6.1f}")

    best = results[0]
    w(f"\n★ Best Sharpe: {best['label']}")
    w(f"  CAGR {best['avg_cagr']*100:+.2f}%, Sharpe {best['avg_sharpe']:.2f}, "
      f"vs SPY {best['avg_excess']:+.2f}%p, t={best['t_stat']:.2f}, "
      f"avg held {best['avg_n_held']:.1f} stocks")

    out_path = os.path.join(OUTPUT_DIR, 'v4_portfolio_rules.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
