"""
Backtest event-driven rules on top of ML Top-20 Biweekly.

Rules tested:
  Baseline: Pure ML Top-20 Biweekly (no event rules)
  R1: Stop-loss -10% per stock (sell, wait until next biweekly)
  R2: Stop-loss -15% per stock (less sensitive)
  R3: Volatility filter (VIX-like via SPY 20-day stdev): 50% cash if vol > threshold
  R4: Buy-the-dip: when SPY drops >3% in 1 day, increase Top-N to 30 next rebalance
  R5: Drawdown halt: when SPY is >15% below peak, hold cash 50%

Output: results/event_rules_walkforward.txt
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from datetime import datetime

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 10
COST_ONEWAY = 0.0005


def load_spy():
    df = pd.read_csv(os.path.join(DATA_DIR, 'SPY.csv'))
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime']).sort_values('Datetime')
    df['Date'] = df['Datetime'].dt.normalize()
    return df.groupby('Date')['Close'].last()


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def backtest_with_rules(close, score_wide, spy_close, rule, params,
                       top_n=TOP_N, rebal_days=REBAL_DAYS, cost=COST_ONEWAY):
    """Backtest with various event rules.

    Rule types:
      'baseline': pure Top-N biweekly
      'stop_loss': sell stock if intra-period loss exceeds threshold
      'vol_filter': cash if SPY rolling std > threshold
      'drawdown_halt': cash if SPY drawdown > threshold
      'buy_dip': dynamic top_n based on recent SPY drop
    """
    score = score_wide.where(close.notna())
    n_days, n_stocks = score.shape
    score_arr = score.values
    close_arr = close.values

    # Pre-compute SPY-based signals
    spy_aligned = spy_close.reindex(close.index, method='ffill')
    spy_ret_1d = spy_aligned.pct_change()
    spy_peak = spy_aligned.cummax()
    spy_drawdown = spy_aligned / spy_peak - 1
    spy_20d_vol = spy_aligned.pct_change().rolling(20).std()
    spy_ret_5d = spy_aligned.pct_change(5)

    # Holdings
    weights = np.zeros((n_days, n_stocks))
    entry_prices = np.full(n_stocks, np.nan)  # for stop-loss
    last_rebal = -rebal_days

    for t in range(n_days):
        row = score_arr[t]
        valid = ~np.isnan(row)

        # Determine current holdings target
        if (t - last_rebal) >= rebal_days:
            # Rebalance
            current_topn = top_n
            cash_frac = 0.0

            # Apply rule for sizing
            if rule == 'vol_filter':
                if pd.notna(spy_20d_vol.iloc[t]) and spy_20d_vol.iloc[t] > params['vol_threshold']:
                    cash_frac = params.get('cash_frac', 0.5)
            elif rule == 'drawdown_halt':
                if pd.notna(spy_drawdown.iloc[t]) and spy_drawdown.iloc[t] < params['dd_threshold']:
                    cash_frac = params.get('cash_frac', 0.5)
            elif rule == 'buy_dip':
                if pd.notna(spy_ret_5d.iloc[t]) and spy_ret_5d.iloc[t] < params['dip_threshold']:
                    current_topn = params.get('expanded_top_n', 30)

            if not valid.any():
                weights[t] = weights[t-1] if t > 0 else weights[t]
                continue

            sorted_scores = np.where(valid, row, -np.inf)
            ranked_idx = np.argsort(-sorted_scores)
            new_top = np.zeros(n_stocks, dtype=bool)
            new_top[ranked_idx[:current_topn]] = True

            stock_frac = 1 - cash_frac
            n_held = new_top.sum()
            if n_held > 0:
                w = stock_frac / n_held
                new_weights = new_top.astype(float) * w
            else:
                new_weights = np.zeros(n_stocks)

            weights[t] = new_weights
            # Update entry prices for stop-loss tracking
            for i, on in enumerate(new_top):
                if on and not np.isnan(close_arr[t, i]):
                    entry_prices[i] = close_arr[t, i]
            for i, on in enumerate(new_top):
                if not on:
                    entry_prices[i] = np.nan
            last_rebal = t
        else:
            # Hold previous, but apply intra-period stop-loss
            prev_w = weights[t-1] if t > 0 else np.zeros(n_stocks)
            new_w = prev_w.copy()
            if rule == 'stop_loss':
                threshold = params['stop_threshold']
                for i in range(n_stocks):
                    if prev_w[i] > 0 and not np.isnan(entry_prices[i]) and not np.isnan(close_arr[t, i]):
                        loss = close_arr[t, i] / entry_prices[i] - 1
                        if loss <= threshold:
                            new_w[i] = 0  # liquidate
            weights[t] = new_w

    # Compute portfolio returns
    weights_df = pd.DataFrame(weights, index=close.index, columns=close.columns)
    held = weights_df.shift(1).fillna(0)
    stock_ret = close.pct_change().fillna(0)
    port_gross = (held * stock_ret).sum(axis=1)
    daily_cost = held.diff().abs().sum(axis=1).fillna(0) * cost
    return port_gross - daily_cost, daily_cost


def run_one_window(close, vol, spy, test_year, hp, rule, params):
    train_start = pd.Timestamp(f'{test_year - 5}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close.index.max():
        return None

    train_mask = (close.index >= train_start) & (close.index < test_start)
    test_mask = (close.index >= test_start) & (close.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

    valid_tickers = close.columns[close[train_mask].notna().sum() >= 252]
    if len(valid_tickers) < 100:
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
    spy_test = spy[(spy.index >= test_start) & (spy.index < test_end)]
    if len(spy_test) < 50:
        return None

    port_ret, dcost = backtest_with_rules(
        test_close, score_wide, spy_test, rule, params)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_s = core.stats(spy_test.pct_change())
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {
        'year': test_year,
        'cagr': s['cagr'], 'sharpe': s['sharpe'], 'mdd': s['mdd'],
        'spy_cagr': spy_s['cagr'] if spy_s else None,
        'excess': excess,
    }


def run_config(close, vol, spy, hp, rule, params, label):
    rows = []
    for test_year in range(1995, 2026):
        r = run_one_window(close, vol, spy, test_year, hp, rule, params)
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
    }


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Event-driven rules walk-forward")
    w(f"  Universe: S&P 500, Top-{TOP_N} Biweekly, cost {COST_ONEWAY*200:.2f}%")
    w("=" * 110)

    print("Loading panel...")
    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy = load_spy()
    hp = dict(ml_model.ML_HP_DEFAULT)
    print(f"  Loaded {close.shape[1]} tickers")

    configs = [
        ('Baseline (no rules)',          'baseline',     {}),
        ('Stop-loss -10%',               'stop_loss',    {'stop_threshold': -0.10}),
        ('Stop-loss -15%',               'stop_loss',    {'stop_threshold': -0.15}),
        ('Stop-loss -20%',               'stop_loss',    {'stop_threshold': -0.20}),
        ('Vol filter (>2%/day)',         'vol_filter',   {'vol_threshold': 0.02, 'cash_frac': 0.5}),
        ('Vol filter (>2.5%/day)',       'vol_filter',   {'vol_threshold': 0.025, 'cash_frac': 0.5}),
        ('Drawdown halt (-15%)',         'drawdown_halt', {'dd_threshold': -0.15, 'cash_frac': 0.5}),
        ('Drawdown halt (-20%)',         'drawdown_halt', {'dd_threshold': -0.20, 'cash_frac': 0.5}),
        ('Buy dip (5d -5% → Top-30)',    'buy_dip',      {'dip_threshold': -0.05, 'expanded_top_n': 30}),
        ('Buy dip (5d -3% → Top-25)',    'buy_dip',      {'dip_threshold': -0.03, 'expanded_top_n': 25}),
    ]

    results = []
    for label, rule, params in configs:
        w(f"\n[{label}] running...")
        r = run_config(close, vol, spy, hp, rule, params, label)
        if r:
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"MDD {r['avg_mdd']*100:.2f}% | "
              f"vs SPY {r['avg_excess']:+.2f}%p | "
              f"win {r['win']}/{r['n_windows']} | "
              f"t={r['t_stat']:.2f}")

    # Sort by Sharpe
    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)

    w(f"\n{'='*120}")
    w(f"## Summary (sorted by Sharpe)")
    w(f"{'Config':<32} {'CAGR':>8} {'Sh':>5} {'MDD':>8} {'vs SPY':>9} {'Win':>8} {'t':>6}")
    w("-" * 120)
    for r in results:
        w(f"{r['label']:<32} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_mdd']*100:>7.2f}% {r['avg_excess']:>+8.2f}%p "
          f"{r['win']:>3}/{r['n_windows']:<3} {r['t_stat']:>6.2f}")

    best = results[0]
    w(f"\n★ Best Sharpe: {best['label']}")
    w(f"  CAGR {best['avg_cagr']*100:+.2f}%, Sharpe {best['avg_sharpe']:.2f}, "
      f"vs SPY {best['avg_excess']:+.2f}%p, t={best['t_stat']:.2f}")

    out_path = os.path.join(OUTPUT_DIR, 'event_rules_walkforward.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
