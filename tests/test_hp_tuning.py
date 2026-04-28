"""
LightGBM hyperparameter random search.

Random sample of hyperparameter combinations, evaluate each via 31-window
walk-forward. Find best by Sharpe.

Search space:
  learning_rate: [0.02, 0.05, 0.1, 0.2]
  num_leaves: [15, 31, 63, 127]
  min_data_in_leaf: [20, 50, 100, 200]
  feature_fraction: [0.7, 0.8, 0.9, 1.0]
  bagging_fraction: [0.6, 0.8, 1.0]
  num_rounds: [100, 200, 400]

Total combinations: 4*4*4*4*3*3 = 2304. Random sample 30 trials.

Output: results/hp_tuning.txt
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime

DATA_DIR = '/home/dlfnek/stock_lab/data/master_sp500'
OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 5  # Weekly (new sweet spot)
TRAIN_YEARS = 7  # new sweet spot
COST_ONEWAY = 0.0005
MIN_UNIVERSE_SIZE = 100
N_TRIALS = 20
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

    valid_tickers = close.columns[close[train_mask].notna().sum() >= int(252 * 0.5)]
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
    port_ret = backtest_with_score(test_close, score_wide)
    s = core.stats(port_ret)
    if s is None or s['days'] < 50:
        return None

    spy_t = spy_ret[(spy_ret.index >= test_start) & (spy_ret.index < test_end)]
    spy_s = core.stats(spy_t)
    excess = (s['cagr'] - spy_s['cagr']) * 100 if spy_s else None

    return {'cagr': s['cagr'], 'sharpe': s['sharpe'], 'excess': excess}


def evaluate_hp(close, vol, hp, spy_ret):
    rows = []
    for test_year in range(2005, 2026):  # 21 windows for speed (vs 31)
        r = run_one_window(close, vol, test_year, hp, spy_ret)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    excesses = [r['excess'] for r in rows if r['excess'] is not None]
    return {
        'n_windows': len(rows),
        'avg_cagr': sum(r['cagr'] for r in rows) / len(rows),
        'avg_sharpe': sum(r['sharpe'] for r in rows) / len(rows),
        'avg_excess': sum(excesses) / len(excesses) if excesses else None,
        'win': sum(1 for e in excesses if e > 0),
        't_stat': t_stat(excesses),
    }


def sample_hp(rng, base_hp):
    hp = dict(base_hp)
    hp['lgb_params'] = dict(base_hp['lgb_params'])
    hp['lgb_params']['learning_rate'] = rng.choice([0.02, 0.05, 0.1, 0.2])
    hp['lgb_params']['num_leaves'] = rng.choice([15, 31, 63, 127])
    hp['lgb_params']['min_data_in_leaf'] = rng.choice([20, 50, 100, 200])
    hp['lgb_params']['feature_fraction'] = rng.choice([0.7, 0.8, 0.9, 1.0])
    hp['lgb_params']['bagging_fraction'] = rng.choice([0.6, 0.8, 1.0])
    hp['num_rounds'] = rng.choice([100, 200, 400])
    return hp


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] LightGBM hyperparameter random search")
    w(f"  Trials: {N_TRIALS}, Seed: {SEED}")
    w("=" * 110)

    close, vol = core.load_panel(master_dir=DATA_DIR)
    spy_ret = load_spy()
    base_hp = dict(ml_model.ML_HP_DEFAULT)
    w(f"Loaded {close.shape[1]} tickers\n")

    # Baseline first
    w(f"\n[Baseline (default HP)] running...")
    base_result = evaluate_hp(close, vol, base_hp, spy_ret)
    if base_result:
        w(f"  CAGR {base_result['avg_cagr']*100:+.2f}% Sh {base_result['avg_sharpe']:.2f} "
          f"vs SPY {base_result['avg_excess']:+.2f}%p t={base_result['t_stat']:.2f}")
        base_result['label'] = 'Baseline (default)'
        base_result['hp'] = {
            'lr': base_hp['lgb_params']['learning_rate'],
            'leaves': base_hp['lgb_params']['num_leaves'],
            'min_data': base_hp['lgb_params']['min_data_in_leaf'],
            'feat_frac': base_hp['lgb_params']['feature_fraction'],
            'bag_frac': base_hp['lgb_params']['bagging_fraction'],
            'rounds': base_hp['num_rounds'],
        }

    # Random search
    rng = random.Random(SEED)
    results = [base_result] if base_result else []
    for trial in range(N_TRIALS):
        hp = sample_hp(rng, base_hp)
        params_summary = {
            'lr': hp['lgb_params']['learning_rate'],
            'leaves': hp['lgb_params']['num_leaves'],
            'min_data': hp['lgb_params']['min_data_in_leaf'],
            'feat_frac': hp['lgb_params']['feature_fraction'],
            'bag_frac': hp['lgb_params']['bagging_fraction'],
            'rounds': hp['num_rounds'],
        }
        label = (f"lr={params_summary['lr']} leaves={params_summary['leaves']} "
                 f"min={params_summary['min_data']} ff={params_summary['feat_frac']} "
                 f"bf={params_summary['bag_frac']} R={params_summary['rounds']}")
        w(f"\n[Trial {trial+1}/{N_TRIALS}] {label}")
        r = evaluate_hp(close, vol, hp, spy_ret)
        if r:
            r['label'] = f"Trial {trial+1}"
            r['hp'] = params_summary
            results.append(r)
            w(f"  CAGR {r['avg_cagr']*100:+.2f}% Sh {r['avg_sharpe']:.2f} "
              f"vs SPY {r['avg_excess']:+.2f}%p t={r['t_stat']:.2f}")

    results.sort(key=lambda r: r['avg_sharpe'], reverse=True)

    w(f"\n{'='*110}")
    w(f"## Top 10 by Sharpe")
    w(f"{'Rank':<5} {'CAGR':>8} {'Sh':>5} {'vs SPY':>9} {'t':>6}  HP")
    w("-" * 110)
    for i, r in enumerate(results[:10]):
        w(f"{i+1:<5} {r['avg_cagr']*100:>+7.2f}% {r['avg_sharpe']:>5.2f} "
          f"{r['avg_excess']:>+8.2f}%p {r['t_stat']:>6.2f}  "
          f"lr={r['hp']['lr']} leaves={r['hp']['leaves']} min={r['hp']['min_data']} "
          f"ff={r['hp']['feat_frac']} bf={r['hp']['bag_frac']} R={r['hp']['rounds']}")

    out_path = os.path.join(OUTPUT_DIR, 'hp_tuning.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
