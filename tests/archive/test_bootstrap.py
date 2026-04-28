"""
Bootstrap robustness check.

Survivorship bias 직접 해결은 어렵지만 (delisted 데이터 못 받아서),
universe를 랜덤하게 부분집합화해서 알파의 안정성을 측정한다.

가설:
- ML 모델이 진짜 알파라면, universe의 30%를 빼도 알파 비슷하게 나와야 함
- 만약 특정 winners (NVDA, AAPL 등)에 강하게 의존하면, 빼면 알파 급락

방법:
- 21-window walk-forward를 30번 반복
- 각 반복마다 universe 70% 랜덤 샘플링
- 21 windows의 평균 알파 분포 측정

Output: results/bootstrap_robustness.txt
"""
import core
import ml_model
import pandas as pd
import numpy as np
import os
from datetime import datetime
import random

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'

TOP_N = 20
REBAL_DAYS = 10
COST_ONEWAY = 0.0005
N_BOOTSTRAP = 30
SUBSET_FRAC = 0.7
SEED = 42


def load_qqq():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'qqq_close.csv'),
                     parse_dates=['Date']).set_index('Date')
    return df['Close'].pct_change()


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


def run_one_window(close_sub, vol_sub, test_year, hp, qqq_ret):
    train_start = pd.Timestamp(f'{test_year - 5}-01-01')
    test_start = pd.Timestamp(f'{test_year}-01-01')
    test_end = pd.Timestamp(f'{test_year + 1}-01-01')
    if test_start > close_sub.index.max():
        return None

    train_mask = (close_sub.index >= train_start) & (close_sub.index < test_start)
    test_mask = (close_sub.index >= test_start) & (close_sub.index < test_end)
    if train_mask.sum() < 252 or test_mask.sum() < 100:
        return None

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

    qqq_t = qqq_ret[(qqq_ret.index >= test_start) & (qqq_ret.index < test_end)]
    qs = core.stats(qqq_t)
    excess = (s['cagr'] - qs['cagr']) * 100 if qs else None
    return excess


def t_stat(values):
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5
    se = std / (n ** 0.5)
    return mean / se if se > 0 else 0


def main():
    out_lines = []
    def w(line=''):
        print(line, flush=True)
        out_lines.append(line)

    w(f"[{datetime.now()}] Bootstrap Robustness Check")
    w(f"  N bootstrap runs: {N_BOOTSTRAP}")
    w(f"  Universe subset:  {int(SUBSET_FRAC*100)}% of tickers (random)")
    w(f"  Walk-forward:     2005~2025 (21 windows)")
    w(f"  Seed:             {SEED}")
    w("=" * 90)

    close_full, vol_full = core.load_panel()
    qqq_ret = load_qqq()
    hp = dict(ml_model.ML_HP_DEFAULT)
    all_tickers = list(close_full.columns)
    n_keep = int(len(all_tickers) * SUBSET_FRAC)

    w(f"\nFull universe: {len(all_tickers)} tickers")
    w(f"Each bootstrap: keep {n_keep} tickers, drop {len(all_tickers)-n_keep}\n")

    rng = random.Random(SEED)
    bootstrap_results = []

    for b in range(N_BOOTSTRAP):
        kept = rng.sample(all_tickers, n_keep)
        close_sub = close_full[kept]
        vol_sub = vol_full[kept]

        excesses = []
        for test_year in range(2005, 2026):
            e = run_one_window(close_sub, vol_sub, test_year, hp, qqq_ret)
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
            'kept_sample': kept[:3] + ['...'] + kept[-3:],
        })
        w(f"  [Run {b+1:>2}] mean alpha {mean_e:+6.2f}%p, t={ts:>5.2f}, "
          f"win {win}/{len(excesses)} ({win/len(excesses)*100:.0f}%)")

    if not bootstrap_results:
        w("\n결과 없음")
        return

    # Aggregate
    means = [r['mean_alpha'] for r in bootstrap_results]
    ts_list = [r['t_stat'] for r in bootstrap_results]
    wins = [r['win_rate'] for r in bootstrap_results]

    w(f"\n{'='*90}")
    w(f"## Bootstrap aggregate ({len(bootstrap_results)} runs)")
    w(f"  Alpha distribution:")
    w(f"    Mean of means:   {sum(means)/len(means):+.2f}%p")
    w(f"    Std of means:    {(sum((m - sum(means)/len(means))**2 for m in means)/len(means))**0.5:.2f}%p")
    w(f"    Min:             {min(means):+.2f}%p")
    w(f"    Max:             {max(means):+.2f}%p")
    w(f"    25th pctile:     {sorted(means)[len(means)//4]:+.2f}%p")
    w(f"    50th pctile:     {sorted(means)[len(means)//2]:+.2f}%p")
    w(f"    75th pctile:     {sorted(means)[3*len(means)//4]:+.2f}%p")

    w(f"\n  Significance (t-stat) distribution:")
    w(f"    Mean t:          {sum(ts_list)/len(ts_list):.2f}")
    w(f"    Min t:           {min(ts_list):.2f}")
    w(f"    Max t:           {max(ts_list):.2f}")
    n_sig = sum(1 for t in ts_list if t > 2)
    w(f"    Runs with t>2:   {n_sig}/{len(ts_list)} ({n_sig/len(ts_list)*100:.0f}%)")

    w(f"\n  Win rate distribution:")
    w(f"    Mean:            {sum(wins)/len(wins):.1f}%")
    w(f"    Min:             {min(wins):.1f}%")
    w(f"    Max:             {max(wins):.1f}%")

    w(f"\n## 해석")
    full_alpha = 10.16  # from ml_extended_walkforward
    w(f"  Full universe (98 tickers) alpha: +{full_alpha:.2f}%p, t=3.25")
    sub_avg = sum(means) / len(means)
    w(f"  Bootstrap (70%, {len(bootstrap_results)} runs) avg: {sub_avg:+.2f}%p")
    drop_pct = (full_alpha - sub_avg) / full_alpha * 100 if full_alpha != 0 else 0
    w(f"  알파 변동: {drop_pct:+.1f}% (양수면 universe 줄여도 알파 유지)")

    if abs(sub_avg - full_alpha) < 2 and sum(1 for t in ts_list if t > 2) / len(ts_list) > 0.7:
        w(f"  → 알파가 universe 부분집합에서도 안정적. 특정 winners 의존 X. ✅")
    elif sub_avg < full_alpha * 0.5:
        w(f"  → 알파 급락. 특정 winners에 강하게 의존. ⚠")
    else:
        w(f"  → 알파 일부 약화. 특정 winners 영향 있음. △")

    df = pd.DataFrame(bootstrap_results)
    df['kept_sample'] = df['kept_sample'].astype(str)
    csv_path = os.path.join(OUTPUT_DIR, 'bootstrap_robustness.csv')
    df.to_csv(csv_path, index=False)
    w(f"\nCSV: {csv_path}")

    out_path = os.path.join(OUTPUT_DIR, 'bootstrap_robustness.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
