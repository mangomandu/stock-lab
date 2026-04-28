"""
Task 2: QQQ real benchmark comparison.

Fetch QQQ daily data via yfinance and compare Weekly Top-20 strategy against
actual QQQ buy-and-hold (cap-weighted, real index).

Output: results/qqq_comparison.txt
"""
import core
import pandas as pd
import numpy as np
import os
from datetime import datetime

OUTPUT_DIR = '/home/dlfnek/stock_lab/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_qqq():
    """Get QQQ daily Close from yfinance."""
    import yfinance as yf
    qqq = yf.Ticker('QQQ').history(period='max', interval='1d').reset_index()
    qqq['Date'] = pd.to_datetime(qqq['Date'], utc=True).dt.tz_convert('UTC').dt.normalize().dt.tz_localize(None)
    qqq = qqq.set_index('Date')['Close']
    return qqq


def cum_to_stats(returns_series, label):
    s = core.stats(returns_series)
    if s is None:
        return f"{label}: no data"
    return s


def main():
    out_lines = []
    def w(line=''):
        print(line)
        out_lines.append(line)

    w(f"[{datetime.now()}] QQQ Real Benchmark Comparison")
    w("=" * 70)

    # Strategy backtest (Weekly Top-20, default HP)
    hp = core.merge_hp({'top_n': 20, 'rebal_days': 5})
    close, vol = core.load_panel()
    rsi_s, ma_s, vol_s = core.compute_scores(close, vol, hp)
    train_mask, test_mask = core.get_period_masks(close, hp)

    train_results = core.grid_search(close, rsi_s, ma_s, vol_s, train_mask, hp)
    best = train_results[0]
    test_strat = core.eval_strategy(close, rsi_s, ma_s, vol_s, best['w'], test_mask, hp)

    # Equal-weight benchmark (existing, from our universe)
    train_eq = core.stats(core.benchmark_eq_weight(close, train_mask))
    test_eq = core.stats(core.benchmark_eq_weight(close, test_mask))

    # QQQ real benchmark
    w("\nFetching QQQ daily data via yfinance...")
    try:
        qqq = fetch_qqq()
    except Exception as e:
        w(f"  ERROR: {e}")
        return

    qqq_ret = qqq.pct_change()

    # Align QQQ to our test period
    test_start = pd.Timestamp(hp['split_date'])
    train_start = pd.Timestamp(hp['train_start'])
    qqq_train = qqq_ret[(qqq_ret.index >= train_start) & (qqq_ret.index < test_start)]
    qqq_test = qqq_ret[qqq_ret.index >= test_start]
    qqq_train_s = core.stats(qqq_train)
    qqq_test_s = core.stats(qqq_test)

    w(f"\nQQQ data range: {qqq.index.min().date()} ~ {qqq.index.max().date()}")
    w(f"Train period:  {hp['train_start']} ~ {hp['split_date']}")
    w(f"Test period:   {hp['split_date']} ~ now")
    w(f"\nBest train weights: {core.fmt_w(best['w'])}")

    w("\n" + "-" * 70)
    w(f"{'Metric':<22} {'Strategy':>14} {'EqW Bench':>14} {'QQQ (real)':>14}")
    w("-" * 70)
    w(f"{'Train CAGR':<22} {best['cagr']*100:>13.2f}% {train_eq['cagr']*100:>13.2f}% {qqq_train_s['cagr']*100:>13.2f}%")
    w(f"{'Train Sharpe':<22} {best['sharpe']:>14.2f} {train_eq['sharpe']:>14.2f} {qqq_train_s['sharpe']:>14.2f}")
    w(f"{'Train MDD':<22} {best['mdd']*100:>13.2f}% {train_eq['mdd']*100:>13.2f}% {qqq_train_s['mdd']*100:>13.2f}%")
    w(f"{'Test  CAGR':<22} {test_strat['cagr']*100:>13.2f}% {test_eq['cagr']*100:>13.2f}% {qqq_test_s['cagr']*100:>13.2f}%")
    w(f"{'Test  Sharpe':<22} {test_strat['sharpe']:>14.2f} {test_eq['sharpe']:>14.2f} {qqq_test_s['sharpe']:>14.2f}")
    w(f"{'Test  MDD':<22} {test_strat['mdd']*100:>13.2f}% {test_eq['mdd']*100:>13.2f}% {qqq_test_s['mdd']*100:>13.2f}%")
    w(f"{'Test  CumRet':<22} {test_strat['cum']*100:>13.2f}% {test_eq['cum']*100:>13.2f}% {qqq_test_s['cum']*100:>13.2f}%")
    w("-" * 70)

    # Excess returns analysis
    w("\n## Excess return analysis (Strategy vs QQQ)")
    train_excess = (best['cagr'] - qqq_train_s['cagr']) * 100
    test_excess = (test_strat['cagr'] - qqq_test_s['cagr']) * 100
    w(f"Train excess CAGR vs QQQ: {train_excess:+.2f}%p")
    w(f"Test  excess CAGR vs QQQ: {test_excess:+.2f}%p (this is the real alpha)")

    # Survivorship bias quantification
    w("\n## Survivorship bias in our equal-weight benchmark")
    survivorship_train = (train_eq['cagr'] - qqq_train_s['cagr']) * 100
    survivorship_test = (test_eq['cagr'] - qqq_test_s['cagr']) * 100
    w(f"Train: EqW {train_eq['cagr']*100:.2f}% vs QQQ {qqq_train_s['cagr']*100:.2f}% = +{survivorship_train:.2f}%p")
    w(f"Test:  EqW {test_eq['cagr']*100:.2f}% vs QQQ {qqq_test_s['cagr']*100:.2f}% = +{survivorship_test:.2f}%p")
    w("(Equal-weight bench inflates because we only see current NASDAQ-100 winners)")

    # Save QQQ data for future reuse
    qqq_path = os.path.join(OUTPUT_DIR, 'qqq_close.csv')
    qqq.to_frame('Close').to_csv(qqq_path)
    w(f"\nQQQ data saved: {qqq_path}")

    # Save report
    out_path = os.path.join(OUTPUT_DIR, 'qqq_comparison.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(out_lines))
    print(f"\nReport: {out_path}")


if __name__ == '__main__':
    main()
