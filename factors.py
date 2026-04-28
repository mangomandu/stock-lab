"""
자양동 Stock Lab — Academic factor library.

Each factor is computed as a wide DataFrame (date × ticker) of raw factor
values. Z-score normalization happens cross-sectionally (per date, across
stocks) to make different-scale factors combinable without magic numbers.

Factors implemented (all peer-reviewed):
- Momentum 12-1 (Jegadeesh & Titman 1993)
- Low Volatility (Frazzini & Pedersen "Betting Against Beta")
- Trend Filter (Faber 2007 "Tactical Asset Allocation")
- Original RSI/MA/VOL (kept for ML feature comparison only)
"""
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Factor lookback periods (academic standard, not arbitrary)
# -----------------------------------------------------------------------------
FACTOR_DEFAULTS = {
    'momentum_long_days':  252,    # 12 months
    'momentum_skip_days':   21,    # last 1 month (reversal effect)
    'lowvol_window_days':  252,    # 1 year of daily returns
    'trend_ma_days':       200,    # classic trend filter
}


# -----------------------------------------------------------------------------
# Raw factor computation
# -----------------------------------------------------------------------------
def momentum_12_1(close, hp=None):
    """Jegadeesh-Titman 12-1: return from 12mo ago to 1mo ago.

    Skips the most recent month to avoid short-term reversal contamination.
    Higher = stronger trend = better.
    """
    if hp is None:
        hp = FACTOR_DEFAULTS
    long_d = hp['momentum_long_days']
    skip_d = hp['momentum_skip_days']
    # Price 21 days ago / Price 252 days ago - 1
    return close.shift(skip_d) / close.shift(long_d) - 1


def low_volatility(close, hp=None):
    """Negative of 1-year daily return volatility.

    Returns higher values for low-volatility stocks (low-vol anomaly:
    Frazzini & Pedersen). Negation lets us treat 'higher = better' uniformly.
    """
    if hp is None:
        hp = FACTOR_DEFAULTS
    window = hp['lowvol_window_days']
    daily_ret = close.pct_change()
    vol = daily_ret.rolling(window).std()
    return -vol  # higher = lower vol = better


def trend_filter(close, hp=None):
    """Binary 1.0 if Close > 200-day MA, else 0.0.

    Faber's classic regime filter. Used to avoid bear markets.
    """
    if hp is None:
        hp = FACTOR_DEFAULTS
    window = hp['trend_ma_days']
    ma = close.rolling(window).mean()
    return (close > ma).astype(float)


# -----------------------------------------------------------------------------
# Original RSI/MA/VOL signals (legacy — for ML feature comparison only)
# -----------------------------------------------------------------------------
def rsi_score(close, period=14):
    """Original mean-reversion RSI score, 0~100. Higher = more oversold."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    return ((70 - rsi.fillna(50)) / 40 * 100).clip(0, 100)


def ma20_above(close, period=20):
    """Binary: Close > 20-day MA. Trend signal."""
    return (close > close.rolling(period).mean()).astype(float) * 100


def volume_surge(close, vol, period=20):
    """Volume relative to 20-day average. Higher = unusual interest."""
    vma = vol.rolling(period).mean()
    vol_ratio = (vol / vma.replace(0, 1e-9)) * 100
    return ((vol_ratio - 50) / 150 * 100).clip(0, 100)


# -----------------------------------------------------------------------------
# Cross-sectional z-score
# -----------------------------------------------------------------------------
def cross_sectional_zscore(factor_df):
    """For each date, compute z-score across all tickers.

    (value - row_mean) / row_std. NaN-safe.
    """
    row_mean = factor_df.mean(axis=1)
    row_std = factor_df.std(axis=1).replace(0, 1e-9)
    return factor_df.sub(row_mean, axis=0).div(row_std, axis=0)


# -----------------------------------------------------------------------------
# Compute all factors at once (returns dict of factor_name -> DataFrame)
# -----------------------------------------------------------------------------
def compute_all_factors(close, vol, hp=None):
    """Compute all factors. Returns dict of name -> raw factor DataFrame."""
    if hp is None:
        hp = FACTOR_DEFAULTS
    return {
        'momentum':  momentum_12_1(close, hp),
        'lowvol':    low_volatility(close, hp),
        'trend':     trend_filter(close, hp),
        'rsi':       rsi_score(close),
        'ma':        ma20_above(close),
        'volsurge':  volume_surge(close, vol),
    }


def compute_zscored_factors(close, vol, hp=None):
    """Compute factors and z-score them cross-sectionally.

    Trend filter is binary so we don't z-score it (keep as 0/1).
    """
    raw = compute_all_factors(close, vol, hp)
    z = {}
    for name, df in raw.items():
        if name == 'trend':
            z[name] = df  # keep binary
        else:
            z[name] = cross_sectional_zscore(df)
    return z


# -----------------------------------------------------------------------------
# Factor combination for portfolio scoring
# -----------------------------------------------------------------------------
def combine_factors(z_factors, weights):
    """Linearly combine z-scored factors. weights = dict {name: weight}."""
    score = None
    for name, w in weights.items():
        if name not in z_factors or w == 0:
            continue
        contrib = z_factors[name] * w
        score = contrib if score is None else score + contrib
    return score


def factor_weight_grid(factor_names, step=0.25):
    """Generate weight candidates summing to 1.0, all non-negative.

    For 3 factors at step 0.25: ~10 candidates.
    For 4 factors at step 0.25: ~35 candidates.
    """
    n = int(round(1 / step))
    k = len(factor_names)

    def recurse(remaining, n_left):
        if n_left == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in recurse(remaining - i, n_left - 1):
                yield (i,) + rest

    candidates = []
    for combo in recurse(n, k):
        w = {name: round(combo[i] * step, 3)
             for i, name in enumerate(factor_names)}
        candidates.append(w)
    return candidates
