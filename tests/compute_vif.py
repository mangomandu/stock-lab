"""
VIF (Variance Inflation Factor) — feature 간 다중공선성 측정.

VIF_i = 1 / (1 - R²_i), where R²_i = R² of regressing feature_i on all other features.

해석:
  VIF = 1     → 독립 (no collinearity)
  VIF < 5     → OK (학계 일반)
  VIF 5-10    → 주의 (moderate collinearity)
  VIF > 10    → 심각 (severe — feature 빼야 함)

또한 pairwise correlation matrix 출력 (sanity check).

사용 예:
  from tests.compute_vif import compute_vif
  vif_df = compute_vif(features_long_df)
  print(vif_df)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_vif(features_df, feature_cols=None):
    """
    features_df: DataFrame with columns = feature names, rows = observations
                 (long format, NaN dropped)
    feature_cols: list of column names. If None, uses all columns.

    Returns DataFrame with 'feature' and 'vif' columns.
    """
    if feature_cols is None:
        feature_cols = list(features_df.columns)

    df = features_df[feature_cols].dropna()
    if len(df) < len(feature_cols) * 10:
        raise ValueError(
            f"Insufficient samples ({len(df)}) for {len(feature_cols)} features"
        )

    X = df.values
    vifs = []
    for i, name in enumerate(feature_cols):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        reg = LinearRegression().fit(X_others, y)
        r_squared = reg.score(X_others, y)
        vif = 1.0 / (1.0 - r_squared) if r_squared < 1 else float('inf')
        vifs.append((name, vif))

    return pd.DataFrame(vifs, columns=['feature', 'vif']).sort_values(
        'vif', ascending=False
    ).reset_index(drop=True)


def correlation_matrix(features_df, feature_cols=None):
    """Pairwise Pearson correlation."""
    if feature_cols is None:
        feature_cols = list(features_df.columns)
    return features_df[feature_cols].dropna().corr()


def print_vif_report(vif_df, corr_df=None, label=''):
    """Pretty-print VIF + correlation matrix."""
    print(f"\n=== VIF Report{' — ' + label if label else ''} ===")
    print(f"\n  VIF (sorted, lower = more independent):")
    for _, r in vif_df.iterrows():
        flag = '  '
        if r['vif'] > 10:
            flag = '✗✗'
        elif r['vif'] > 5:
            flag = '✗ '
        elif r['vif'] > 2:
            flag = '⚠ '
        else:
            flag = '✓ '
        print(f"    {r['feature']:<20} {r['vif']:>8.2f}  {flag}")

    if corr_df is not None:
        print(f"\n  Pairwise correlation:")
        print(corr_df.round(3).to_string())


# Self-test
def _self_test():
    np.random.seed(42)
    n = 1000

    # Independent features → VIF ≈ 1
    df_indep = pd.DataFrame({
        'a': np.random.randn(n),
        'b': np.random.randn(n),
        'c': np.random.randn(n),
    })
    vif1 = compute_vif(df_indep)
    print(f"Independent features: max VIF = {vif1['vif'].max():.2f} (expect ~1)")
    assert vif1['vif'].max() < 1.5

    # Collinear: c = 0.95*a + small noise → VIF for a, c high
    df_coll = df_indep.copy()
    df_coll['c'] = 0.95 * df_coll['a'] + 0.05 * np.random.randn(n)
    vif2 = compute_vif(df_coll)
    print(f"\nCollinear (a-c):")
    print_vif_report(vif2)
    assert vif2[vif2['feature'].isin(['a', 'c'])]['vif'].max() > 5

    print("\n✓ Self-test passed")


if __name__ == '__main__':
    _self_test()
