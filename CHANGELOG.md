# CHANGELOG

전체 변경 이력 (소수점 minor 포함). README 개발 일지에는 major만 표시.

---

## v0.8.0 (2026-05-06) — 27 features × LightGBM × Sharadar × 한투 cost

**Base**: 27 features (price 10 + cross 2 + sector 3 + fundamental 10 + macro 2)
        × LightGBM (depth=4, n_est=200, lr=0.1)
        × Sharadar historical S&P 500 universe
        × 한국투자증권 평시 cost 0.25% per side
        × Weekly rebal (5d) + 10y train + yearly retrain
        × NaN fillna(median) + multi-horizon rank target
        × 60일 PIT lag

검증 결과 (1998-01-01 ~ 2026-05-04, 28년 walk-forward):
- α gross vs SPY:    **+11.83%p**
- α net   vs SPY:    **+5.44%p**  (cost 0.25% 적용)
- Sharpe:            0.56
- MDD:               -59.6%
- Rank IC:           +0.0166
- Annual turnover:   21.7x

**vs v0.7.2 baseline (4 features, +3.98%p)**: net α +1.46%p 개선 (cost 5배 환경에서).

**Feature importance top 5**:
1. yield_curve  20.3%   (macro regime)
2. vix_level     7.1%   (volatility regime)
3. beta_to_spy   6.8%   (market sensitivity)
4. roe_ttm       5.0%   (quality)
5. pb            5.0%   (value)

→ 모델 alpha source = "macro regime detection + regime-specific stock picking" (비선형 interaction, Tree만 잡음).

**검증된 약점 (다음 단계 후보)**:
- 2023-2024 AI rally 못 따라감 (-46%p 누적)
- Bull regime에서 약함, Bear/Recovery에서 강함 (contrarian/defensive)

**개발 단계 매핑**:
- Phase X: 29 features 빌드 (data prep)
- Phase Y v1 (실패): B1 fast retrain + daily rebal + NaN drop → -6.7%p
- Phase Y v3 (성공): Gemini fix 적용 → 24 cells bake-off → LGBM winner
- LGBM Sweep: 32 cells HP refinement → d=4 n=200 lr=0.1 재확인

**파일**:
- 코드:  `phase_y_pure_ml_v2.py`, `phase_y_lgbm_sweep.py`, `analyze_phase_y.py`
- 결과:  `results/phase_y_v3_results.json`, `results/lgbm_sweep_results.json`,
        `results/phase_y_lgbm_yearly.csv`, `results/phase_y_lgbm_importance.csv`,
        `results/phase_y_feature_corr.csv`
- Features:  `data/panels/features/*.parquet` (29개)
- Universe:  `data/sharadar/SP500.parquet` (historical members)

---

## v0.7.2 (2026-05-05) — Sharadar baseline (survivorship-corrected)

4 features (lowvol + rsi + volsurge + momentum_12_1) × Ridge α=1.0
× Sharadar historical S&P 500 (delisted 포함)
× cost 0.05% × daily rebal × 10y train × yearly retrain.

검증 결과: α +3.98%p, Sharpe 0.60, t=1.76 (n.s.).

**v0.7.1 +45.29%p의 진짜 정체 발견**:
- Survivorship bias 차감: -27.90%p
- Code 차이 (gauss-rank vs z-score): -16.37%p
- Data source: -2.26%p (yfinance vs Sharadar 차이 미세)

→ v0.7.1 +45%p의 86%가 부풀림. v0.7.2 +3.98%p가 진짜 baseline.

---

## v5 (2026-04-28 ~ 진행 중)

**Base**: 3-feature minimum (lowvol + rsi + volsurge) + Ridge α=1.0 + 7y train + Weekly + S&P 500 universe.

검증 알파 (leakage fix 후): **+32.85%p** (Sharpe 1.74, t=6.52) — 31 windows 1995-2025.
**+ Hysteresis exit_50 적용 시: alpha ~+35.5%p 추정** (이전 +37.28에서 leakage 차감 ~1.7%p).

### v5.4 (2026-04-29 저녁) — Target leakage 수정
- **버그 발견**: train target = 10일 forward 수익률 → train 마지막 10일의 target이 test 기간 가격 사용 → 누설
- **수정**: `ml_model.get_train_test_features`에서 train mask 마지막 forward_days(10일) buffer 적용
- **재검증 결과** (v5 baseline):
  - Alpha 34.56 → **32.85%p** (Δ -1.71%p, 5% 부풀림 발견)
  - Sharpe 1.79 → **1.74**
  - Win rate 28/31 → 27/31
  - t-stat 6.75 → 6.52 (여전히 매우 강함)
- **다른 테스트 영향**: 모든 config 동등하게 ~1.7%p 하향. 상대 비교 (Top-N×Cap, Hysteresis 등) 유효성 보존.
- 이전 보고된 모든 alpha 수치는 leakage 부풀림 포함 → README 정정

### v5.3 (2026-04-29 오후 늦게) — Hysteresis 채택 + 곡선 확정
- **Hysteresis 검증** (1차 — exit_n ∈ {20, 25, 30, 40, 50}, 2차 — {45, 50, 55, 60, 65, 75}):
  - 전체 곡선: 50에서 alpha peak (+37.28%p), 45에서 Sharpe peak (1.81), 65 이상 degradation
  - exit_50 best: alpha **+2.72%p** (34.56 → 37.28), Sharpe +0.01, turnover -37% (10.82% → 6.74%)
  - 공짜 점심 (cost ↓ + alpha ↑ 동시)
  - Sweet zone: exit_45~55 (모두 Sharpe ≥ 1.80, alpha 36.6-37.3%p)
  - 메커니즘: 매주 강제 회전이 비용 + 변동성만 늘림. 30~50등 약간 떨어진 종목 holding 유지하면 alpha generation 더 강함
- **Top-N 늘리기 vs Hysteresis 비교** (kill됨, 부분 결과):
  - Top-30 (no hyst) ≈ Top-20+exit_30 비슷한 turnover지만 알파는 hysteresis 우월 예상
  - Hysteresis = 자본 집중 (1/20=5%), Top-50 = 분산 (1/50=2%) → 신호 희석
- **Sector-relative score 검증** ❌: alpha -4.54%p (Sharpe 1.85 ↑이지만 절대 알파 큰 손해)
- **Ensemble (Ridge + LightGBM) 검증** ❌: alpha -7%p (3-feature에서 LGBM Sharpe 1.43 약함)
- `current_portfolio.py`에 `HYST_EXIT` tunable parameter 추가 (default 50)
- prev portfolio 자동 로드 (results/current_portfolio.csv)
- Tests: `test_hysteresis.py`, `test_hysteresis_deep.py`, `test_sector_relative.py`, `test_ensemble.py`, `test_topn_vs_hyst.py` (killed)

### v5.2 (2026-04-29 오후)
- **Effective N 진단**: yearly 31 windows + weekly 69 + baseline 비교
- 결과: 우리 모델 EffN 4.0, baseline (random Top-20) 대비 +0.35 → 우리만의 몰빵 X
- **ETF buffer 검증**:
  - SPY (같은 자산군) — 모든 비율 무가치
  - TLT (장기채) — Sharpe ↑ but alpha -18%p (트레이드오프, 안정형 옵션)
  - GLD (금) — TLT와 비슷
- **2축 운용 구조** 도입: PROFILE (standard / low_risk) + Parameters (TOP_N, SECTOR_CAP)
- TLT 40% buffer 옵션 추가 (low_risk profile)
- Tests: `test_effective_n_yearly.py`, `test_effective_n_baseline.py`, `test_etf_buffer.py`, `test_tlt_gld_buffer.py`

### v5.1 (2026-04-29 오전)
- **Top-N × Cap 매트릭스 (3-feature 환경)**: 15 configs × 31 windows
  - Sharpe 1위: Top-15 Cap15% / Top-20 Cap20% (둘 다 1.84)
  - Alpha 1위: Top-10 Cap30% (+45.2%p)
  - 모든 config가 v4 (6-feature) 대비 Sharpe +0.05~0.16 개선
- **Weekly walk-forward 2025-2026**: 69주 OOS 시뮬레이션
  - 누적 +135.87%p 초과수익, 승률 62.3%
- Tests: `test_3feature_topn_cap_cross.py`, `test_weekly_walkforward.py`, `test_ridge_alpha.py`

### v5.0 (2026-04-28)
- **Feature ablation breakthrough**: 6 features → 3 features (lowvol+rsi+volsurge)
- 결과: Sharpe 1.63 → 1.77 (+0.14), alpha +31.0%p → +34.2%p
- 발견: momentum/MA/trend가 redundant. 학계 표준 모멘텀이 우리 모델에선 noise
- Live trading 시작: $360 시드 (학생 학비) → SNDK, LITE, SATS, WDC, CIEN, COHR
- 라이브 portfolio 변경 (2026-04-28 저녁): SATS×2, COIN, SMCI, LITE, SNDK
- Tests: `test_feature_ablation.py`, `test_minimum_features.py`, `test_3feature_bootstrap.py`

---

## v4 (2026-04-27 ~ 04-28 오전)

- Ridge regression (vs LightGBM/XGBoost) 비교 후 **Ridge 채택**
- 7y train + Weekly rebalance 정착
- Sector cap 옵션 도입 (25-30% sweet)
- Walk-forward 31 windows: alpha +31.0%p, Sharpe 1.63
- Universe: S&P 500 + 17 ETF (520 종목) 다운로드
- Tests: `test_model_compare.py`, `test_train_window.py`, `test_sector_cap.py`, `test_topn_cap_cross.py`

---

## v3 (2026-04-26 ~ 04-27)

- Universe 확장: NASDAQ-100 (98) → S&P 500 + ETF (520)
- Walk-forward 31 windows (1995-2025): alpha +18.6%p (vs SPY), t=5.69
- 결정적 신뢰도 향상 (p < 0.0001)

---

## v2 (2026-04-25 ~ 04-26)

- Walk-forward 21 windows (2005-2025) 확장
- LightGBM ML 모델 도입
- Bootstrap robustness 검증 (30 runs)
- alpha +10.16%p (t=3.25, p<0.01)

---

## v1 (2026-04-24 ~ 04-25)

- 초기 NASDAQ-100 + RSI/MA/VOL 가중합 모델
- 11 windows walk-forward
- Biweekly Top-20 +2.94%p alpha (vs QQQ)
- 핸드크래프트 가중치 → ML 전환 결정
