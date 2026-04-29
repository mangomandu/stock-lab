# CHANGELOG

전체 변경 이력 (소수점 minor 포함). README 개발 일지에는 major만 표시.

---

## v5 (2026-04-28 ~ 진행 중)

**Base**: 3-feature minimum (lowvol + rsi + volsurge) + Ridge α=1.0 + 7y train + Weekly + S&P 500 universe.

검증 알파: +34.16%p (Sharpe 1.77, t=6.63) — 31 windows 1995-2025.
**+ Hysteresis exit_50 적용 시: alpha +37.28%p (Sharpe 1.80, turnover -37%)**

### v5.3 (2026-04-29 오후 늦게) — Hysteresis 채택
- **Hysteresis 검증** (exit_n ∈ {20, 25, 30, 40, 50}, 31 windows):
  - exit_50 best: alpha **+2.72%p** (34.6 → 37.3), Sharpe +0.01, turnover -37% (10.82% → 6.74%)
  - 공짜 점심 (cost ↓ + alpha ↑ 동시)
  - 메커니즘: 매주 강제 회전이 비용 + 변동성만 늘림. 30~50등 약간 떨어진 종목 holding 유지하면 alpha generation 더 강함
- **Sector-relative score 검증** ❌: alpha -4.54%p (Sharpe 1.85 ↑이지만 절대 알파 큰 손해)
- **Ensemble (Ridge + LightGBM) 검증** ❌: alpha -7%p (3-feature에서 LGBM Sharpe 1.43 약함)
- `current_portfolio.py`에 `HYST_EXIT` tunable parameter 추가 (default 50)
- prev portfolio 자동 로드 (results/current_portfolio.csv)
- Tests: `test_hysteresis.py`, `test_sector_relative.py`, `test_ensemble.py`

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
