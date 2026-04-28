# 자양동 Stock Lab

S&P 500 universe + LightGBM/Ridge ML 기반 quant 투자 모델.

## 검증 결과 (v4)

- **Walk-forward 31 windows (1995-2025)**
- **Avg CAGR: +43.69%** (Sharpe 1.64)
- **vs SPY alpha: +31.03%p** (t-stat 6.02, p < 0.0001)
- 승률 28/31 (90%)

자세한 내용: `reports/final_tuning_report.md`

## 핵심 설정

| 항목 | 값 |
|---|---|
| Universe | S&P 500 + 17 ETF (518 tickers) |
| Model | Ridge regression (또는 LightGBM) |
| Features | momentum(12-1), lowvol, trend, rsi, ma, volsurge |
| Train years | 7 |
| Forward horizon | 10 days |
| Rebalance | Weekly (5 trading days) |
| Top-N | 20 (또는 5/10/30 옵션) |
| Cost | 0.10% round-trip |

## 폴더 구조

```
stock_lab/
├── core.py                  # 백테스트 엔진 (HP dict, build_holdings, stats)
├── factors.py               # 학술 팩터 (momentum 12-1, low-vol, trend filter)
├── ml_model.py              # LightGBM 파이프라인 (Ridge로 대체 가능)
├── current_portfolio.py     # 매수 추천 (실시간 학습 + Top-N 출력)
├── fetch_sp500_list.py      # Wikipedia에서 S&P 500 명단 가져오기
├── download_sp500.py        # yfinance로 일봉 데이터 다운로드
├── validate_sp500.py        # 데이터 검증 (IPO date, universe size 등)
├── requirements.txt
│
├── data/
│   ├── master_sp500/        # 종목별 일봉 CSV (518 종목)
│   ├── sp500_tickers.txt    # 현재 S&P 500 명단
│   ├── etf_tickers.txt      # ETF 명단
│   └── all_sp500_tickers.txt
│
├── tests/                   # 검증 스크립트 (재실행 가능)
│   ├── test_qqq_benchmark.py        # QQQ 실비교
│   ├── test_cost_sensitivity.py     # 비용 민감도
│   ├── test_train_window.py         # 7년이 sweet spot
│   ├── test_forward_horizon.py      # 10d/Weekly 최적
│   ├── test_hp_tuning.py            # LightGBM HP search
│   ├── test_model_compare.py        # Ridge > LightGBM 발견
│   ├── test_ml_sp500.py             # 31 windows walk-forward
│   ├── test_event_catalog.py        # 큰 이벤트 자동 식별
│   ├── test_event_response.py       # 섹터 회복 패턴
│   ├── test_event_rules.py          # Stop-loss 등 룰 검증
│   ├── test_v4_portfolio_rules.py   # Top-N vs Score-weighted
│   └── test_v4_bootstrap.py         # 30 runs robustness
│
├── results/                 # 검증 결과 (.txt, .csv)
│   ├── current_portfolio.csv        # 최신 매수 추천
│   ├── ml_sp500_walkforward.csv     # 31 windows 결과
│   └── archive/                     # 옛 결과들
│
└── reports/
    ├── final_tuning_report.md       # 최종 보고서 (v4)
    └── archive/                     # 옛 보고서들
```

## 사용법

### 1. 데이터 다운로드 (한 번만)

```bash
python3 fetch_sp500_list.py    # Wikipedia에서 명단
python3 download_sp500.py      # yfinance로 일봉 (약 1-2분)
python3 validate_sp500.py      # 검증
```

### 2. 매수 추천 받기

```bash
python3 current_portfolio.py
```

출력: 최신 데이터로 학습한 Top-20 종목 + 매수 비중. CSV: `results/current_portfolio.csv`

### 3. 검증 재실행

```bash
python3 tests/test_ml_sp500.py        # 31 windows 알파 측정
python3 tests/test_train_window.py    # train years 그리드
python3 tests/test_v4_bootstrap.py    # robustness
```

## 운용 룰

```
주간 (매주 월요일):
  1. python3 current_portfolio.py     # 새 Top-N 출력
  2. 토스증권에서 차이 종목 매매 (소수점 매매)
```

## 검증된 사실

- ✅ Universe 확장 (NASDAQ-100 → S&P 500): 알파 +8%p
- ✅ 7년 train > 5년 (sweet spot)
- ✅ Weekly > Biweekly > Daily
- ✅ Ridge > LightGBM > XGBoost > MLP > Random Forest
- ✅ Top-N: 작을수록 알파 ↑, MDD ↑ (변동 ↑)
- ❌ Stop-loss: 손해 (회복기 놓침)
- ❌ Vol filter / Drawdown halt: 손해
- ❌ CASH 통합: 효과 미미
- ⚠ Survivorship bias: ~35% 부풀림 추정 (Bootstrap)

## 알려진 한계

- Survivorship bias 미해결 (현재 S&P 500만 보유 → API 가입 필요)
- 비용 0.20% 이상이면 알파 사라짐 (한국 broker 직접 매매 어려움)
- 미국 무수수료 broker 또는 토스증권 외화통장 필요

## 다음 단계 후보

- Bootstrap robustness check on v4 (진행 중)
- Score-weighted portfolio rule (진행 중)
- 새 features (다양한 momentum horizon, regime)
- 앙상블 (Ridge + LightGBM)
- Survivorship bias 해결 (Tiingo/Alpha Vantage API)
- 한국 시장 추가 (pykrx)
