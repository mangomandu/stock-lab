import pandas as pd
import numpy as np
import time
import os
import yfinance as yf
from datetime import datetime, timedelta

# 하이퍼파라미터 설정 (엄격한 기준)
WEIGHTS = {'RSI': 0.3, 'MA': 0.3, 'DIV': 0.4}
RSI_PERIOD = 14
SHORT_MA = 20
LONG_MA = 60

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_grade(score):
    if score > 80: return "Strong Buy"
    if score >= 70: return "Buy"
    if score >= 40: return "Hold"
    if score >= 20: return "Sell"
    return "Strong Sell"

def analyze():
    ticker_symbol = "KO"
    hist_path = "data/KO_max_history.csv"
    realtime_path = "data/KO_realtime.csv"
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 정교화된 엄격한 분석기 가동 시작...")
    ticker = yf.Ticker(ticker_symbol)
    # 최신 배당률 정보 가져오기
    try:
        annual_dividend = ticker.info.get('trailingAnnualDividendRate', 1.94)
    except:
        annual_dividend = 1.94

    last_processed_len = 0
    last_status_time = datetime.now()

    while True:
        try:
            # 1. 데이터 병합
            combined_df = pd.DataFrame()
            if os.path.exists(hist_path):
                hist_df = pd.read_csv(hist_path)
                combined_df = hist_df.tail(100).copy() # 지표 계산용 버퍼
            
            if os.path.exists(realtime_path):
                real_df = pd.read_csv(realtime_path)
                if len(real_df) > last_processed_len:
                    last_processed_len = len(real_df)
                    combined_df = pd.concat([combined_df, real_df], ignore_index=True)
                else:
                    time.sleep(1)
                    continue
            else:
                time.sleep(5)
                continue

            # 2. 로직 적용 (최소 데이터 확인)
            if len(combined_df) >= LONG_MA:
                # RSI (30%)
                rsi_values = calculate_rsi(combined_df['Close'], RSI_PERIOD)
                current_rsi = rsi_values.iloc[-1]
                # RSI 30이하(과매도)일 때 고득점, 70이상(과매수)일 때 저득점
                rsi_score = np.clip((70 - current_rsi) / 40 * 100, 0, 100) if not np.isnan(current_rsi) else 50

                # 이평선 (30%) - 엄격한 필터
                ma20 = combined_df['Close'].rolling(window=SHORT_MA).mean().iloc[-1]
                ma60 = combined_df['Close'].rolling(window=LONG_MA).mean().iloc[-1]
                current_price = combined_df['Close'].iloc[-1]
                
                ma_score = 0
                if current_price > ma20: ma_score += 50
                if current_price > ma60: ma_score += 50
                
                # 배당 매력 (40%)
                current_yield = annual_dividend / current_price
                div_score = np.clip((current_yield - 0.025) / 0.01 * 100, 0, 100)

                # 최종 점수 계산
                total_score = (rsi_score * WEIGHTS['RSI'] + 
                               ma_score * WEIGHTS['MA'] + 
                               div_score * WEIGHTS['DIV'])

                grade = get_grade(total_score)
                reason = f"RSI:{current_rsi:.1f}, Price:{current_price:.2f}, Yield:{current_yield*100:.2f}%"
                
                # 출력 필터링 (80점 초과 또는 20점 미만)
                if total_score > 80 or total_score < 20:
                    print(f"[REPORT] Score: {total_score:.1f} ({grade}), Reason: {reason}")
                
                # 1시간 주기 상태 보고
                if datetime.now() - last_status_time >= timedelta(hours=1):
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 분석기 정상 작동 중. 현재 점수: {total_score:.1f} ({grade})")
                    last_status_time = datetime.now()

        except Exception as e:
            print(f"분석 중 에러 발생: {e}")

        time.sleep(1)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    analyze()
