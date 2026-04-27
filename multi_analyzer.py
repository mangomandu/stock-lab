import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

MASTER_DIR = '/home/dlfnek/stock_lab/data/master'

def calculate_rsi(series, period=14):
    if len(series) < period + 1: return 50.0
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # 0으로 나누기 방지 및 정밀도 확보
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def analyze():
    print(f"[{datetime.now()}] Unified Analyzer 가동...")
    while True:
        if not os.path.exists(MASTER_DIR):
            time.sleep(5); continue
        
        files = [f for f in os.listdir(MASTER_DIR) if f.endswith('.csv')]
        for file in files:
            ticker = file.replace('.csv', '')
            try:
                # 전체 데이터를 로드하여 지표 연속성 확보
                df = pd.read_csv(os.path.join(MASTER_DIR, file))
                if len(df) < 20: continue
                
                close = df['Close'].iloc[-1]
                rsi = calculate_rsi(df['Close'])
                
                # RSI 50.0 고정 여부 체크 (정밀 분석)
                rsi_score = np.clip((70 - rsi) / 40 * 100, 0, 100)
                
                ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
                ma60 = df['Close'].rolling(window=60).mean().iloc[-1]
                ma_score = (50 if close > ma20 else 0) + (50 if close > ma60 else 0)
                
                total_score = rsi_score * 0.5 + ma_score * 0.5
                
                if total_score > 85 or total_score < 15:
                    grade = "Strong Buy" if total_score > 85 else "Strong Sell"
                    print(f"[REPORT] Score: {total_score:.1f} ({grade}), Ticker: {ticker}, Reason: RSI:{rsi:.2f}, Price:{close:.2f}")
            except Exception:
                continue
        time.sleep(30)

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    analyze()
