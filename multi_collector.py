import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

MASTER_DIR = '/home/dlfnek/stock_lab/data/master'
TICKER_FILE = '/home/dlfnek/data/nasdaq_top100.txt'

def fetch_lifetime_history(ticker):
    file_path = f"{MASTER_DIR}/{ticker}.csv"
    if os.path.exists(file_path): return
    
    try:
        t_obj = yf.Ticker(ticker)
        # 일봉 전체
        df_daily = t_obj.history(period="max", interval="1d").reset_index()
        if df_daily.empty: return
        
        df_daily = df_daily[['Date', 'Close', 'Volume']]
        df_daily.columns = ['Datetime', 'Close', 'Volume']
        df_daily['Datetime'] = pd.to_datetime(df_daily['Datetime']).dt.tz_localize(None)

        # 분봉 최근 7일
        df_min = t_obj.history(period="7d", interval="1m").reset_index()
        if not df_min.empty:
            df_min = df_min[['Datetime', 'Close', 'Volume']]
            df_min['Datetime'] = pd.to_datetime(df_min['Datetime']).dt.tz_localize(None)
            combined = pd.concat([df_daily, df_min]).drop_duplicates('Datetime').sort_values('Datetime')
        else:
            combined = df_daily

        combined.to_csv(file_path, index=False)
        print(f"[{ticker}] 복원 완료")
    except Exception:
        pass

def multi_collect():
    with open(TICKER_FILE, 'r') as f:
        tickers = [t.strip() for t in f.read().split(',') if t.strip()]
    
    os.makedirs(MASTER_DIR, exist_ok=True)
    
    # 1. 병렬 역사 복원 (Thread 10개 가동)
    print(f"[{datetime.now()}] 고속 병렬 복원 시작...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_lifetime_history, tickers)
    
    print(f"[{datetime.now()}] 모든 종목 복원 프로세스 완료. 실시간 모니터링 시작.")

    # 2. 실시간 Append 루프
    while True:
        for ticker in tickers:
            try:
                t_obj = yf.Ticker(ticker)
                live_hist = t_obj.history(period="1d", interval="1m")
                if not live_hist.empty:
                    last = live_hist.iloc[-1:]
                    dt = last.index[0].strftime('%Y-%m-%d %H:%M:%S')
                    close = last['Close'].values[0]
                    vol = int(last['Volume'].values[0])
                    with open(f"{MASTER_DIR}/{ticker}.csv", 'a') as f:
                        f.write(f"{dt},{close},{vol}\n")
                time.sleep(0.05)
            except: continue
        time.sleep(30)

if __name__ == '__main__':
    multi_collect()
