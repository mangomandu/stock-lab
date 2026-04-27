import os
import time
from datetime import datetime

MASTER_DIR = '/home/dlfnek/stock_lab/data/master'
TICKER_FILE = '/home/dlfnek/data/nasdaq_top100.txt'

def monitor():
    if not os.path.exists(TICKER_FILE):
        print("티커 파일이 없습니다.")
        return

    with open(TICKER_FILE, 'r') as f:
        target_count = len([t for t in f.read().split(',') if t.strip()])

    print(f"[{datetime.now()}] 목표 {target_count}개 종목 감시 시작...")

    while True:
        if os.path.exists(MASTER_DIR):
            current_files = [f for f in os.listdir(MASTER_DIR) if f.endswith('.csv')]
            current_count = len(current_files)
            
            if current_count >= target_count:
                print("\n" + "="*50)
                print("★ ★ ★  MISSION COMPLETE  ★ ★ ★")
                print(f"나스닥 100개 전 종목 마스터 파일 생성 완료!")
                print(f"완료 시각: {datetime.now()}")
                print("="*50 + "\n")
                break
            else:
                print(f"현재 진행률: {current_count}/{target_count} ({(current_count/target_count)*100:.1f}%)", end='\r')
        
        time.sleep(30)

if __name__ == '__main__':
    monitor()
