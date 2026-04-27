import yfinance as yf
import time
import os
from datetime import datetime

def realtime_collect():
    ticker = "KO"
    ko = yf.Ticker(ticker)
    file_path = "data/KO_realtime.csv"
    
    # data/ 폴더 확인 및 생성
    os.makedirs("data", exist_ok=True)
    
    # 파일이 없으면 헤더 생성
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("Datetime,Close,Volume\n")
            
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 실시간 데이터 수집을 시작합니다...")
            
    while True:
        try:
            # 1분 단위 데이터 가져오기
            hist = ko.history(period="1d", interval="1m")
            
            if not hist.empty:
                # 가장 최신 데이터 추출
                last_row = hist.iloc[-1]
                # 타임존 정보가 포함된 datetime을 문자열로 변환
                current_time = hist.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                close_price = last_row['Close']
                volume = last_row['Volume']
                
                # CSV 파일에 데이터 추가 (Append)
                with open(file_path, "a") as f:
                    f.write(f"{current_time},{close_price},{volume}\n")
                    
                # 터미널 출력
                print(f"현재 시각: {current_time}, KO 가격: {close_price:.2f} 수집 완료")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 장이 닫혀있거나 데이터를 가져올 수 없습니다.")
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 에러 발생: {e}")
            
        # 60초 대기
        time.sleep(60)

if __name__ == "__main__":
    # 버퍼링 없이 즉시 출력되도록 설정
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    realtime_collect()
