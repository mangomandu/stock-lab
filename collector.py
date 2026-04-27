import yfinance as yf
import os

def collect_data():
    # KO(코카콜라) 전체 데이터 가져오기
    ticker = "KO"
    ko = yf.Ticker(ticker)
    hist = ko.history(period="max")

    # data/ 폴더 생성 및 저장
    os.makedirs("data", exist_ok=True)
    file_path = "data/KO_max_history.csv"
    hist.to_csv(file_path)

    print(f"{ticker} 전체 데이터 수집 완료! (경로: {file_path})")

if __name__ == "__main__":
    collect_data()
