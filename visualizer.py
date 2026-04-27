import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_data():
    file_path = "data/KO_max_history.csv"
    if not os.path.exists(file_path):
        print(f"에러: {file_path} 파일이 없습니다.")
        return

    # 데이터 로드 (Date를 인덱스로)
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # 시각화 설정
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 상단: 주가 흐름 (Close Price)
    ax1.plot(df.index, df['Close'], color='blue', label='Close Price')
    ax1.set_title('KO Historical Analysis')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True)

    # 하단: 배당금 지급 현황 (Dividends)
    ax2.bar(df.index, df['Dividends'], color='red', label='Dividends', width=100)
    ax2.set_ylabel('Dividends (USD)')
    ax2.set_xlabel('Year')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # 결과 저장
    output_chart = "KO_max_chart.png"
    plt.savefig(output_chart)
    print(f"시각화 완료! 차트가 저장되었습니다: {output_chart}")

if __name__ == "__main__":
    visualize_data()
