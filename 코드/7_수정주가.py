import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime

# -------------------------
# 1) 기존 팩터 데이터 불러오기
# -------------------------
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성.parquet"
df_factor = pd.read_parquet(FILE_PATH, engine="pyarrow")

# 종목코드 정리
tickers = df_factor["종목코드"].dropna().unique().tolist()

# -------------------------
# 2) 월별 첫 영업일 Close 가져오기
# -------------------------
start_date = "2016-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

price_list = []

for i, ticker in enumerate(tickers, 1):
    try:
        # 데이터 다운로드 (수정주가 포함)
        df_price = fdr.DataReader(ticker, start_date, end_date)

        if df_price.empty:
            print(f"{ticker} 데이터 없음, 건너뜀")
            continue

        # 월별 첫 영업일 기준 Close만 추출
        monthly = df_price.resample("MS").first()[["Close"]].copy()
        monthly["종목코드"] = ticker
        monthly.reset_index(inplace=True)
        monthly.rename(columns={"Date": "날짜", "Close": "종가"}, inplace=True)

        price_list.append(monthly)
        print(f"{i}/{len(tickers)} {ticker} 처리 완료 ({len(monthly)} 개)")

    except Exception as e:
        print(f"{ticker} 오류: {e}")

# -------------------------
# 3) 합쳐서 Parquet 저장
# -------------------------
df_monthly_price = pd.concat(price_list, ignore_index=True)

OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\월별첫영업일가격.parquet"
df_monthly_price.to_parquet(OUT_FILE, engine="pyarrow", index=False)

print("저장 완료:", OUT_FILE)
print(df_monthly_price.head())
