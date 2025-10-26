import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime

# -------------------------
# 1) 기존 팩터 데이터 불러오기
# -------------------------
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성.parquet"
df_factor = pd.read_parquet(FILE_PATH, engine="pyarrow")

tickers = df_factor["종목코드"].dropna().unique().tolist()

# -------------------------
# 2) 월별 첫 영업일 Close 가져오기 (YF 우선, 없으면 NAVER)
# -------------------------
start_date = "2016-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

price_list = []

for i, ticker in enumerate(tickers, 1):
    try:
        # Yahoo 우선 시도 (배당·분할 반영 수정주가)
        ticker_yahoo = f"{ticker}.KS"  # 필요시 코스닥은 .KQ 조정
        df_price = fdr.DataReader(f"YAHOO:{ticker_yahoo}", start_date, end_date)

        # 비어있거나, Adj Close 누락 시 → NAVER fallback
        if df_price.empty or "Adj Close" not in df_price.columns:
            print(f"[{ticker}] Yahoo 데이터 없음 → NAVER로 대체")
            df_price = fdr.DataReader(ticker, start_date, end_date)
            df_price["Price"] = df_price["Close"]
        else:
            df_price["Price"] = df_price["Adj Close"]

        if df_price.empty:
            print(f"{ticker} 데이터 없음, 건너뜀")
            continue

        # 월별 첫 영업일 종가
        monthly = df_price.resample("MS").first()[["Price"]].copy()
        monthly["종목코드"] = ticker
        monthly.reset_index(inplace=True)
        monthly.rename(columns={"Date": "날짜", "Price": "종가"}, inplace=True)

        price_list.append(monthly)
        print(f"{i}/{len(tickers)} {ticker} 처리 완료 ({len(monthly)} 개)")

    except Exception as e:
        print(f"{ticker} 오류: {e}")

# -------------------------
# 3) 저장
# -------------------------
df_monthly_price = pd.concat(price_list, ignore_index=True)

OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\월별첫영업일가격_YF_NAVER.parquet"
df_monthly_price.to_parquet(OUT_FILE, engine="pyarrow", index=False)

print("저장 완료:", OUT_FILE)
print(df_monthly_price.head())
