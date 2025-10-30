import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime

FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성.parquet"
df_factor = pd.read_parquet(FILE_PATH, engine="pyarrow")

tickers = df_factor["종목코드"].dropna().unique().tolist()

start_date = "2016-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

price_list = []

for i, ticker in enumerate(tickers, 1):
    df_price = pd.DataFrame()
    source = None

    try:
        # 1단계: YAHOO (코스피: .KS)
        df_price = fdr.DataReader(f"YAHOO:{ticker}.KS", start_date, end_date)
        if not df_price.empty and "Adj Close" in df_price.columns:
            df_price["Price"] = df_price["Adj Close"]
            source = "YAHOO_KS"
        else:
            raise ValueError("YAHOO_KS empty")

    except Exception:
        try:
            # 2단계: YAHOO (코스닥: .KQ)
            df_price = fdr.DataReader(f"YAHOO:{ticker}.KQ", start_date, end_date)
            if not df_price.empty and "Adj Close" in df_price.columns:
                df_price["Price"] = df_price["Adj Close"]
                source = "YAHOO_KQ"
            else:
                raise ValueError("YAHOO_KQ empty")

        except Exception:
            try:
                # 3단계: NAVER fallback (배당 미반영)
                df_price = fdr.DataReader(ticker, start_date, end_date)
                if not df_price.empty:
                    df_price["Price"] = df_price["Close"]
                    source = "NAVER"
                else:
                    print(f"{i}/{len(tickers)} {ticker}: NAVER도 데이터 없음, 건너뜀")
                    continue

            except Exception as e:
                print(f"{i}/{len(tickers)} {ticker}: 전체 실패 - {e}")
                continue

    # 월별 첫 영업일 기준 종가
    monthly = df_price.resample("MS").first()[["Price"]].copy()
    monthly["종목코드"] = ticker
    monthly["source"] = source
    monthly.reset_index(inplace=True)
    monthly.rename(columns={"Date": "날짜", "Price": "종가"}, inplace=True)
    price_list.append(monthly)

    print(f"{i}/{len(tickers)} {ticker} 완료 ({len(monthly)}건, 소스={source})")

# 전체 합치기 및 저장
if price_list:
    df_monthly_price = pd.concat(price_list, ignore_index=True)
    OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\월별첫영업일가격.parquet"
    df_monthly_price.to_parquet(OUT_FILE, engine="pyarrow", index=False)
    print("저장 완료:", OUT_FILE)
else:
    print("다운로드된 데이터 없음.")
