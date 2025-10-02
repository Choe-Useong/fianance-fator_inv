from pykrx import stock
import pandas as pd
from datetime import datetime

ticker = "000020"   # 동화약품

records = []

# 현재 연도
this_year = datetime.today().year

# 2015~현재까지 결산년도
for year in range(2015, this_year):
    # 결산일 다음해 6월 영업일
    june_days = stock.get_previous_business_days(year=year+1, month=6)
    if not june_days:
        continue
    last_day = june_days[-1].strftime("%Y%m%d")
    
    try:
        snap = stock.get_market_cap_by_ticker(last_day)
        if ticker in snap.index:
            row = snap.loc[ticker].to_dict()
            row["종목코드"] = ticker
            row["기준연도"] = year
            row["날짜"] = last_day
            records.append(row)
            print(f"{year}년 결산 → {last_day} 시총 {row['시가총액']}")
    except Exception as e:
        print(f"{year}년 {last_day} 처리 오류: {e}")

df_check = pd.DataFrame(records)
print(df_check)
