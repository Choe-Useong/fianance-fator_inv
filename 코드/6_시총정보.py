import pandas as pd
from pykrx import stock
from datetime import datetime
import numpy as np
import re


# 1) 기존 팩터 데이터 불러오기
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터재무데이터.parquet"
df = pd.read_parquet(FILE_PATH, engine="pyarrow")

# 혹시 인덱스로 저장됐으면 풀기
df = df.reset_index()



result = []

# 현재 연도 가져오기 (예: 2025)
this_year = datetime.today().year

# 2016년 ~ 현재 연도까지
for year in range(2016, this_year + 1):
    days = stock.get_previous_business_days(year=year, month=6)
    if not days:
        continue
    
    # 6월 마지막 영업일
    last_day = days[-1].strftime("%Y%m%d")
    
    try:
        snap = stock.get_market_cap_by_ticker(last_day)
        snap = snap.reset_index().rename(columns={"티커": "종목코드"})
        snap["날짜"] = last_day
        snap["연도"] = year
        result.append(snap)
        print(f"{year}년 {last_day} 처리 완료 (종목 수 {len(snap)})")
    except Exception as e:
        print(f"{year}년 {last_day} 처리 중 오류: {e}")

df_marketcap = pd.concat(result, ignore_index=True)

print(df_marketcap.head())
print(df_marketcap.tail())
print(df_marketcap["연도"].unique())


# 1) 종목코드 정규화
def clean_ticker(x):
    if pd.isna(x):
        return x
    return re.sub(r"[^0-9]", "", str(x))

df["종목코드"] = df["종목코드"].apply(clean_ticker)
df_marketcap["종목코드"] = df_marketcap["종목코드"].apply(clean_ticker)

# 2) 결산연도와 시총연도 컬럼 추가
df["결산연도"] = pd.to_datetime(df["결산기준일"]).dt.year
df["시총연도"] = df["결산연도"] + 1

# 3) df_marketcap에서 필요한 컬럼만
df_marketcap_sel = df_marketcap[["종목코드", "연도", "시가총액",'거래대금']].copy()

# 4) merge (종목코드 + 시총연도 vs 연도)
df_merged = df.merge(
    df_marketcap_sel,
    left_on=["종목코드", "시총연도"],
    right_on=["종목코드", "연도"],
    how="left"
)

# 5) 정리: 불필요한 보조 컬럼 제거
df_merged = df_merged.drop(columns=["연도"])



df_merged["SIZE"] = df_merged["시가총액"].apply(lambda x: np.log(x) if pd.notna(x) and x > 0 else pd.NA)
df_merged["BM"] = df_merged["지배주주지분"] / df_merged["시가총액"]
df_merged["GP_A"] = df_merged["매출총이익"] / df_merged["자산총계"]
df_merged["TURNOVER"] = df_merged["거래대금"] / df_merged["시가총액"]


# 1. '결산기준일' 컬럼을 datetime 타입으로 변환합니다.
# errors='coerce' 옵션은 날짜 형식에 맞지 않는 값이 있을 경우 NaT(Not a Time)로 변환해줍니다.
df_merged['결산기준일'] = pd.to_datetime(df_merged['결산기준일'], errors='coerce')

# 2. 월이 12인 행만 필터링합니다. (월이 12가 아닌 행을 drop하는 효과)
df_merged = df_merged[df_merged['결산기준일'].dt.month == 12]




OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성.parquet"
df_merged.to_parquet(OUT_FILE, engine="pyarrow", index=False)
print("팩터 포함 데이터 저장 완료:", OUT_FILE)
