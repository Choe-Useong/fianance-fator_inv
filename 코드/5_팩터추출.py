import pandas as pd
from pathlib import Path

FILE_PATH = Path(r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet")

CODE = "[246720]"          # 종목코드
NAME = ""                # 회사명 (정확히 회사명 컬럼만 사용)
STATEMENT = "단일포괄손익계산서"  # 제표종류앞 컬럼에서 검색할 문자열
YEAR = "2023"            # 연도
MAX_ROWS = 100

df = pd.read_parquet(FILE_PATH, engine="pyarrow")

pd.set_option("display.max_rows", MAX_ROWS)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

filtered = df.copy()

# 종목코드 필터
if "종목코드" in df.columns and CODE:
    filtered = filtered[filtered["종목코드"].astype(str) == str(CODE)]

# 회사명 필터 (정확히 '회사명' 컬럼만)
if NAME and "회사명" in df.columns:
    filtered = filtered[filtered["회사명"].astype(str).str.contains(NAME, na=False)]

# 제표종류앞 컬럼 필터 (정확히 일치)
if STATEMENT and "제표종류앞" in df.columns:
    filtered = filtered[filtered["제표종류앞"].astype(str) == STATEMENT]

# 연도 필터
if "결산기준일" in filtered.columns and YEAR:
    years = filtered["결산기준일"].astype(str).str.extract(r"(\d{4})", expand=False)
    filtered = filtered.assign(연도=years)
    filtered = filtered[filtered["연도"] == str(YEAR)]

# 주요 컬럼만 표시
cols = [c for c in [
    "종목코드", "회사명", "연도", "결산기준일", "연결구분",
    "제표종류앞", "항목코드_통일", "항목명", "당기"
] if c in filtered.columns]
filtered = filtered[cols] if cols else filtered

print(f"\n총 {len(filtered)}행")
print(filtered.head(MAX_ROWS))
