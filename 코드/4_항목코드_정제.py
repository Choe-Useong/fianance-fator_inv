import pandas as pd

# 1) 파일 경로
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet"

# 2) Parquet 읽기
df = pd.read_parquet(FILE_PATH, engine="pyarrow")


# 문자열 전체 공백 제거 (데이터 전체 전처리)
for col in df.columns:
    if pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].str.replace(r"\s+", "", regex=True)


# 3) 항목코드 통일 → 앞 접두사 ifrs_ / ifrs-full_ 제거
df["항목코드_통일"] = (
    df["항목코드"]
    .str.replace("ifrs-full_", "", regex=False)
    .str.replace("ifrs_", "", regex=False)
    .str.replace("dart_", "", regex=False)
)

# 4) 저장 (새 파일로 저장 권장)
OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet"
df.to_parquet(OUT_FILE, engine="pyarrow", index=False)

# 5) 유니크 확인
unique_codes = df["항목코드_통일"].unique()
print("총 항목코드 개수:", len(unique_codes))
print("샘플 20개:", unique_codes[:20])

print("완료! IFRS_UNIFIED.parquet 저장됨")
