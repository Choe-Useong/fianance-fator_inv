import pandas as pd

# 원본 파일 경로
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\ALL.parquet"

# 1) parquet 읽기
df = pd.read_parquet(FILE_PATH)

# 2) 제표종류앞: 콤마 앞 첫 토큰
df["제표종류앞"] = df["재무제표종류"].str.split(",").str[0].str.strip()

# 3) 연결구분: 문자열에 "연결"/"별도" 있는지
def classify_conn(s: str) -> str:
    if pd.isna(s):
        return "기타"
    s = str(s)
    if "연결" in s:
        return "연결"
    elif "별도" in s:
        return "별도"
    else:
        return "기타"

df["연결구분"] = df["재무제표종류"].apply(classify_conn)

# 4) 결과 다시 덮어쓰기
df.to_parquet(FILE_PATH, engine="pyarrow", index=False)

print("완료! ALL.parquet 갱신됨 (제표종류앞, 연결구분 컬럼 추가)")
