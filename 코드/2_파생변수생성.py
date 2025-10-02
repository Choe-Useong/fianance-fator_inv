import pandas as pd
import os

# 1) 파일 경로
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\ALL.parquet"

# 2) Parquet 읽기
df = pd.read_parquet(FILE_PATH)

# 3) 제표종류앞: 콤마 앞 부분 추출
df["제표종류앞"] = df["재무제표종류"].str.split(",").str[0].str.strip()

# 4) 연결구분: "연결", "별도", 없으면 "기타"
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

# 5) 주재무: 같은 회사+결산기준일 그룹 안에 "연결"이 있으면 연결만 True,
#            없으면 별도만 True
has_conn = (
    df.groupby(["종목코드", "결산기준일"])["연결구분"]
      .transform(lambda g: "연결" in g.values)
)

df["주재무"] = df.apply(
    lambda row: (has_conn.loc[row.name] and row["연결구분"] == "연결")
                or (not has_conn.loc[row.name] and row["연결구분"] == "별도"),
    axis=1
)

# 6) 덮어쓰기 저장
df.to_parquet(FILE_PATH, engine="pyarrow", index=False)

print("완료! ALL.parquet 갱신됨 (제표종류앞, 연결구분, 주재무 추가)")
