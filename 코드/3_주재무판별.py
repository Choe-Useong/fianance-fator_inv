from pathlib import Path
import pandas as pd

# 원본 경로
INPUT_PATH = Path(r"C:\Users\admin\Desktop\재무제표정리\통합재무\ALL.parquet")

# 출력 경로
OUTPUT_PATH = INPUT_PATH.with_name("주재무_사업보고서_코스피코스닥.parquet")

# 필터 기준
TARGET_REPORT = "사업보고서"
TARGET_MARKETS = {"코스닥시장상장법인", "유가증권시장상장법인"}

# 1) 읽기
df = pd.read_parquet(INPUT_PATH, engine="pyarrow")

# 2) 조건 결합
mask = (
    (df["주재무"] == True)
    & (df["보고서종류"] == TARGET_REPORT)
    & (df["시장구분"].isin(TARGET_MARKETS))
)

filtered = df.loc[mask].copy()

# 3) 저장
filtered.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

print(f"완료! {len(filtered)} 행 저장됨 → {OUTPUT_PATH}")
