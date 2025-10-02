from pathlib import Path
import pandas as pd

INPUT_PATH = Path(r"C:\Users\admin\Desktop\재무제표정리\통합재무\IFRS_UNIFIED.parquet")
OUTPUT_PATH = INPUT_PATH.with_name(f"{INPUT_PATH.stem}_filtered{INPUT_PATH.suffix}")

TARGET_REPORT = "사업보고서"
TARGET_MARKETS = {"코스닥시장상장법인", "유가증권시장상장법인"}

df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
mask = (df["보고서종류"] == TARGET_REPORT) & df["시장구분"].isin(TARGET_MARKETS)
filtered = df.loc[mask]
filtered.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

print(f"Saved {len(filtered)} rows to {OUTPUT_PATH}")
