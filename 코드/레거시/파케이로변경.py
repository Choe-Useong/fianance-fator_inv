import pandas as pd
from pathlib import Path

base_dir = Path("통합")
tsv_files = list(base_dir.rglob("*.tsv"))
print("발견된 TSV:", len(tsv_files))

dfs = []
for file in tsv_files:
    try:
        df = pd.read_csv(file, sep="\t", encoding="cp949")
        dfs.append(df)
    except Exception as e:
        print("⚠️ 오류:", file, e)

# 전부 합치기
if dfs:
    all_df = pd.concat(dfs, ignore_index=True)
    out_file = base_dir / "finance_all.parquet"
    all_df.to_parquet(out_file, engine="pyarrow", compression="zstd")
    print("✅ 저장 완료:", out_file)
else:
    print("❌ 합칠 데이터 없음")
