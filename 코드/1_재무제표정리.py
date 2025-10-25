import os
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

# === 설정 ===
ENCODING = "cp949"
ROOT = Path(r"C:\Users\admin\Desktop\재무제표정리")
OUT_FILE = Path(r"C:\Users\admin\Desktop\재무제표정리\통합재무\ALL.parquet")
BATCH_SIZE = 200_000

os.makedirs(OUT_FILE.parent, exist_ok=True)

# === 연속된 탭을 하나로 취급 ===
def safe_split(line: str):
    """더블탭(\t\t 등)을 하나로 취급해서 split"""
    return re.split(r"\t+", (line or "").strip())

# === 헤더 정리: 첫 번째 '당기*'까지만 포함 ===
def clean_header(raw_header):
    tokens = []
    for tok in raw_header:
        cleaned = " ".join(tok.split()).strip()
        if cleaned == "":
            continue
        if cleaned.startswith("당기"):
            tokens.append("당기")
            break
        tokens.append(cleaned)
    return tokens

# === ParquetWriter 초기화 ===
writer = None

# === 모든 txt 파일 순회 ===
for src in ROOT.rglob("*.txt"):
    with open(src, "r", encoding=ENCODING, errors="replace") as f:
        lines = f.read().splitlines()
    if not lines:
        continue

    # 헤더 정리
    header_raw = safe_split(lines[0])
    header = clean_header(header_raw)

    # '당기' 컬럼이 없으면 해당 파일 스킵
    if "당기" not in header:
        print(f"[스킵] {src.name}: '당기' 컬럼 없음")
        continue

    rows = []

    # 본문 데이터 처리
    for line in lines[1:]:
        toks = safe_split(line)

        # 헤더 길이에 맞춰 자르거나 NA로 채움
        if len(toks) > len(header):
            toks = toks[:len(header)]
        elif len(toks) < len(header):
            toks += [np.nan] * (len(header) - len(toks))

        rows.append(toks)

        # batch 크기 도달 시 Parquet 기록
        if len(rows) >= BATCH_SIZE:
            df = pd.DataFrame(rows, columns=header)
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUT_FILE, table.schema, compression="snappy")
            writer.write_table(table)
            rows.clear()

    # 마지막 남은 데이터 flush
    if rows:
        df = pd.DataFrame(rows, columns=header)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_FILE, table.schema, compression="snappy")
        writer.write_table(table)

# === 마무리 ===
if writer:
    writer.close()

print("완료: ALL.parquet 저장됨 (더블탭 통합 + '당기'까지만 컬럼 + '당기' 없는 파일 스킵)")
