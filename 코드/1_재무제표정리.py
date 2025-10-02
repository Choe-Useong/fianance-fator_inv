"""
코드 설명
1. 현재 폴더 및 하위 폴더의 모든 .txt 파일을 찾아서 읽음
2. 각 파일의 헤더를 정리하여 공통 스키마를 만듦
3. 각 파일의 데이터를 공통 스키마에 맞춰 정리하고
4. 일정 크기(batch) 단위로 Parquet 파일에 기록
"""

import pandas as pd                  # 데이터프레임 처리를 위한 pandas
from pathlib import Path             # 경로 처리를 위한 Path 클래스
import pyarrow as pa                 # Parquet 변환을 위한 pyarrow
import pyarrow.parquet as pq         # Parquet 쓰기를 위한 모듈
import os

ENCODING = "cp949"       # TXT 파일을 읽을 때 사용할 문자 인코딩
ROOT = Path(".")         # 현재 폴더를 시작 경로로 설정
files = list(Path(ROOT).rglob("*.txt"))  # 현재 폴더 및 하위 폴더의 모든 .txt 파일 수집
BATCH_SIZE = 200_000     # 한 번에 처리할 데이터 행(batch) 크기
OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\ALL.parquet" # 최종 저장할 Parquet 파일 이름
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

def norm_token(text: str) -> str:
    """문자열에서 공백/탭 등을 정규화하여 깔끔하게 만듦"""
    return " ".join((text or "").split()).strip()  # 문자열을 공백 단위로 나누고 → 다시 공백으로 합치고 → 양 끝 공백 제거

def clean_header(raw_header):
    """
    헤더 정리:
      - 맨 앞부터 시작
      - 첫 번째 '당기*' 컬럼까지 포함
      - '당기*'는 '당기'로 이름 통일
      - 빈 칸 토큰은 무시
    """
    tokens = []   # 정리된 컬럼 이름들을 담을 리스트
    indices = []  # 각 컬럼의 원래 위치(인덱스)를 담을 리스트
    for idx, tok in enumerate(raw_header):        # 헤더에 있는 항목들을 하나씩 순회
        cleaned = norm_token(tok)                 # 항목 이름을 공백 정리
        if cleaned == "":                         # 만약 비어 있다면
            continue                              # 건너뛴다
        if cleaned.startswith("당기"):            # 만약 '당기'로 시작한다면
            tokens.append("당기")                 # 이름을 '당기'로 통일해서 추가
            indices.append(idx)                   # 원래 인덱스 위치도 기록
            break                                 # 이후 컬럼은 버리고 반복 종료
        tokens.append(cleaned)                    # 그 외에는 정리된 이름 그대로 추가
        indices.append(idx)                       # 인덱스도 함께 추가
    mapping = {tokens[i]: indices[i] for i in range(len(tokens))}  # 컬럼 이름과 인덱스를 짝지어 딕셔너리 생성
    return tokens, mapping                        # (정리된 컬럼 리스트, 매핑 딕셔너리) 반환



import re

import re

def safe_split(line: str, expected_vals: int = 3):
    """
    line: 데이터 라인
    expected_vals: 당기/전기/전전기 값 개수 (보통 3)
    return: 탭/공백 보정된 토큰 리스트 (strip 처리 포함)
    """
    # 연속 탭을 하나로 취급
    parts = re.split(r"\t+", line.strip())
    if len(parts) < 2:
        return [p.strip() for p in parts]

    label, candidates = parts[0].strip(), parts[1:]

    # 공백 제거 후 빈 값 제외
    non_empty = [c.strip() for c in candidates if c.strip() != ""]

    # 값 개수가 기대치만큼 있으면 그 값들만 반환
    if len(non_empty) == expected_vals:
        return [label] + non_empty

    # 그렇지 않으면 원본 parts에서 strip만 적용
    return [p.strip() for p in parts]




# 1. 기준 스키마 확정 (첫 번째 파일 기준)
with open(files[0], "r", encoding=ENCODING, errors="replace") as f:  # 첫 번째 txt 파일 열어서
    first_header = f.readline().rstrip("\r\n").split("\t")           # 첫 줄(헤더)을 개행 제거 후 탭으로 나눔
columns, _ = clean_header(first_header)                             # clean_header로 정리해서 기준 스키마 확정

# 2. ParquetWriter 준비
writer = None   # 아직 ParquetWriter가 없으므로 None으로 초기화

# 3. 모든 파일 순회
for src in files:                                                    # 모든 txt 파일에 대해 반복
    with open(src, "r", encoding=ENCODING, errors="replace") as f:   # 현재 파일 열어서
        lines = f.read().splitlines()                                # 줄 단위로 읽어 리스트로 저장
    if not lines:                                                    # 만약 파일이 비어 있다면
        continue                                                     # 건너뜀

    # 현재 파일 헤더 정리 → 기준 스키마와 매핑
    raw_header = lines[0].split("\t")                                # 첫 줄(헤더)을 탭으로 나눔
    _, local_map = clean_header(raw_header)                          # clean_header로 로컬 매핑 생성

    # 데이터 라인 처리 (batch 단위로 바로 기록)
    rows = []                                                        # 데이터를 임시로 담을 리스트
    for line in lines[1:]:                                           # 첫 줄 제외하고 데이터 줄 반복
        toks = safe_split(line, expected_vals=3)  #line.split("\t")                                    # 한 줄을 탭으로 나눔
        row = []                                                     # 한 줄 데이터를 담을 리스트
        for name in columns:                                         # 기준 스키마 순서대로 돌면서
            idx = local_map.get(name)                                # 현재 컬럼 이름의 원래 위치를 찾음
            if idx is not None and idx < len(toks):                  # 인덱스가 있고 데이터 범위 안이면
                row.append(toks[idx].strip())                        # 해당 위치 값의 공백 제거 후 추가
            else:
                row.append("")                                       # 값이 없으면 빈 문자열 추가
        rows.append(row)                                             # 완성된 한 줄 데이터를 rows에 추가

        # 배치 사이즈 도달 → 바로 Parquet 기록
        if len(rows) >= BATCH_SIZE:                                  # rows 크기가 BATCH_SIZE 이상이면
            df = pd.DataFrame(rows, columns=columns)                 # rows를 DataFrame으로 변환
            table = pa.Table.from_pandas(df, preserve_index=False)   # DataFrame을 pyarrow Table로 변환
            if writer is None:                                       # writer가 없으면
                writer = pq.ParquetWriter(OUT_FILE, table.schema, compression="snappy")  # 새 ParquetWriter 생성
            writer.write_table(table)                                # Table을 Parquet에 기록
            rows.clear()                                             # rows 비우기

    # 파일 끝나고 남은 행 flush
    if rows:                                                         # 아직 남은 데이터가 있으면
        df = pd.DataFrame(rows, columns=columns)                     # DataFrame으로 변환
        table = pa.Table.from_pandas(df, preserve_index=False)       # pyarrow Table로 변환
        if writer is None:                                           # writer가 없으면
            writer = pq.ParquetWriter(OUT_FILE, table.schema, compression="snappy")  # 새 ParquetWriter 생성
        writer.write_table(table)                                    # Table을 Parquet에 기록

# 4. ParquetWriter 닫기
if writer:                                                           # writer가 존재하면
    writer.close()                                                   # writer 닫기

print("완료! ALL.parquet 저장됨")                                    # 완료 메시지 출력





'''
for line in lines[1:]:                                           # 첫 줄 제외하고 데이터 줄 반복
    # 안전 분리: 연속된 탭도 하나로 보고, 공백이 아닌 값이 3개 있으면 그대로 반환
    toks = safe_split(line, expected_vals=3)

    # --- 롤백용 (원래 코드) ---
    # toks = line.split("\t")   # 단순히 탭 기준으로만 분리

'''