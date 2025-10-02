# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 경로/설정
# =========================
FACTOR_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성.parquet"      # 덮어쓸(또는 업데이트할) 연간 팩터 파일
PRICE_FILE  = r"C:\Users\admin\Desktop\재무제표정리\통합재무\월별첫영업일가격.parquet"   # (날짜, 종목코드, 종가) — 수정주가 기반

# 저장 정책: True면 FACTOR_FILE을 직접 덮어씀 / False면 새 파일로 저장
OVERWRITE   = False
OUT_FILE    = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터데이터_완성_withMOM.parquet"

# 리밸 월(고정 7월)
REBAL_MONTH = 7

# 계산할 모멘텀 스펙들: (J, S) = (누적개월 수, 최근 제외 개월 수)
# 예) 12–1, 12–6 둘 다 넣고 싶으면 아래처럼 유지. 하나만 원하면 하나만 남기세요.
MOM_SPECS   = [(12, 1), (12, 6),(6,1)]

# =========================
# 1) 데이터 로드 & 정리
# =========================
df_factor = pd.read_parquet(FACTOR_FILE, engine="pyarrow").copy()
df_price  = pd.read_parquet(PRICE_FILE,  engine="pyarrow").copy()

# 키 정규화
df_factor["종목코드"] = df_factor["종목코드"].astype(str).str.strip().str.zfill(6)
df_factor["결산연도"] = df_factor["결산연도"].astype(int)

df_price["종목코드"]  = df_price["종목코드"].astype(str).str.strip().str.zfill(6)
df_price["날짜"]      = pd.to_datetime(df_price["날짜"], errors="coerce")
df_price = df_price.dropna(subset=["날짜", "종가"])

# =========================
# 2) 월수익률 & 모멘텀(12–S) 계산
# =========================
# wide pivot (index=날짜, columns=종목코드)
px = df_price.pivot(index="날짜", columns="종목코드", values="종가").sort_index()
rets = px.pct_change(fill_method=None)  # 월수익률

def compute_mom_panel(rets_df: pd.DataFrame, J: int, S: int) -> pd.DataFrame:
    """
    MOM_{J–S}(t) = exp( sum_{m=t-J-S+1}^{t-S} log(1+r_m) ) - 1
      - shift(S): 최근 S개월 제외(룩어헤드 방지)
      - rolling(J): 과거 J개월 누적
    """
    logcum = np.log1p(rets_df).shift(S).rolling(J, min_periods=J).sum()
    return np.expm1(logcum)

# 여러 스펙을 한 번에 계산 → long 형태로 결합
mom_series = {}
for (J, S) in MOM_SPECS:
    panel = compute_mom_panel(rets, J, S)               # wide
    mom_series[f"MOM_{J}_{S}"] = panel.stack()          # MultiIndex Series: (날짜, 종목코드)

mom_long = pd.DataFrame(mom_series).reset_index()
mom_long.rename(columns={"level_0": "날짜", "level_1": "종목코드"}, inplace=True)

# =========================
# 3) 리밸월(7월) 스냅샷 → 결산연도(Y-1) 매핑
# =========================
snap = mom_long[mom_long["날짜"].dt.month == REBAL_MONTH].copy()
snap["연도"]     = snap["날짜"].dt.year
snap["결산연도"] = snap["연도"] - 1

# 같은 (결산연도, 종목코드)에 2건 이상 존재할 수 없지만, 혹시를 대비해 가장 최근만 유지
cols_keep = ["결산연도", "종목코드"] + [c for c in mom_long.columns if c.startswith("MOM_")]
snap = (snap[cols_keep]
        .sort_values(["결산연도", "종목코드"])
        .drop_duplicates(["결산연도", "종목코드"], keep="last"))

# =========================
# 4) 팩터 파케이에 모멘텀 컬럼 합치기
# =========================
# 기존에 동일 이름의 MOM 컬럼이 있다면 삭제 후 재생성(덮어쓰기)
exist_mom_cols = [c for c in df_factor.columns if c.startswith("MOM_")]
if exist_mom_cols:
    df_factor = df_factor.drop(columns=exist_mom_cols)

df_out = df_factor.merge(snap, on=["결산연도", "종목코드"], how="left")

# =========================
# 5) 저장
# =========================
if OVERWRITE:
    # 안전을 위해 백업 권장: Path(FACTOR_FILE).with_suffix(".bak.parquet")
    df_out.to_parquet(FACTOR_FILE, engine="pyarrow", index=False)
    print("[INFO] 덮어쓰기 완료 →", FACTOR_FILE)
else:
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(OUT_FILE, engine="pyarrow", index=False)
    print("[INFO] 새 파일 저장 →", OUT_FILE)

# =========================
# 6) 요약 출력
# =========================
new_mom_cols = [c for c in df_out.columns if c.startswith("MOM_")]
print("[INFO] 추가된 모멘텀 컬럼:", new_mom_cols)
print("[INFO] 샘플 5행]\n", df_out[["결산연도", "종목코드"] + new_mom_cols].head())
