import pandas as pd

# 1) 데이터 로드
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet"
df = pd.read_parquet(FILE_PATH, engine="pyarrow")

# 2) 비금융사만 필터링
exclude_industries = ["은행및저축기관", "금융지원서비스업", "보험업",'신탁업및집합투자업','기타금융업']
df_non_fin = df[~df["업종명"].isin(exclude_industries)].copy()


# 결산월이 12월인 것만 선택
df_non_fin = df_non_fin[df_non_fin["결산월"] == '12']


# "당기" 컬럼을 숫자로 변환 (쉼표 제거 후 float)
df_non_fin["당기"] = (
    df_non_fin["당기"]
    .astype(str)                # 문자열로 변환
    .str.replace(",", "", regex=False)  # 쉼표 제거
    .replace({"": None})        # 빈 문자열 -> NA
    .astype(float)              # float으로 변환
)




# 3) 기준 집합: 종목코드 × 결산기준일 + 업종명 포함
base = (
    df_non_fin[["종목코드", "결산기준일", "업종명"]]
    .drop_duplicates()
    .sort_values(["종목코드", "결산기준일"])
)

# 4) 멀티인덱스 만들기 (업종명은 컬럼으로 남김)
idx = pd.MultiIndex.from_frame(base[["종목코드", "결산기준일"]],
                               names=["종목코드", "결산기준일"])

# 5) 스켈레톤 생성 후 업종명 컬럼 합치기
skeleton = pd.DataFrame(
    index=idx,
    columns=["지배주주지분", "매출총이익", "자산총계"],
    dtype="Float64"
).reset_index().merge(base, on=["종목코드", "결산기준일"], how="left").set_index(["종목코드", "결산기준일"])

print(skeleton.head(15))



##########################################################################


import re

# 1) 지배주주지분 관련 라벨 후보
owner_labels = [
    "지배기업소유주지분","지배기업의소유주지분","지배기업의소유주에게귀속되는자본",
    "지배기업의소유지분","지배회사소유지분","지배기업소유지분", '지배기업소유주지분',
    "지배기업소유주에귀속되는자본","지배주주지분","지배기업소유지분합계",
    "지배기업지분","지배기업의소유주에게귀속되는지분","지배기업주주지분"
]

def clean_label(x: str) -> str:
    if pd.isna(x):
        return ""
    x = re.sub(r"\s+", "", x)          # 공백 제거
    x = re.sub(r"[^가-힣0-9]", "", x)  # 한글/숫자만
    return x

df_non_fin["항목명_clean"] = df_non_fin["항목명"].apply(clean_label)

def get_owner_equity(sub: pd.DataFrame) -> float:
    # 연결/별도 구분
    has_consol = (sub["연결구분"] == "연결").any()

    if has_consol:
        # ---------------- 연결 기준 ----------------
        sub = sub[sub["연결구분"] == "연결"]

        # (1) IFRS 태그 직접
        direct = sub.loc[sub["항목코드_통일"] == "EquityAttributableToOwnersOfParent", "당기"].dropna()
        if len(direct) > 0:
            return float(direct.iloc[0])

        # (2) Equity - NCI
        equity = sub.loc[sub["항목코드_통일"] == "Equity", "당기"].dropna()
        nci    = sub.loc[sub["항목코드_통일"] == "NoncontrollingInterests", "당기"].dropna()
        if len(equity) > 0 and len(nci) > 0:
            return float(equity.iloc[0] - nci.iloc[0])

        # (3) 지배주주 라벨
        mask = sub["항목명_clean"].isin(owner_labels)
        direct_label = sub.loc[mask, "당기"].dropna()
        if len(direct_label) > 0:
            return float(direct_label.iloc[0])

        # (4) 자본총계/자본의총계 라벨
        mask_total = sub["항목명_clean"].isin(["자본총계", "자본의총계"])
        equity_total = sub.loc[mask_total, "당기"].dropna()
        if len(equity_total) > 0:
            if len(nci) > 0:
                return float(equity_total.iloc[0] - nci.iloc[0])
            else:
                return float(equity_total.iloc[0])

        # (5) 마지막 fallback: Equity만 있으면
        if len(equity) > 0:
            return float(equity.iloc[0])

        return pd.NA

    else:
        # ---------------- 별도 기준 ----------------
        sub = sub[sub["연결구분"] == "별도"]
        
        # (1) Equity 우선
        equity = sub.loc[sub["항목코드_통일"] == "Equity", "당기"].dropna()
        if len(equity) > 0:
            return float(equity.iloc[0])
        
        # (2) 항목명 자본총계 / 자본의총계
        mask_total = sub["항목명_clean"].isin(["자본총계", "자본의총계"])
        equity_total = sub.loc[mask_total, "당기"].dropna()
        if len(equity_total) > 0:
            return float(equity_total.iloc[0])
        
        # (3) 마지막 fallback: Assets - Liabilities
        assets = sub.loc[sub["항목코드_통일"] == "Assets", "당기"].dropna()
        liab   = sub.loc[sub["항목코드_통일"] == "Liabilities", "당기"].dropna()
        if len(assets) > 0 and len(liab) > 0:
            return float(assets.iloc[0] - liab.iloc[0])

        return pd.NA






# 3) 회사×결산일 그룹별 적용
owner_equity_vals = (
    df_non_fin.groupby(["종목코드", "결산기준일"])
              .apply(get_owner_equity)
)

# 4) skeleton 업데이트
skeleton.loc[owner_equity_vals.index, "지배주주지분"] = owner_equity_vals.values


# 확인
print(skeleton.head(20))

# 숫자 변환 시도: 숫자가 아니면 NaN으로 바뀜
col = pd.to_numeric(skeleton["지배주주지분"], errors="coerce")

# NA 탐지 (원래 NA + 변환 실패한 문자/공백 포함)
na_or_non_numeric = col.isna()

print(na_or_non_numeric.sum(), "개 NA/빈칸/문자값")

# 해당 행 추출
na_cases = skeleton[na_or_non_numeric]

# 회사코드와 결산기준일만 출력
na_list = na_cases.index.to_frame(index=False)
print("지배주주지분 NA/빈칸/문자 케이스 수:", len(na_list))
print(na_list.head(20))







def get_total_assets(sub: pd.DataFrame) -> float:
    """
    sub: 특정 회사×결산기준일 DataFrame
    return: 자산총계 값 또는 NA
    """
    # (1) IFRS 태그 직접
    assets = sub.loc[sub["항목코드_통일"] == "Assets", "당기"].dropna()
    if len(assets) > 0:
        return float(assets.iloc[0])

    # (2) 대체 IFRS 태그 (부채+자본총계)
    eq_and_liab = sub.loc[sub["항목코드_통일"] == "EquityAndLiabilities", "당기"].dropna()
    if len(eq_and_liab) > 0:
        return float(eq_and_liab.iloc[0])

    # (3) 항목명 라벨 기반
    asset_labels = ["자산총계", "자산의총계"]
    mask = sub["항목명_clean"].isin(asset_labels)
    direct_label = sub.loc[mask, "당기"].dropna()
    if len(direct_label) > 0:
        return float(direct_label.iloc[0])

    # (4) Liabilities + Equity 역산
    liab = sub.loc[sub["항목코드_통일"] == "Liabilities", "당기"].dropna()
    equity = sub.loc[sub["항목코드_통일"] == "Equity", "당기"].dropna()
    if len(liab) > 0 and len(equity) > 0:
        return float(liab.iloc[0] + equity.iloc[0])

    # (5) 실패 시
    return pd.NA





# 회사×결산일 그룹별 적용
assets_vals = (
    df_non_fin.groupby(["종목코드", "결산기준일"])
              .apply(get_total_assets)
)

# skeleton 업데이트
skeleton.loc[assets_vals.index, "자산총계"] = assets_vals.values



# 확인
print(skeleton.head(20))

# 숫자 변환 시도: 숫자가 아니면 NaN으로 바뀜
col = pd.to_numeric(skeleton["자산총계"], errors="coerce")

# NA 탐지 (원래 NA + 변환 실패한 문자/공백 포함)
na_or_non_numeric = col.isna()

print(na_or_non_numeric.sum(), "개 NA/빈칸/문자값")

# 해당 행 추출
na_cases = skeleton[na_or_non_numeric]

# 회사코드와 결산기준일만 출력
na_list = na_cases.index.to_frame(index=False)
print("자산총계 NA/빈칸/문자 케이스 수:", len(na_list))
print(na_list.head(20))




def get_gross_profit(sub: pd.DataFrame) -> float:
    """
    sub: 특정 회사×결산기준일 DataFrame (손익계산서 행들)
    return: 매출총이익 값 또는 NA
    """
    # (1) IFRS 태그 직접 (GrossProfit)
    gp = sub.loc[sub["항목코드_통일"] == "GrossProfit", "당기"].dropna()
    if len(gp) > 0:
        return float(gp.iloc[0])
    
    # (2) Revenue - CostOfSales
    rev = sub.loc[sub["항목코드_통일"] == "Revenue", "당기"].dropna()
    cogs = sub.loc[sub["항목코드_통일"] == "CostOfSales", "당기"].dropna()
    if len(rev) > 0 and len(cogs) > 0:
        return float(rev.iloc[0] - cogs.iloc[0])
    
    # (3) 항목명 라벨 기반 ('매출총이익')
    gp_label = sub.loc[sub["항목명_clean"].str.contains("매출총이익", na=False), "당기"].dropna()
    if len(gp_label) > 0:
        return float(gp_label.iloc[0])
    
    # (4) 보조: '매출액/매출원가' 라벨 기반
    rev_label = sub.loc[sub["항목명_clean"].str.contains("매출액", na=False), "당기"].dropna()
    cogs_label = sub.loc[sub["항목명_clean"].str.contains("매출원가", na=False), "당기"].dropna()
    if len(rev_label) > 0 and len(cogs_label) > 0:
        return float(rev_label.iloc[0] - cogs_label.iloc[0])
    
    # (5) 매출액 - (재료비 + 인건비)
    # 5-1) IFRS 태그 기준
    raw_mat = sub.loc[sub["항목코드_통일"] == "RawMaterialsAndConsumablesUsed", "당기"].dropna()
    labor   = sub.loc[sub["항목코드_통일"] == "EmployeeBenefitsExpense", "당기"].dropna()
    if len(rev) > 0 and (len(raw_mat) > 0 or len(labor) > 0):
        total_cost = 0
        if len(raw_mat) > 0:
            total_cost += raw_mat.iloc[0]
        if len(labor) > 0:
            total_cost += labor.iloc[0]
        return float(rev.iloc[0] - total_cost)
    
    # 5-2) 항목명 클린 기반
    if len(rev_label) > 0:
        mat_cost = sub.loc[sub["항목명_clean"].str.contains("재료비", na=False), "당기"].dropna()
        if len(mat_cost) > 0:
            total_cost = mat_cost.iloc[0]
            return float(rev_label.iloc[0] - total_cost)

    
    # (6) 최후 fallback: '영업수익 - 영업비용'
    oper_rev = sub.loc[sub["항목명_clean"].str.contains("영업수익", na=False), "당기"].dropna()
    oper_exp = sub.loc[sub["항목명_clean"].str.contains("영업비용", na=False), "당기"].dropna()
    if len(oper_rev) > 0 and len(oper_exp) > 0:
        return float(oper_rev.iloc[0] - oper_exp.iloc[0])
    
    return pd.NA












# 회사 × 결산일 그룹별 적용
gp_vals = (
    df_non_fin.groupby(["종목코드", "결산기준일"])
              .apply(get_gross_profit)
)

# skeleton에 매출총이익 컬럼 업데이트
skeleton.loc[gp_vals.index, "매출총이익"] = gp_vals.values





# 확인
print(skeleton.head(20))

# 숫자 변환 시도: 숫자가 아니면 NaN으로 바뀜
col = pd.to_numeric(skeleton["매출총이익"], errors="coerce")

# NA 탐지 (원래 NA + 변환 실패한 문자/공백 포함)
na_or_non_numeric = col.isna()

print(na_or_non_numeric.sum(), "개 NA/빈칸/문자값")

# 해당 행 추출
na_cases = skeleton[na_or_non_numeric]

# 회사코드와 결산기준일만 출력
na_list = na_cases.index.to_frame(index=False)
print("매출총이익 NA/빈칸/문자 케이스 수:", len(na_list))
print(na_list.head(20))




# ----------------------------------
# 스켈레톤 저장하기
# ----------------------------------
OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터재무데이터.parquet"

# Parquet으로 저장 (압축 옵션도 가능)
skeleton.to_parquet(OUT_FILE, engine="pyarrow", compression="snappy")

print(f"완료! {OUT_FILE} 저장됨")

