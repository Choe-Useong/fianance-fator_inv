import pandas as pd
import numpy as np
import re

# 1) 데이터 로드
FILE_PATH = r"C:\Users\admin\Desktop\재무제표정리\통합재무\주재무_사업보고서_코스피코스닥.parquet"
df = pd.read_parquet(FILE_PATH, engine="pyarrow")


df["당기"] = (
    df["당기"]
    .astype(str)                   # 혹시 모를 object → str 변환
    .str.replace(",", "", regex=False)  # 쉼표 제거
    .replace(["None", "NA", ""], np.nan)  # 이상치 처리
    .astype(float)                 # float 변환
)


# 2) 비금융사만 필터링
exclude_industries = ["은행및저축기관", "금융지원서비스업", "보험업",'신탁업및집합투자업','기타금융업']
df_non_fin = df[~df["업종명"].isin(exclude_industries)].copy()

# 결산월이 12월인 것만 선택
df_non_fin = df_non_fin[df_non_fin["결산월"] == '12']

# 항목명 전처리 함수
def clean_label(x: str) -> str:
    if pd.isna(x):
        return ""
    x = re.sub(r"\s+", "", x)          # 공백 제거
    x = re.sub(r"[^가-힣()]", "", x)    # 한글/괄호만 유지 (로마자/숫자/기호 제거)
    return x

df_non_fin["항목명_clean"] = df_non_fin["항목명"].apply(clean_label)





# 3) 기준 집합: 종목코드 × 결산기준일 + 업종명 포함
base = (
    df_non_fin[["종목코드", "결산기준일", "업종명"]]
    .drop_duplicates()
    .sort_values(["종목코드", "결산기준일"])
)

# 4) 멀티인덱스 만들기 (업종명은 컬럼으로 남김)
idx = pd.MultiIndex.from_frame(base[["종목코드", "결산기준일"]],
                               names=["종목코드", "결산기준일"])

# 빈 스켈레톤 생성 (인덱스만 유지)
skeleton = pd.DataFrame(index=idx).reset_index()

# base 병합
skeleton = skeleton.merge(base, on=["종목코드", "결산기준일"], how="left").set_index(["종목코드", "결산기준일"])

# 필요한 컬럼을 순차적으로 추가
skeleton["지배주주지분"] = pd.Series(dtype="Float64")
skeleton["매출총이익"] = pd.Series(dtype="Float64")
skeleton["자산총계"] = pd.Series(dtype="Float64")
skeleton["지배주주순이익"] = pd.Series(dtype="Float64")

print(skeleton.head())




##########################################################################

# 1) 지배주주지분 관련 라벨 후보
owner_labels = [
    "지배기업소유주지분","지배기업의소유주지분","지배기업의소유주에게귀속되는자본",
    "지배기업의소유지분","지배회사소유지분","지배기업소유지분", '지배기업소유주지분',
    "지배기업소유주에귀속되는자본","지배주주지분","지배기업소유지분합계",
    "지배기업지분","지배기업의소유주에게귀속되는지분","지배기업주주지분"
]


def get_owner_equity(sub: pd.DataFrame) -> float:
    """
    연결일 경우 → 지배주주지분 기준으로 계산
    별도일 경우 → 자본총계 또는 Equity 기준으로 계산

    [연결 재무제표 처리 로직]
      - IFRS 코드 'EquityAttributableToOwnersOfParent' 존재 시: 지배주주귀속자본을 직접 사용.
      - 없을 경우 총자본(Equity) - 비지배지분(NoncontrollingInterests)으로 계산.
      - 항목명에 '지배기업지분', '지배주주지분' 등 포함 시 해당 값을 사용.
      - 위 값이 없으면 자본총계(또는 자본의총계) - 비지배지분 으로 대체.
      - 모든 경우에 실패하면 마지막으로 'Equity' 항목만 단독 사용.

    [별도 재무제표 처리 로직]
      - IFRS 코드 'Equity' 존재 시 해당 값을 사용.
      - 없을 경우 항목명이 '자본총계' 또는 '자본의총계'인 값을 사용.
      - 위 항목들이 모두 없으면 자산총계(Assets) - 부채총계(Liabilities)로 계산.

    반환값:
      - float 형 자기자본(지배주주기준)
      - 찾을 수 없는 경우 pd.NA 반환
    """

    # ---------------- 연결 기준 ----------------
    if (sub["연결구분"] == "연결").any():
        sub = sub[sub["연결구분"] == "연결"]

        # 재무상태표만 사용
        if "제표종류앞" not in sub.columns:
            return pd.NA
        sub = sub[sub["제표종류앞"].astype(str) == "재무상태표"]
        if sub.empty:
            return pd.NA

        # (1) IFRS 태그: 지배기업지분
        direct = sub.loc[sub["항목코드_통일"] == "EquityAttributableToOwnersOfParent", "당기"].dropna()
        if len(direct) > 0:
            return float(direct.iloc[0])

        # (2) Equity - NCI
        equity = sub.loc[sub["항목코드_통일"] == "Equity", "당기"].dropna()
        nci = sub.loc[sub["항목코드_통일"] == "NoncontrollingInterests", "당기"].dropna()
        if len(equity) > 0 and len(nci) > 0:
            return float(equity.iloc[0] - nci.iloc[0])

        # (3) 항목명 기반 지배주주지분 매칭
        mask = sub["항목명_clean"].isin(owner_labels)
        val = sub.loc[mask, "당기"].dropna()
        if len(val) > 0:
            return float(val.iloc[0])

        # (4) 자본총계 - NCI (Fallback)
        mask_total = sub["항목명_clean"].isin(["자본총계", "자본의총계"])
        equity_total = sub.loc[mask_total, "당기"].dropna()
        if len(equity_total) > 0:
            if len(nci) > 0:
                return float(equity_total.iloc[0] - nci.iloc[0])
            else:
                return float(equity_total.iloc[0])

        # (5) 마지막 fallback: Equity만 존재
        if len(equity) > 0:
            return float(equity.iloc[0])

        return pd.NA

    # ---------------- 별도 기준 ----------------
    elif (sub["연결구분"] == "별도").any():
        sub = sub[sub["연결구분"] == "별도"]

        # 재무상태표만 사용
        if "제표종류앞" not in sub.columns:
            return pd.NA
        sub = sub[sub["제표종류앞"].astype(str) == "재무상태표"]
        if sub.empty:
            return pd.NA

        # (1) IFRS 태그: Equity
        equity = sub.loc[sub["항목코드_통일"] == "Equity", "당기"].dropna()
        if len(equity) > 0:
            return float(equity.iloc[0])

        # (2) 항목명 자본총계 / 자본의총계
        mask_total = sub["항목명_clean"].isin(["자본총계", "자본의총계"])
        equity_total = sub.loc[mask_total, "당기"].dropna()
        if len(equity_total) > 0:
            return float(equity_total.iloc[0])

        # (3) 마지막 fallback: 자산 - 부채
        assets = sub.loc[sub["항목코드_통일"] == "Assets", "당기"].dropna()
        liab = sub.loc[sub["항목코드_통일"] == "Liabilities", "당기"].dropna()
        if len(assets) > 0 and len(liab) > 0:
            return float(assets.iloc[0] - liab.iloc[0])

        return pd.NA

    # ---------------- 둘 다 없으면 ----------------
    else:
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
    특정 회사×결산기준일 단위(sub DataFrame)에서 '자산총계' 값을 추출한다.
    재무상태표(대차대조표) 데이터만 사용하도록 필터링 포함.

    [로직 구조]

      (0) '제표종류앞' 컬럼 필터링
          - 재무상태표가 아닌 손익계산서·현금흐름표 행 제거.
          - 제표종류앞 컬럼이 없거나 재무상태표 데이터가 없으면 NA 반환.

      ① IFRS 코드 'Assets' 우선
      ② IFRS 코드 'EquityAndLiabilities' 대체 사용
      ③ 항목명('자산총계', '자산의총계') 기반 탐색
      ④ Liabilities + Equity 역산 (A = L + E)
      ⑤ 모든 단계 실패 시 pd.NA 반환
    """
    # (0) 재무상태표 행만 사용
    if "제표종류앞" not in sub.columns:
        return pd.NA
    sub = sub[sub["제표종류앞"].astype(str) == "재무상태표"]
    if sub.empty:
        return pd.NA

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
    특정 회사×결산기준일 단위(sub DataFrame)에서 '매출총이익(Gross Profit)' 값을 추출한다.
    손익계산서 또는 포괄손익계산서 데이터만 사용.

    [로직 구조]

      (0) '제표종류앞' 필터링
          - 손익계산서/포괄손익계산서가 아닌 데이터 제거.
          - '제표종류앞' 컬럼이 없거나 해당 데이터가 없으면 NA 반환.

      ① IFRS 코드 'GrossProfit'
          - 표준 손익계산서 태그로, 매출총이익이 명시된 경우 직접 사용.

      ② IFRS 코드 'Revenue' - 'CostOfSales'
          - 매출총이익이 명시되지 않은 경우, 매출액과 매출원가를 통해 직접 계산.

      ③ 항목명 기반 ('매출총이익')
          - 한글 라벨이 '매출총이익'인 경우 직접 사용.

      ④ 항목명 '매출액' - '매출원가'
          - IFRS 코드가 누락된 경우 한글 항목명을 이용해 간접 계산.

      ⑤ 보조: '매출액 - (재료비 + 인건비)'
          - 원가 세부 항목이 분리되어 있는 경우 이를 합산하여 매출총이익 근사값 계산.
          - IFRS 코드 'RawMaterialsAndConsumablesUsed' / 'EmployeeBenefitsExpense' 또는
            한글 항목명 '재료비', '인건비'를 기반으로 계산.

      ⑥ 최후 fallback: '영업수익 - 영업비용'
          - 매출액/원가 항목이 없고, 영업수익·영업비용만 존재할 경우 사용.
          - 매출총이익 근사치로 간주.

      반환값:
        - float 형 매출총이익 (Gross Profit)
        - 찾을 수 없는 경우 pd.NA 반환
    """

    # (0) 손익계산서/포괄손익계산서 행만 사용
    if "제표종류앞" not in sub.columns:
        return pd.NA
    sub = sub[sub["제표종류앞"].astype(str).str.contains("손익계산서|포괄손익계산서", na=False)]
    if sub.empty:
        return pd.NA

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




def get_owner_net_income(sub: pd.DataFrame) -> float:
    """
    지배주주순이익 추출: 연결이 존재하면 연결 기준, 없으면 별도 기준.
    우선순위
      - 연결: ProfitLossAttributableToOwnersOfParent → ProfitLoss - ProfitLossAttributableToNoncontrollingInterests → 라벨(지배주주순이익 계열) → 라벨(당기순이익/연결당기순이익) - 비지배주주순이익 → ProfitLoss
      - 별도: ProfitLoss → 라벨(당기순이익/순이익)
    """
    has_consol = (sub["연결구분"] == "연결").any()

    if has_consol:
        sub = sub[sub["연결구분"] == "연결"]
        # 손익/포괄손익 재무제표만 사용 (없으면 계산하지 않음)
        stmt_col_candidates = ["재무제표명", "재무제표종류", "재무제표구분"]
        stmt_col_pref = next((c for c in stmt_col_candidates if c in sub.columns), None)
        if not stmt_col_pref:
            return pd.NA
        sub_stmt = sub[sub[stmt_col_pref].astype(str).str.contains("손익", na=False)]
        if sub_stmt.empty:
            return pd.NA
        sub = sub_stmt

        direct = sub.loc[sub["항목코드_통일"] == "ProfitLossAttributableToOwnersOfParent", "당기"].dropna()
        if len(direct) > 0:
            return float(direct.iloc[0])

        total = sub.loc[sub["항목코드_통일"] == "ProfitLoss", "당기"].dropna()
        nci_attr = sub.loc[sub["항목코드_통일"] == "ProfitLossAttributableToNoncontrollingInterests", "당기"].dropna()
        if len(total) > 0 and len(nci_attr) > 0:
            return float(total.iloc[0] - nci_attr.iloc[0])

        owner_ni_labels = [
            "지배주주순이익","지배주주순이익(손실)","지배주주순이익(순손실)",
            "지배기업소유주순이익","지배기업소유주순이익(손실)","지배기업소유주순이익(순손실)",
            "지배기업의소유주에게귀속되는당기순이익","지배기업의소유주에게귀속되는당기순이익(손실)","지배기업의소유주에게귀속되는당기순이익(순손실)",
            "지배기업소유주에게귀속되는순이익","지배기업소유주에게귀속되는순이익(손실)","지배기업소유주에게귀속되는순이익(순손실)",
            "지배주주에귀속되는당기순이익","지배주주에귀속되는당기순이익(손실)","지배주주에귀속되는당기순이익(순손실)",
            "지배기업주주지분에귀속되는당기순이익","지배기업주주지분에귀속되는당기순이익(손실)","지배기업주주지분에귀속되는당기순이익(순손실)",
            '지분법적용대상인관계기업의당기순손익에대한지분'
        ]
        mask = sub["항목명_clean"].isin(owner_ni_labels)
        direct_label = sub.loc[mask, "당기"].dropna()
        if len(direct_label) > 0:
            return float(direct_label.iloc[0])

        # (3b) 손익/포괄손익표 내 '지배주주지분' 계열 라벨을 순이익으로 간주
        stmt_col_candidates = ["재무제표명", "재무제표종류", "재무제표구분"]
        stmt_col = next((c for c in stmt_col_candidates if c in sub.columns), None)
        if stmt_col:
            if sub[stmt_col].astype(str).str.contains("손익", na=False).any():
                eq_like_mask = sub["항목명_clean"].isin(owner_labels)
                eq_like = sub.loc[eq_like_mask, "당기"].dropna()
                if len(eq_like) > 0:
                    return float(eq_like.iloc[0])

        # '당기순이익(손실)' 같은 케이스 포함을 위해 부분매칭 사용
        total_label = sub.loc[
            sub["항목명_clean"].isin([
                "연결당기순이익","연결당기순이익(손실)","연결당기순이익(순손실)",
                "당기순이익","당기순이익(손실)","당기순이익(순손실)",
                "순이익","순이익(손실)","순이익(순손실)",
            ]),
            "당기"
        ].dropna()
        if len(total_label) > 0:
            if len(nci_attr) > 0:
                return float(total_label.iloc[0] - nci_attr.iloc[0])
            nci_label = sub.loc[
                sub["항목명_clean"].isin([
                    "비지배주주순이익","비지배주주순이익(손실)","비지배주주순이익(순손실)",
                    "비지배지분순이익","비지배지분순이익(손실)","비지배지분순이익(순손실)",
                    "비지배주주에귀속되는당기순이익","비지배주주에귀속되는당기순이익(손실)","비지배주주에귀속되는당기순이익(순손실)",
                ]),
                "당기"
            ].dropna()
            if len(nci_label) > 0:
                return float(total_label.iloc[0] - nci_label.iloc[0])
            return float(total_label.iloc[0])

        if len(total) > 0:
            return float(total.iloc[0])

        return pd.NA

    else:
        sub = sub[sub["연결구분"] == "별도"]
        # 손익/포괄손익 재무제표만 사용 (없으면 계산하지 않음)
        stmt_col_candidates = ["재무제표명", "재무제표종류", "재무제표구분"]
        stmt_col_pref = next((c for c in stmt_col_candidates if c in sub.columns), None)
        if not stmt_col_pref:
            return pd.NA
        sub_stmt = sub[sub[stmt_col_pref].astype(str).str.contains("손익", na=False)]
        if sub_stmt.empty:
            return pd.NA
        sub = sub_stmt
        direct = sub.loc[sub["항목코드_통일"] == "ProfitLoss", "당기"].dropna()
        if len(direct) > 0:
            return float(direct.iloc[0])

        # (1b) 손익/포괄손익표 내 '지배주주지분' 계열 라벨을 순이익으로 간주
        stmt_col_candidates = ["재무제표명", "재무제표종류", "재무제표구분"]
        stmt_col = next((c for c in stmt_col_candidates if c in sub.columns), None)
        if stmt_col:
            if sub[stmt_col].astype(str).str.contains("손익", na=False).any():
                eq_like_mask = sub["항목명_clean"].isin(owner_labels)
                eq_like = sub.loc[eq_like_mask, "당기"].dropna()
                if len(eq_like) > 0:
                    return float(eq_like.iloc[0])

        # 별도에서도 '당기순이익(손실)' 등 부분 포함 허용
        label = sub.loc[
            sub["항목명_clean"].isin([
                "당기순이익","당기순이익(손실)","당기순이익(순손실)",
                "순이익","순이익(손실)","순이익(순손실)"
            ]),
            "당기"
        ].dropna()
        if len(label) > 0:
            return float(label.iloc[0])

        return pd.NA


# 회사 × 결산일 그룹별 적용 (지배주주순이익)
owner_ni_vals = (
    df_non_fin.groupby(["종목코드", "결산기준일"]).apply(get_owner_net_income)
)
skeleton.loc[owner_ni_vals.index, "지배주주순이익"] = owner_ni_vals.values

# 간단 진단 출력
col = pd.to_numeric(skeleton["지배주주순이익"], errors="coerce")
na_or_non_numeric = col.isna()
print(na_or_non_numeric.sum(), "개 NA/빈칸/문자값")
na_cases = skeleton[na_or_non_numeric]
na_list = na_cases.index.to_frame(index=False)
print("지배주주순이익 NA/빈칸/문자 케이스 수:", len(na_list))
print(na_list.head(20))




# ----------------------------------
# 스켈레톤 저장하기
# ----------------------------------
OUT_FILE = r"C:\Users\admin\Desktop\재무제표정리\통합재무\팩터재무데이터.parquet"

# Parquet으로 저장 (압축 옵션도 가능)
skeleton.to_parquet(OUT_FILE, engine="pyarrow", compression="snappy")

print(f"완료! {OUT_FILE} 저장됨")
