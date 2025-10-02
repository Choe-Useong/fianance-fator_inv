from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

# Config
ROOT: str = '.'                 # search start folder
PATTERN: str = '*.txt'          # source file pattern
OUTPUT_DIR: str = '?듯빀'        # direct output root
ENCODING: str = 'cp949'


def _norm_token(s: str) -> str:
    return ' '.join((s or '').split()).strip()


def _is_numeric_token(tok: str) -> bool:
    return tok.isdigit()


def _parse_type_and_sector_from_name(path: Path) -> Tuple[str, str] | None:
    base = path.stem
    parts = base.split('_')
    if len(parts) < 4:
        return None
    typ = parts[3]
    sector = '?쇰컲'
    if len(parts) >= 5:
        tok5 = parts[4]
        if tok5 and tok5 != '?곌껐' and not _is_numeric_token(tok5):
            sector = tok5
    return typ, sector


def _collect_input_files(root: str, pattern: str, exclude_dirs: List[Path]) -> List[Path]:
    root_path = Path(root).resolve()
    files = sorted((p for p in root_path.rglob(pattern) if p.is_file()), key=lambda p: str(p).lower())
    ex_res = [d.resolve() for d in exclude_dirs]
    out: List[Path] = []
    for p in files:
        rp = p.resolve()
        if any(d in rp.parents for d in ex_res):
            continue
        out.append(p)
    return out


def _select_indices(header: List[str]) -> Tuple[List[int], int]:
    norm = [_norm_token(h) for h in header]
    try:
        idx_item = norm.index('??ぉ紐?)
    except ValueError:
        idx_item = len(norm) - 1
    idx_dangi = -1
    for j in range(idx_item + 1, len(norm)):
        if norm[j].startswith('?밴린'):
            idx_dangi = j
            break
    keep_idx = list(range(0, idx_item + 1))
    if idx_dangi >= 0:
        keep_idx.append(idx_dangi)
    return keep_idx, idx_item


def _parse_stmt_details(text: str) -> Tuple[str, str]:
    """Parse details from 재무제표종류.

    Input examples:
      - "손익계산서, 기능별 분류 - 별도재무제표"
      - "재무상태표, 유동/비유동법-연결재무제표"

    Returns:
      (재무제표분류, 연결구분)
        재무제표분류: 콤마(,) 뒤 하이픈(-) 앞 텍스트. 하이픈 없으면 콤마 뒤 전체.
        연결구분:     하이픈(-) 뒤 텍스트. 없으면 빈 문자열.
    """
    t = _norm_token(text)
    if not t:
        return '', ''

    after = ''
    if ',' in t:
        after = t.split(',', 1)[1].strip()

    kind, scope = '', ''
    if after:
        if '-' in after:
            kind, scope = [x.strip() for x in after.split('-', 1)]
        else:
            kind = after.strip()

    return kind, scope
