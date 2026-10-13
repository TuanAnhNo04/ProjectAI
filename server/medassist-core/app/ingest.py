"""Nap danh muc thuoc tu file CUC BO (Day 18).

QUY TAC DU LIEU (AGENTS.md muc 3: khong duoc bia nguon y khoa):
1. KHONG tu sinh ten thuoc, hoat chat hay tuong tac thuoc. Moi thu deu doc
   tu file CSV co san trong `data/`.
2. Moi ban ghi mang `source` tro ve DUNG file da nap va co
   `is_authoritative=False` - vi du lieu dev KHONG phai nguon lam sang.
3. `interactions` co y de trong: khong co nguon de kiem chung thi khong dung.
   Xem `has_interaction_data()`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.sqlite_repository import normalize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DEV_SOURCE_LABEL = "dev-sample: data/medicines.dev.csv"

#: Nguong tin cay cho cau tra loi co trich dan. Du lieu dev KHONG dat nguong nay
#: -> agent phai tu choi tra loi y khoa.
MIN_AUTHORITATIVE_SCORE = 0.8


@dataclass
class IngestedDrug:
    """Mot dong thuoc sau khi chuan hoa."""

    name: str
    composition: str | None
    uses: str | None
    side_effects: str | None
    manufacturer: str | None = None
    ingredients: list[str] = field(default_factory=list)
    source: str = DEV_SOURCE_LABEL
    is_authoritative: bool = False


def split_ingredients(composition: str | None) -> list[str]:
    """Tach hoat chat tu chuoi thanh phan (nhan biet ca dau phay va cham phay)."""
    if not composition:
        return []
    parts = composition.replace(";", ",").split(",")
    return [normalize(p) for p in parts if normalize(p)]


def _to_percent(value: Any) -> float | None:
    """Doc phan tram; o rong -> None (khong ep thanh 0 de khong bia so lieu)."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except ValueError:
        return None


def _to_text(value: Any) -> str | None:
    """Chuoi rong -> None de UI hien 'khong co thong tin' thay vi in chuoi rong."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_csv(path: str | Path | None = None) -> list[IngestedDrug]:
    """Doc CSV danh muc thuoc cuc bo."""
    csv_path = Path(path) if path else DATA_DIR / "medicines.dev.csv"
    if not csv_path.exists():
        return []

    drugs: list[IngestedDrug] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = _to_text(row.get("Medicine_Name"))
            if not name:
                # Dong khong co ten -> bo, KHONG tu dat ten thuoc.
                continue
            composition = _to_text(row.get("Composition"))
            drugs.append(
                IngestedDrug(
                    name=name,
                    composition=composition,
                    uses=_to_text(row.get("Uses")),
                    side_effects=_to_text(row.get("Side_effects")),
                    manufacturer=_to_text(row.get("Manufacturer")),
                    ingredients=split_ingredients(composition),
                    source=f"dev-sample: {csv_path.name}",
                    is_authoritative=False,
                )
            )
    return drugs


def has_interaction_data() -> bool:
    """Co bang tuong tac thuoc DANG TIN hay khong?

    Hien LUON la False: khong co nguon de kiem chung tuong tac thuoc thi khong
    dung bang nay. Agent PHAI tu choi cau hoi ve tuong tac thuoc.
    """
    return False


def to_rows(drugs: list[IngestedDrug]) -> list[dict[str, Any]]:
    """Chuyen sang dict de dua vao kho (bo sung khoa con thieu bang None)."""
    return [
        {
            "name": d.name,
            "composition": d.composition,
            "uses": d.uses,
            "side_effects": d.side_effects,
            "manufacturer": d.manufacturer,
            "excellent_review": _to_percent(None),
            "average_review": _to_percent(None),
            "poor_review": _to_percent(None),
            "ingredients": json.dumps(d.ingredients),
            "source": d.source,
            "is_authoritative": d.is_authoritative,
        }
        for d in drugs
    ]
