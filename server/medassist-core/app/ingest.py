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

from app.provenance import clean_date, clean_text
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
    source_url: str | None = None
    retrieved_at: str | None = None
    license: str | None = None
    reviewed_by: str | None = None


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


#: Tên cột mặc định (bộ dữ liệu thuốc của app gốc). Đổi bằng `--col` khi nguồn khác tên cột.
DEFAULT_COLUMNS = {
    "name": "Medicine_Name",
    "composition": "Composition",
    "uses": "Uses",
    "side_effects": "Side_effects",
    "manufacturer": "Manufacturer",
}


@dataclass
class IngestReport:
    """Thống kê một lần nạp — **chỉ số liệu**, tuyệt đối không chứa nội dung dòng.

    ⚠️ Dữ liệu thật có thể nhạy cảm/bản quyền; báo cáo này được in ra màn hình và có
    thể bị dán vào chat hay log, nên không bao giờ mang tên thuốc hay văn bản nguồn.
    """

    rows_read: int = 0
    imported: int = 0
    skipped_no_name: int = 0
    duplicates: int = 0
    truncated_fields: int = 0
    missing_columns: list[str] = field(default_factory=list)
    columns_found: list[str] = field(default_factory=list)


class IngestError(ValueError):
    """File nguồn không dùng được (thiếu cột bắt buộc, không đọc được...)."""


def load_csv_mapped(
    path: str | Path,
    columns: dict[str, str] | None = None,
    *,
    source: str,
    authoritative: bool = False,
    dedupe: bool = True,
    source_url: str | None = None,
    retrieved_at: str | None = None,
    license: str | None = None,
    reviewed_by: str | None = None,
    max_field_len: int = 4000,
) -> tuple[list[IngestedDrug], IngestReport]:
    """Đọc CSV danh mục thuốc **thật** theo ánh xạ cột, kèm báo cáo.

    - `source` bắt buộc (nhãn trỏ về đúng nguồn, ghi vào từng bản ghi).
    - Không tự sinh/sửa nội dung y khoa: ô rỗng -> `None`; ô quá dài bị cắt và đếm.
    - Dòng không tên bị bỏ; trùng (cùng tên + thành phần sau chuẩn hoá) bị bỏ nếu `dedupe`.
    - Đọc `utf-8-sig` để chịu được file Excel có BOM.
    """
    if not source or not source.strip():
        raise IngestError("Thiếu `source`: mọi bản ghi phải trỏ về một nguồn cụ thể.")
    if source_url and not source_url.strip().lower().startswith("https://"):
        raise IngestError("--source-url phải là https.")
    if authoritative and not (reviewed_by and reviewed_by.strip()):
        raise IngestError("Nguồn lâm sàng (authoritative) cần reviewed_by.")
    prov = {
        "source_url": (source_url or "").strip() or None,
        "retrieved_at": clean_date(retrieved_at),
        "license": clean_text(license),
        "reviewed_by": clean_text(reviewed_by),
    }
    if retrieved_at and not prov["retrieved_at"]:
        raise IngestError("--retrieved-at phải là ngày ISO (YYYY-MM-DD).")
    csv_path = Path(path)
    if not csv_path.is_file():
        raise IngestError(f"Không thấy file: {csv_path.name}")

    cols = {**DEFAULT_COLUMNS, **(columns or {})}
    report = IngestReport()
    drugs: list[IngestedDrug] = []
    seen: set[tuple[str, str]] = set()

    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            report.columns_found = list(reader.fieldnames or [])
            report.missing_columns = [
                f"{key}={col}" for key, col in cols.items() if col not in report.columns_found
            ]
            if cols["name"] not in report.columns_found:
                raise IngestError(
                    f"Thiếu cột tên thuốc '{cols['name']}'. Cột có trong file: "
                    + ", ".join(report.columns_found or ["(không có)"])
                )
            for row in reader:
                report.rows_read += 1
                name = _to_text(row.get(cols["name"]))
                if not name:
                    report.skipped_no_name += 1
                    continue

                def field_of(key: str) -> str | None:
                    text = _to_text(row.get(cols[key]))
                    if text and len(text) > max_field_len:
                        report.truncated_fields += 1
                        return text[:max_field_len]
                    return text

                composition = field_of("composition")
                if dedupe:
                    key = (normalize(name), normalize(composition or ""))
                    if key in seen:
                        report.duplicates += 1
                        continue
                    seen.add(key)
                drugs.append(
                    IngestedDrug(
                        name=name[:max_field_len],
                        composition=composition,
                        uses=field_of("uses"),
                        side_effects=field_of("side_effects"),
                        manufacturer=field_of("manufacturer"),
                        ingredients=split_ingredients(composition),
                        source=source.strip(),
                        is_authoritative=authoritative,
                        **prov,
                    )
                )
    except UnicodeDecodeError as exc:
        raise IngestError("File không phải UTF-8. Lưu lại dạng CSV UTF-8 rồi thử lại.") from exc
    except csv.Error as exc:
        raise IngestError(f"File CSV hỏng: {type(exc).__name__}") from exc

    report.imported = len(drugs)
    return drugs, report


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
            "source_url": d.source_url,
            "retrieved_at": d.retrieved_at,
            "license": d.license,
            "reviewed_by": d.reviewed_by,
        }
        for d in drugs
    ]
