"""Nạp danh mục thuốc MẪU (dữ liệu giả) vào CSDL dev.

Chạy: `python -m app.seed` (đọc `MEDASSIST_DB`; thiếu thì dùng `data/medassist.dev.db`).

⚠️ Chỉ dùng cho **dev**: `data/medicines.dev.csv` là dữ liệu mẫu, mọi bản ghi mang
`is_authoritative=False` nên agent sẽ từ chối trả lời y khoa dựa trên nó (quyết định D3).
"""

from __future__ import annotations

import os
from pathlib import Path

from app.ingest import load_csv, to_rows
from app.sqlite_repository import SQLiteRepository

DEFAULT_DEV_DB = Path(__file__).resolve().parent.parent / "data" / "medassist.dev.db"


def seed_if_empty(repo: SQLiteRepository) -> int:
    """Nạp CSV mẫu nếu bảng thuốc đang rỗng; trả số bản ghi đã nạp (0 nếu đã có)."""
    if repo.list_medicines(1, 0).total > 0:
        return 0
    return repo.add_medicines(to_rows(load_csv()))


def main() -> None:
    db_path = Path(os.environ.get("MEDASSIST_DB") or DEFAULT_DEV_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    repo = SQLiteRepository(db_path)
    try:
        added = seed_if_empty(repo)
    finally:
        repo.close()
    print(f"seed: {added} bản ghi mới vào {db_path.name}")


if __name__ == "__main__":
    main()
