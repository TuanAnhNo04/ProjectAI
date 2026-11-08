"""Chính sách dữ liệu — Day 29. Retention và xoá dữ liệu người dùng.

Chính sách đầy đủ (dữ liệu nào được lưu, ở đâu, bao lâu) nằm ở `docs/PRIVACY.md`.
Module này chỉ là phần **thực thi** của nó trên server.
"""

from __future__ import annotations

import os

from app.observability import audit
from app.sqlite_repository import SQLiteRepository

#: Biến môi trường đặt số ngày giữ lịch sử quét. `0`/thiếu = **không** tự xoá.
RETENTION_ENV = "MEDASSIST_HISTORY_RETENTION_DAYS"


def retention_days_from_env() -> int:
    """Số ngày giữ lịch sử; `0` nếu thiếu, rỗng, âm hoặc không phải số.

    ⚠️ Giá trị rác trả `0` (tắt) chứ không ném lỗi: cấu hình sai retention không được
    làm server không khởi động, và cũng không được hiểu thành "xoá ngay".
    """
    raw = os.environ.get(RETENTION_ENV, "").strip()
    try:
        days = int(raw)
    except ValueError:
        return 0
    return days if days > 0 else 0


def apply_retention(repo: SQLiteRepository) -> int:
    """Xoá lịch sử quá hạn theo cấu hình; trả số bản ghi đã xoá."""
    days = retention_days_from_env()
    if days <= 0:
        return 0
    removed = repo.purge_history_older_than(days)
    if removed:
        audit("retention_purge", "ok")
    return removed
