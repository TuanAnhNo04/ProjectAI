"""Dependency dùng chung cho các route (get repo, xác thực user).

Tách riêng để route không lặp lại logic và để test thay kho chỉ cần override
`get_repository`.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, Header

from app.errors import CODE_UNAUTHORIZED, APIError
from app.jwt_tokens import decode
from app.sqlite_repository import SQLiteRepository

#: Kho dùng chung cho tiến trình. Route **không** tự tạo kho.
_repository: SQLiteRepository | None = None


def get_repository() -> SQLiteRepository:
    """Kho hiện tại.

    Đường dẫn lấy từ biến môi trường `MEDASSIST_DB`; mặc định file trong thư mục
    `data/`. Test override hàm này bằng kho in-memory.
    """
    global _repository
    if _repository is None:
        db_path = os.environ.get("MEDASSIST_DB")
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _repository = SQLiteRepository(db_path or ":memory:")
    return _repository


def reset_repository() -> None:
    """Đóng và bỏ kho — dùng khi test hoặc khởi động lại."""
    global _repository
    if _repository is not None:
        _repository.close()
    _repository = None


def current_user_id(authorization: str | None = Header(default=None)) -> int:
    """Lấy `user_id` từ header `Authorization: Bearer <token>`.

    ⚠️ Bắt buộc mọi endpoint đọc/ghi **dữ liệu người dùng** phải dùng hàm này
    (xem `AGENTS.md` mục 3 về phân quyền theo `user_id`).
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Thiếu token xác thực.",
        )
    token = authorization.split(" ", 1)[1].strip()
    payload = decode(token)

    # Token hợp lệ về chữ ký nhưng **hết hạn** -> trả thông báo riêng để client
    # biết cần gọi `/v1/auth/refresh` thay vì bắt người dùng đăng nhập lại.
    if payload is None:
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Token không hợp lệ hoặc đã hết hạn.",
        )
    # Access token **không** được dùng ở endpoint refresh (và ngược lại).
    if payload.get("type") != "access":
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Loại token không hợp lệ cho endpoint này.",
        )

    sub = payload.get("sub")
    if not isinstance(sub, int):
        raise APIError(
            status_code=401,
            code=CODE_UNAUTHORIZED,
            message="Token không hợp lệ hoặc đã hết hạn.",
        )
    return sub