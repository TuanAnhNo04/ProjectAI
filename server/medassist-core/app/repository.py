"""Cổng kho dữ liệu — để sau này đổi SQLite -> Postgres mà **không sửa route**.

Nguyên tắc ports & adapters của app Flutter (xem `lib/domain/ports/`) áp dụng
y hệt ở đây: route chỉ biết interface `Repository`, không biết đang lưu ở đâu.

Lý do Day 15 dùng SQLite (stdlib) thay vì Postgres: máy dev **không có Docker**
nên không chạy được Postgres/pgvector. Interface giữ nguyên để khi có Postgres
chỉ cần viết thêm adapter, không sửa endpoint.
"""

from __future__ import annotations

from typing import Protocol

from app.schemas import (
    HistoryItemResponse,
    MedicineResponse,
    SearchResponse,
    UserResponse,
)


class Repository(Protocol):
    """Hợp đồng kho dữ liệu mà mọi route dùng."""

    # -- Người dùng --
    def create_user(self, username: str, password_hash: str, email: str) -> UserResponse:
        """Tạo tài khoản. Ném lỗi `conflict` nếu `username` đã tồn tại."""
        ...

    def get_user_by_username(self, username: str) -> UserResponse | None: ...

    # -- Thuốc --
    def search_medicines(self, query: str, limit: int, offset: int) -> SearchResponse: ...

    def list_medicines(self, limit: int, offset: int) -> SearchResponse:
        """Danh sách tất cả thuốc (trang `M0` của lịch sử)."""
        ...

    # -- Lịch sử (luôn lọc theo user_id — chống IDOR) --
    def add_history(
        self,
        user_id: int,
        filepath: str,
        medicines: list[MedicineResponse],
    ) -> HistoryItemResponse: ...

    def list_history(self, user_id: int, limit: int, offset: int) -> list[HistoryItemResponse]:
        """Chỉ trả bản ghi **của `user_id`**."""
        ...

    def delete_history(self, user_id: int, item_id: int) -> bool:
        """Xoá bản ghi thuộc `user_id`; trả `False` nếu không có hoặc không thuộc.

        ⚠️ Điều kiện `user_id` nằm **trong** truy vấn — nếu lọc ở tầng route thì
        vẫn còn đường đọc dữ liệu người khác (xem `AGENTS.md` mục 3).
        """
        ...
