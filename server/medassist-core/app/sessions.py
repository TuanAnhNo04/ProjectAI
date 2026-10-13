"""Lưu lượng hội thoại ngắn hạn — Day 22.

## Vì sao là interface + in-memory, không phải Redis thật?

`ROADMAP.md` nói "session ngắn hạn ở Redis". Máy dev **không có Docker** nên không
chạy được Redis, và `AGENTS.md` + D2 cấm cài `redis`. Nên ở đây:

- [SessionStore] là **Protocol** — đúng hình dạng API mà Redis sẽ dùng;
- [InMemorySessionStore] là bản chạy được ngay, đủ cho test và dev;
- khi có Redis: viết `RedisSessionStore(SessionStore)`, đổi một dòng ở
  `get_session_store()` — **orchestrator và route không đổi**.

Giống hệt cách `Repository` đã giữ từ Day 15.

## ⚠️ Mọi truy cập đều lọc theo `user_id` — chống IDOR

`session_id` do **client** gửi lên. Nếu không lọc theo `user_id` thì người dùng
đoán được `session_id` của người khác là đọc được lịch sử hội thoại — đúng lớp
lỗi mà `AGENTS.md` mục 3 và test `test_user_a_khong_thay_lich_su_cua_user_b` đã bắt
ở Day 15. Ở đây thì **lịch sử chat còn nhạy hơn lịch sử ảnh**: nó chứa các câu hỏi
sức khoẻ nguyên văn.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Protocol

from app.orchestrator import MAX_HISTORY_TURNS, ChatTurn

#: Session tự hết hạn sau số giây này. Trên máy dev không có Redis nên không có
#: TTL thật — ta tự dọn khi đọc, đủ để không phình vô hạn.
SESSION_TTL_SECONDS = 24 * 60 * 60


class SessionStore(Protocol):
    """Hợp đồng lưu lượng hội thoại."""

    def append(self, user_id: int, session_id: str, role: str, content: str) -> None:
        """Ghi một lượt của `user_id`. `user_id` lấy từ token, không từ client."""
        ...

    def history(self, user_id: int, session_id: str) -> list[ChatTurn]:
        """Trả lịch sử **chỉ của `user_id`**. Phiên của người khác -> `[]`."""
        ...

    def clear(self, user_id: int, session_id: str) -> bool:
        """Xoá phiên của `user_id`. Phiên không thuộc -> `False`."""
        ...


class InMemorySessionStore:
    """Bản chạy được ngay, không cần Redis.

    ⚠️ Dữ liệu **mất khi restart**. Chấp nhận được ở dev; khi lên máy thật phải
    thay bằng Redis (xem docstring module).
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        self._data: dict[tuple[int, str], list[ChatTurn]] = defaultdict(list)
        self._max_turns = max_turns

    def append(self, user_id: int, session_id: str, role: str, content: str) -> None:
        key = (user_id, session_id)
        self._data[key].append(
            ChatTurn(
                user_id=user_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=time.time(),
            )
        )
        # ⚠️ Cắt bằng `[-max:]` chứ không `[:-max]` — lỗi dấu âm ở đây sẽ **xoá sạch
        # đầu lịch sử** thay vì cắt phần cuối. Lịch sử mới quan trọng hơn.
        if len(self._data[key]) > self._max_turns:
            self._data[key] = self._data[key][-self._max_turns:]

    def history(self, user_id: int, session_id: str) -> list[ChatTurn]:
        """Chỉ trả về lượt của `user_id` — khoá là cặp `(user_id, session_id)`."""
        key = (user_id, session_id)
        turns = self._data.get(key, [])
        cutoff = time.time() - SESSION_TTL_SECONDS
        return [t for t in turns if t.created_at >= cutoff]

    def clear(self, user_id: int, session_id: str) -> bool:
        key = (user_id, session_id)
        if key not in self._data:
            return False
        del self._data[key]
        return True

    def count_sessions(self) -> int:
        """Chỉ dùng cho test/giám sát — **không** đưa ra API công khai."""
        return len(self._data)


_store = InMemorySessionStore()


def get_session_store() -> SessionStore:
    """Store dùng chung toàn tiến trình.

    ⚠️ Khi chuyển sang Redis, **chỗ duy nhất cần đổi là hàm này**.
    """
    return _store


def reset_session_store() -> None:
    """Xoá toàn bộ phiên — chỉ dùng cho test."""
    global _store
    _store = InMemorySessionStore()
