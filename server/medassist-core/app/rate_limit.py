"""Giới hạn tần suất đăng nhập — chống dò mật khẩu (brute force).

## Vì sao không dùng Redis?
ROADMAP Day 16 ghi "rate limit login qua Redis", nhưng máy dev **không có Docker**
nên không chạy được Redis; quy tắc bảo mật cũng cấm tự thêm package. Vì vậy bản
này dùng bộ đếm **trong RAM** của tiến trình.

## Giới hạn đã biết (phải nói rõ, không được giả vờ đã xong)
- Chỉ tính trong **một tiến trình** → khi chạy nhiều worker (uvicorn `--workers 4`)
  thì mỗi worker một bộ đếm, tổng hạn mền lên gấp số worker.
- Mất sạch khi restart.
- **Không** dùng được khi scale nhiều máy.

→ Khi có Docker: chuyển sang Redis (cùng interface `RateLimiter`), **không sửa route**.

> Ghi chú bảo mật: đây là lớp phòng thủ **thứ hai**. Lớp thứ nhất là không rò rỉ
> cho biết tài khoản nào tồn tại (xem `_GENERIC_LOGIN_ERROR` trong `routers/auth.py`).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

#: Số lần thất bại tối đa trong cửa sổ thời gian.
DEFAULT_MAX_ATTEMPTS = 5

#: Độ dài cửa sổ (giây).
DEFAULT_WINDOW_SECONDS = 60


class RateLimitExceeded(Exception):
    """Bị chặn vì thử quá nhiều lần."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(f"quá nhiều lần thử, thử lại sau {retry_after_seconds}s")
        self.retry_after_seconds = retry_after_seconds


class InMemoryRateLimiter:
    """Bộ đếm cửa sổ trượt trong RAM (thread-safe)."""

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
    ) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Ghi nhận 1 lần thử; ném [RateLimitExceeded] nếu vượt hạn."""
        now = time.monotonic()
        with self._lock:
            bucket = self._attempts[key]
            # Xoá các lần thử đã ngoài cửa sổ.
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()

            if len(bucket) >= self.max_attempts:
                retry_after = int(self.window_seconds - (now - bucket[0])) + 1
                raise RateLimitExceeded(max(retry_after, 1))

            bucket.append(now)

    def reset(self, key: str) -> None:
        """Xoá bộ đếm sau khi đăng nhập thành công."""
        with self._lock:
            self._attempts.pop(key, None)

    def attempts_left(self, key: str) -> int:
        """Còn bao nhiêu lần thử (dùng cho header `X-RateLimit-Remaining`)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._attempts.get(key)
            if not bucket:
                return self.max_attempts
            recent = sum(1 for t in bucket if now - t <= self.window_seconds)
        return max(self.max_attempts - recent, 0)


#: Khoá rate limit = IP + tên đăng nhập. Gộp cả hai để chặn được cả
#: (a) một tài khoản bị dò từ nhiều IP và (b) một IP dò nhiều tài khoản.
def login_rate_key(username: str, client_host: str) -> str:
    return f"login:{client_host}:{username.strip().lower()}"


#: Bộ đếm dùng chung cho tiến trình.
_limiter = InMemoryRateLimiter()


def get_limiter() -> InMemoryRateLimiter:
    return _limiter


def reset_limiter() -> None:
    """Xoá toàn bộ bộ đếm — dùng cho test."""
    with _limiter._lock:  # noqa: SLF001  (reset toàn cục, chấp nhận)
        _limiter._attempts.clear()  # noqa: SLF001