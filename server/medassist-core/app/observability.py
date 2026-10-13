"""Quan sát hệ thống — Day 24. `trace_id`, đo đạc, và **giữ cho log luôn sạch**.

## Ba nguyên tắc không được phá (quyết định `D4`)

1. **Chỉ được ghi 4 trường**: `trace_id`, `latency_ms`, `error_code`, `route`.
2. **Tuyệt đối không ghi**: nội dung tin nhắn, tên thuốc, email, token, `Authorization`.
3. ⚠️ **Query string cũng là dữ liệu sức khoẻ.** `GET /v1/search?name=panadol` — chữ
   "panadol" trong URL chính là thông tin thuốc người dùng đang tra. Nên ta chỉ log
   **route template** (`/v1/search`), **không** log URL thật. Đây là chỗ dễ sót nhất
   vì log access của web server mặc định ghi cả URL.

## Vì sao chi phí token **không** nằm trong log

ROADMAP Day 24 yêu cầu "log chi phí token theo user". Log ở nơi nhiều người đọc và
tồn tại lâu; gắn `user_id` vào log là tạo một bảng liên kết vĩnh viễn giữa danh
tính và hành vi tìm thuốc — đúng loại dữ liệu `AGENTS.md` mục 1 cấm.

Nên ta đếm vào [UsageTracker] trong RAM và chỉ trả cho **chính người đó** qua
`/v1/agent/usage`. Log vẫn sạch. Nếu sau này cần tổng hợp toàn hệ thống thì xuất
ra hệ thống đo lường riêng, **không** viết vào log ứng dụng.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

#: Trường **được phép** xuất hiện trong log. Khớp `schemas.ALLOWED_LOG_FIELDS`.
SAFE_LOG_FIELDS = frozenset({"trace_id", "latency_ms", "error_code", "route"})

#: `trace_id` của request đang xử lý. `contextvars` để đúng khi chạy đồng thời —
#: biến module thường sẽ lẫn giữa các request.
current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "trace_id", default=None
)

#: Tham số động trong đường dẫn (`/v1/history/{item_id}`) -> gộp về `{id}`.
_TEMPLATE_RE = re.compile(r"\{[^}]+\}")

MAX_ROUTE_CHARS = 80

logger = logging.getLogger("medassist.access")


def new_trace_id() -> str:
    """Sinh `trace_id` mới.

    Dùng `uuid4().hex` — ngẫu nhiên thật, **không** phải thời gian hay đếm số: hai
    request phải không đoán được `trace_id` của nhau.
    """
    return uuid.uuid4().hex


def route_template(path: str) -> str:
    """Chuẩn hoá đường dẫn thành template, bỏ phần có thể chứa dữ liệu.

    `?name=panadol` bị cắt hoàn toàn; `/v1/history/12` -> `/v1/history/{id}`.
    """
    without_query = path.split("?", 1)[0]
    return _TEMPLATE_RE.sub("{id}", without_query)[:MAX_ROUTE_CHARS]


def access_log(
    *,
    trace_id: str,
    route: str,
    latency_ms: float,
    status_code: int,
    error_code: str | None = None,
) -> None:
    """Ghi **một** dòng log truy cập.

    ⚠️ Hàm này **không nhận** tham số tùy ý — chỉ những trường an toàn. Thiết kế
    chữ ký như vậy để việc gọi sai **không biên dịch được**, thay vì chỉ dễ quên.
    """
    safe_route = route_template(route)
    fields: dict[str, Any] = {
        "trace_id": trace_id,
        "route": safe_route,
        "latency_ms": round(latency_ms, 1),
        "error_code": error_code or f"http_{status_code}",
    }
    # Lọc phòng thủ: nếu sau này ai thêm khoá mới mà quên rà lại, dòng log này
    # vẫn chỉ in trường đã duyệt.
    logger.info(
        "access",
        extra={k: v for k, v in fields.items() if k in SAFE_LOG_FIELDS},
    )


@dataclass
class Usage:
    """Số liệu dùng của **một** người dùng."""

    requests: int = 0
    prompt_chars: int = 0
    output_chars: int = 0
    #: Với provider **mock**, chi phí tiền thật luôn bằng 0. Ta vẫn đo độ dài để
    #: khi đổi sang provider thật đã có sẵn cơ sở so sánh.
    cost_usd: float = 0.0
    by_route: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_chars": self.prompt_chars,
            "output_chars": self.output_chars,
            "cost_usd": round(self.cost_usd, 6),
            "by_route": dict(self.by_route),
        }


class UsageTracker:
    """Đếm số liệu sử dụng **theo `user_id`** — chỉ trả cho chính người đó.

    ⚠️ Dữ liệu trong RAM, **mất khi restart**. Và không nên ghi ra log (xem docstring
    module). Khi lên máy thật thì đưa vào bảng riêng kèm chính sách xoá.
    """

    def __init__(self) -> None:
        self._data: dict[int, Usage] = defaultdict(Usage)

    def record(
        self, user_id: int, route: str, prompt_chars: int, output_chars: int
    ) -> None:
        if user_id <= 0:
            return
        entry = self._data[user_id]
        entry.requests += 1
        entry.prompt_chars += max(0, prompt_chars)
        entry.output_chars += max(0, output_chars)
        key = route_template(route)
        entry.by_route[key] = entry.by_route.get(key, 0) + 1

    def get(self, user_id: int) -> dict[str, Any]:
        """Số liệu của `user_id`. Người khác không có đường gọi được hàm này."""
        return self._data.get(user_id, Usage()).as_dict()

    def total_users(self) -> int:
        """Chỉ dùng cho test/giám sát — **không** đưa ra API công khai."""
        return len(self._data)


_tracker = UsageTracker()


def get_usage_tracker() -> UsageTracker:
    return _tracker


def reset_usage_tracker() -> None:
    """Xoá số liệu — chỉ dùng cho test."""
    global _tracker
    _tracker = UsageTracker()


class Timer:
    """Đo thời gian xử lý một request.

    Dùng `time.perf_counter()` chứ không `time.time()`: `time.time()` **bị** đổi khi
    đồng hồ hệ thống chỉnh (NTP) — có thể ra số âm, mà số âm trong báo cáo độ trễ là
    vô nghĩa.
    """

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
