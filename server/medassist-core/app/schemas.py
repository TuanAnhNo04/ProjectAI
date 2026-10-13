"""Schema Pydantic — **contract** giữa server và app Flutter.

Đây là nơi chốt hợp đồng dữ liệu. Tên trường và kiểu phải khớp với model Dart
(`lib/data/models/`) để hai bên không phải sửa song song.

⚠️ Quy tắc bảo mật (xem `AGENTS.md`):
- `password` chỉ xuất hiện trong **request** (`RegisterRequest`/`LoginRequest`)
  và **không bao giờ** trong response. Xem `tests/test_security.py`.
- Không schema nào trả về `password_hash`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Format lỗi thống nhất (xem `app/errors.py`)."""

    code: str = Field(examples=["validation_error"])
    message: str
    detail: Any | None = None


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str
    version: str
    time: datetime


# --- Auth -------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6, max_length=256)
    email: str = Field(max_length=254)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserResponse(BaseModel):
    """**Không** bao giờ chứa `password` hay `password_hash`."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str | None = None
    created_at: datetime | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Số giây hết hạn access token")
    #: Refresh token **bắt buộc** — client dùng nó khi access hết hạn.
    refresh_token: str | None = Field(
        default=None,
        description="Dùng để gọi /v1/auth/refresh. Chỉ lưu hash ở server.",
    )
    refresh_expires_in: int | None = None


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=4096)


class AuthResponse(BaseModel):
    """Phản hồi chung cho `/auth/register` và `/auth/login`."""

    user: UserResponse
    token: TokenResponse


# --- Thuốc / tìm kiếm -------------------------------------------------------


class MedicineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    composition: str | None = None
    uses: str | None = None
    side_effects: str | None = None
    manufacturer: str | None = None
    excellent_review_percent: float | None = None
    average_review_percent: float | None = None
    poor_review_percent: float | None = None
    source: str | None = None
    is_authoritative: bool = False
    ingredients: list[str] = Field(default_factory=list)
    updated_at: datetime | None = None


class UsageResponse(BaseModel):
    """Số liệu sử dụng **của chính người đang đăng nhập**.

    ⚠️ Endpoint này chỉ trả số liệu của người gọi — không có đường để xem của
    người khác. `user_id` lấy từ token, **không** nhận từ query (nếu nhận thì
    thành IDOR).
    """

    requests: int
    prompt_chars: int
    output_chars: int
    cost_usd: float
    by_route: dict[str, int]


class SearchResponse(BaseModel):
    results: list[MedicineResponse] = Field(default_factory=list)
    query: str | None = None
    total: int = 0


# --- Log an toàn (quyết định D4) --------------------------------------------

#: Danh sách trường **được phép** ghi log. Mọi thứ ngoài danh sách đều bị chặn.
ALLOWED_LOG_FIELDS = frozenset({"trace_id", "latency_ms", "error_code", "route"})


def safe_log_fields(**fields: object) -> dict[str, object]:
    """Chỉ giữ lại trường được phép ghi log.

    ⚠️ Chỉ **lọc khoá**, không lọc giá trị — nên chỉ dùng cho giá trị đã biết là an
    toàn (mã lỗi, số, route). Tuyệt đối không truyền `question`, `text`,
    `medicine_name`, `email`, `token`, `password`.

    Vì sao có hàm này thay vì quy ước miệng: một lần gọi nhầm trong code mới sẽ
    im lặng tạo ra log chứa dữ liệu sức khoẻ — và loại lỗi đó rất khó phát hiện
    sau này vì log trông vẫn bình thường.
    """
    return {k: v for k, v in fields.items() if k in ALLOWED_LOG_FIELDS}


def assert_log_safe(**fields: object) -> None:
    """Ném `ValueError` nếu có trường không được phép ghi log.

    Dùng trong test để chặn sớm hơn `safe_log_fields` (vốn lọc im lặng).
    """
    leaked = set(fields) - ALLOWED_LOG_FIELDS
    if leaked:
        raise ValueError(
            f"Trường không được phép ghi log: {sorted(leaked)}. "
            f"Chỉ được ghi: {sorted(ALLOWED_LOG_FIELDS)}."
        )



# --- Phân tích ảnh ----------------------------------------------------------



class AnalyzeResponse(BaseModel):
    """Kết quả nhận diện thuốc từ ảnh.

    Khớp với `AnalysisResult` phía Dart: server Flask trả `medicine_info`, đây là
    tên chuẩn hoá cho API mới.
    """

    medicines: list[MedicineResponse] = Field(default_factory=list)
    text: str | None = None
    analyzed_at: datetime | None = None


# --- Lịch sử ----------------------------------------------------------------


class HistoryItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filepath: str
    search_results: list[MedicineResponse] = Field(default_factory=list)
    created_at: datetime | None = None


class HistoryCreateRequest(BaseModel):
    """Body khi app ghi một bản ghi lịch sử.

    Dùng **body** chứ không phải query string vì `medicines` là mảng object —
    đưa lên query sẽ không tự mã hoá được (và dễ bị giới hạn độ dài URL).
    """

    filepath: str = Field(min_length=1, max_length=500)
    medicines: list[MedicineResponse] = Field(default_factory=list)


class HistoryResponse(BaseModel):
    history: list[HistoryItemResponse] = Field(default_factory=list)
    total: int = 0


# --- LLM / Agent (Day 20–21) ------------------------------------------------


class LLMMessage(BaseModel):
    """Một message gửi tới provider.

    `role` giới hạn ở 3 giá trị để không mở đường cho vai trò lạ (ví dụ `tool`
    để mô phỏng kết quả công cụ — không cần thiết và mở rộng bề mặt tấn công).
    """

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResponse(BaseModel):
    """Kết quả provider trả về.

    `text=None` + `refused=True` là **hợp đồng từ chối**, không phải lỗi: nghĩa là
    không đủ căn cứ để trả lời. `refusal_reason` là mã ổn định cho client.
    """

    text: str | None = None
    provider: str = Field(examples=["mock"])
    model: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    citations: list[str] = Field(default_factory=list)


class CitationResponse(BaseModel):
    """Một trích dẫn trả về cho app.

    ⚠️ `is_authoritative=False` -> app **phải** hiện cảnh báo "dữ liệu tham khảo,
    không thay thế tư vấn của dược sĩ/bác sĩ".
    """

    id: str
    medicine_name: str
    field: str = Field(description="Tên trường nguồn: uses, side_effects, ...")
    excerpt: str
    source: str | None = None
    is_authoritative: bool = False
    #: Điểm truy hồi 0..1 — để app hiện mức độ liên quan.
    score: float = 0.0


class AgentAnswerResponse(BaseModel):
    """Câu trả lời của agent (Day 21).

    ⚠️ `answered=False` -> `text` là `None` và app **không** được hiện câu trả lời.
    """

    answered: bool
    text: str | None = None
    refusal_reason: str | None = Field(
        default=None,
        description="Mã từ chối: no_evidence | not_authoritative | out_of_scope",
    )
    citations: list[CitationResponse] = Field(default_factory=list)
    #: Luôn bắt buộc hiện, kể cả khi đã trả lời.
    disclaimer: str = Field(
        default=(
            "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn của "
            "dược sĩ hoặc bác sĩ."
        )
    )
    provider: str = Field(examples=["mock"])


class AgentAskRequest(BaseModel):
    """Câu hỏi của người dùng gửi tới `/v1/agent/ask`.

    ⚠️ `max_length=500` để một câu hỏi khổng lồ không làm phình prompt.
    """

    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class AgentChatRequest(BaseModel):
    """Một lượt chat (Day 22), trả về dạng **SSE**.

    ⚠️ `session_id` do client sinh. Server **không tin** nó là của người gọi —
    mọi lượt đọc/ghi đều lọc theo `user_id` lấy từ token (xem `app/sessions.py`).
    """

    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=8, max_length=64)