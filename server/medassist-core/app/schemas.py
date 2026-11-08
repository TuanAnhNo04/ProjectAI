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
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.provenance import safe_source_url


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


class DeleteAccountRequest(BaseModel):
    """Xác nhận xoá tài khoản bằng mật khẩu (hành động không thể hoàn tác)."""

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
    # Provenance (tuỳ chọn; NULL = chưa rõ, không bịa).
    source_url: str | None = None
    retrieved_at: str | None = None
    license: str | None = None
    reviewed_by: str | None = None

    @field_validator("source_url")
    @classmethod
    def _safe_url(cls, v: str | None) -> str | None:
        return safe_source_url(v) or None


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
    #: Provenance. `source_url` chỉ giữ lại nếu qua allowlist (xem app/provenance.py).
    source_url: str = ""
    retrieved_at: str = ""
    license: str = ""
    reviewed_by: str = ""

    @field_validator("source_url", mode="before")
    @classmethod
    def _safe_url(cls, v: str | None) -> str:
        return safe_source_url(v)

    @field_validator("retrieved_at", "license", "reviewed_by", mode="before")
    @classmethod
    def _none_to_empty(cls, v: str | None) -> str:
        return v or ""


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


class NewsArticleResponse(BaseModel):
    """Một bài tin.

    ⚠️ Cố ý **không** có trường nội dung đầy đủ: ta trích dẫn và dẫn link về bài gốc, không
    đăng lại bài (xem phần điều khoản nguồn trong `app/news.py`).
    """

    id: str
    title: str
    summary: str = ""
    #: Link bài gốc; rỗng với bài mẫu kèm theo app.
    url: str = ""
    source: str
    published_at: str | None = None
    image_url: str | None = None


class NewsResponse(BaseModel):
    articles: list[NewsArticleResponse] = Field(default_factory=list)
    total: int = 0
    #: True = đây là bài mẫu, chưa bật nguồn tin thật. App PHẢI hiện rõ điều này.
    is_sample: bool = False
    #: Nguồn tải hỏng lần này — để app nói thật thay vì im lặng hiện thiếu tin.
    failed_sources: list[str] = Field(default_factory=list)
    #: Tên miền trong allowlist "cổng tin tức đã kiểm chứng" đang áp dụng.
    verified_sources: list[str] = Field(default_factory=list)
    disclaimer: str = ""


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
    #: Day 27 — client gửi danh sách thuốc đang dùng (từ hồ sơ sức khoẻ local)
    #: để agent tham chiếu khi trả lời. Dữ liệu này **không** lưu server.
    current_medications: list[str] = Field(default_factory=list, max_length=20)
    #: Day 32 — `true` CHỈ KHI người dùng đã đồng ý trong UI cho câu hỏi này được
    #: diễn đạt lại bởi LLM đám mây (đã khử định danh). `false`/vắng mặt -> server
    #: KHÔNG BAO GIỜ gọi nhà cung cấp đám mây cho lượt này, bất kể cấu hình server là
    #: gì — đây là lớp phòng vệ thứ hai, độc lập với cờ `MEDASSIST_LLM` phía server
    #: (xem `app/orchestrator.py::_select_provider_for_request`).
    cloud_ai_consent: bool = False
    #: Bệnh nền / dị ứng từ hồ sơ local (tuỳ chọn). Không lưu server; là DỮ LIỆU.
    health_context: HealthContext | None = None

#: Một mục văn bản ngắn do người dùng nhập (bệnh nền, dị ứng, tên thuốc).
ShortItem = Annotated[str, Field(min_length=1, max_length=100)]
#: Một chủ đề gần đây (Lifestyle) — dài hơn chút nhưng vẫn giới hạn.
TopicItem = Annotated[str, Field(min_length=1, max_length=200)]


class HealthContext(BaseModel):
    """Ngữ cảnh sức khoẻ từ hồ sơ LOCAL của app — server **không** lưu.

    ⚠️ Là DỮ LIỆU không tin cậy: chỉ dùng để so khớp chuỗi / đặt trong vùng dữ liệu
    của prompt, không bao giờ là chỉ dẫn.
    """

    model_config = ConfigDict(extra="forbid")

    conditions: list[ShortItem] = Field(default_factory=list, max_length=20)
    allergies: list[ShortItem] = Field(default_factory=list, max_length=20)
    #: Triệu chứng báo gần đây (từ bộ nhớ LOCAL). Chỉ để so khớp tái phát; không vào prompt.
    recent_symptoms: list[ShortItem] = Field(default_factory=list, max_length=5)


# `AgentChatRequest.health_context` tham chiếu `HealthContext` khai báo ở trên.
AgentChatRequest.model_rebuild()


class LifestyleRequest(BaseModel):
    """`POST /v1/agent/lifestyle` — **stateless**: server không lưu, không log."""

    current_medications: list[ShortItem] = Field(default_factory=list, max_length=20)
    conditions: list[ShortItem] = Field(default_factory=list, max_length=20)
    recent_topics: list[TopicItem] = Field(default_factory=list, max_length=5)
    cloud_ai_consent: bool = False


class LifestyleCitation(BaseModel):
    id: str
    medicine_name: str
    source: str | None = None
    excerpt: str
    #: Nguồn chính thống đã kiểm chứng? Mặc định False (client cũ vẫn chạy).
    is_authoritative: bool = False
    source_url: str = ""
    retrieved_at: str = ""
    license: str = ""
    reviewed_by: str = ""

    @field_validator("source_url", mode="before")
    @classmethod
    def _safe_url(cls, v: str | None) -> str:
        return safe_source_url(v)

    @field_validator("retrieved_at", "license", "reviewed_by", mode="before")
    @classmethod
    def _none_to_empty(cls, v: str | None) -> str:
        return v or ""


class LifestyleTip(BaseModel):
    title: str
    body: str
    kind: Literal["medication", "general", "diet"]
    citations: list[LifestyleCitation] = Field(default_factory=list)


class LifestyleResponse(BaseModel):
    tips: list[LifestyleTip] = Field(default_factory=list, max_length=5)
    disclaimer: str
    generated_by: Literal["rules", "claude"] = "rules"
    #: Bệnh nền (chuỗi của chính người dùng) chưa có tài liệu ăn uống nạp sẵn.
    unmatched_conditions: list[str] = Field(default_factory=list, max_length=20)
