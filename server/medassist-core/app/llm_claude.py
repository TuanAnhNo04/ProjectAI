"""Adapter LLM đám mây (Claude, Anthropic Messages API) — Day 32.

## Vẫn phải đủ 3 điều kiện của `AGENTS.md` mục 2 ở TẦNG GỌI, không phải tầng này

File này cài đặt đúng giao thức Claude, nhưng **không tự quyết định có được phép gọi
không** — ba lớp phòng vệ nằm ở nơi khác, và thiếu một lớp là không gọi được gì cả:

1. **Đồng ý UI (có lưu thời điểm).** App chỉ gửi `cloud_ai_consent=true` sau khi người
   dùng bật công tắc ở màn Quyền riêng tư; `ConsentStore` lưu kèm mốc thời gian + phiên
   bản chính sách (`lib/data/local/consent_store.dart::CloudAiConsentStore`).
2. **Khử định danh.** `app/orchestrator.py` gọi `app/deidentify.py::deidentify()` trên
   câu hỏi **trước khi** đưa vào đây — file này không tự khử định danh, vì làm hai lần ở
   hai nơi dễ khiến người đọc tưởng chỗ kia đã lo rồi.
3. **`docs/PRIVACY.md` đã ghi** tên bên nhận (Anthropic), dữ liệu gửi, chính sách lưu trữ.

Ở tầng này, khoá cứng còn lại: **không có `ANTHROPIC_API_KEY` thì không dựng được
provider** — `build_claude_provider()` ném `RuntimeError`, `get_llm_provider()` bắt lỗi
đó và rơi về `mock`. Không có khoá nào được đọc từ nơi khác (không nhận qua tham số, body
HTTP, hay file) — chỉ biến môi trường của tiến trình server.

## Tư liệu kỹ thuật (xác minh 28/10/2026, xem `docs/PRIVACY.md` §6.1 để biết nguồn)

- Endpoint: `POST https://api.anthropic.com/v1/messages`
- Header bắt buộc: `x-api-key`, `anthropic-version`, `Content-Type: application/json`
- `temperature`/`top_p`/`top_k` đã bị loại khỏi API — **không gửi**, gửi lên sẽ nhận 400.
- `content` trong response là **mảng block**; phải lọc `type == "text"` rồi nối lại.
- Không có SDK Dart chính thức — vì vậy app **không bao giờ** gọi Claude trực tiếp, chỉ
  server (khoá nằm ở máy chủ, APK giải nén được là lộ khoá nếu nhúng trong app).
"""

from __future__ import annotations

import logging
import os

from app.schemas import LLMMessage, LLMResponse

API_URL = "https://api.anthropic.com/v1/messages"
#: Khoá đọc DUY NHẤT từ biến môi trường của tiến trình server — không có đường nào khác.
API_KEY_ENV = "ANTHROPIC_API_KEY"
#: Model có thể đổi qua biến môi trường vì ID model thay đổi theo thời gian; đừng ghim
#: cứng một chuỗi rồi quên là nó có thể lỗi thời.
MODEL_ENV = "MEDASSIST_CLAUDE_MODEL"
DEFAULT_MODEL = "claude-sonnet-5-5"
ANTHROPIC_VERSION = "2026-10-21"
#: "effort" theo yêu cầu chủ dự án (Sonnet, effort medium) — xem `output_config` trong
#: tài liệu Messages API; không phải `temperature`.
DEFAULT_EFFORT = "medium"

_log = logging.getLogger("medassist")


def _refused(model: str, reason: str) -> LLMResponse:
    return LLMResponse(
        text=None, provider="claude", model=model, refused=True, refusal_reason=reason
    )


class ClaudeProvider:
    """`LLMProvider` gọi Claude thật qua HTTP. Mọi lỗi -> `refused=True`, không ném."""

    name = "claude"

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        if not api_key.strip():
            raise ValueError("Thiếu khoá API.")
        self._api_key = api_key
        self.model = model
        self._timeout = timeout

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import httpx

        # ⚠️ `temperature` CỐ Ý không gửi: tham số này đã bị loại khỏi Messages API
        # (gửi lên nhận 400). Giữ tham số trong chữ ký để khớp `LLMProvider` Protocol.
        del temperature

        # Messages API: `system` tách riêng khỏi `messages`; `messages` phải bắt đầu
        # bằng role `user` và không có role `system` lẫn vào.
        system_parts = [m.content for m in messages if m.role == "system"]
        turn_messages = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        if not turn_messages or turn_messages[0]["role"] != "user":
            return _refused(self.model, "bad_request")

        # Thinking thích ứng mặc định bật trên Sonnet 5.5 và TÍNH VÀO max_tokens: với
        # 512 token, phần suy nghĩ + tiếng Việt (nhiều token) dễ cắt cụt trước khi có mã
        # trích dẫn -> bị grounding loại -> rơi về câu trích nguyên văn. Sàn 1024.
        body: dict[str, object] = {
            "model": self.model,
            "max_tokens": max(max_tokens, 1024),
            "messages": turn_messages,
            "output_config": {"effort": DEFAULT_EFFORT},
            "stream": False,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self._timeout, trust_env=False) as client:
                response = client.post(API_URL, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            # Chỉ mã HTTP (401 sai khoá, 400 sai tham số, 429 hết hạn mức...): không có
            # nội dung nào của người dùng hay của phản hồi.
            return _refused(self.model, f"http_{exc.response.status_code}")
        except (httpx.HTTPError, ValueError):
            # ⚠️ Cố ý nuốt chi tiết lỗi: message lỗi của Anthropic có thể lặp lại
            # một phần nội dung gửi lên (câu hỏi đã khử định danh), không log nó.
            return _refused(self.model, "provider_unavailable")

        if data.get("stop_reason") == "refusal":
            return _refused(self.model, "provider_refused")

        if data.get("stop_reason") == "max_tokens":
            # Cắt cụt giữa câu -> có thể thiếu mã trích dẫn; không dùng nửa câu trả lời.
            return _refused(self.model, "truncated_max_tokens")

        blocks = data.get("content")
        if not isinstance(blocks, list):
            return _refused(self.model, "bad_response")
        text = "".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        if not text:
            return _refused(self.model, "no_evidence")
        return LLMResponse(text=text, provider=self.name, model=self.model, refused=False)


def build_claude_provider() -> ClaudeProvider:
    """Dựng provider từ biến môi trường. Ném `RuntimeError`/`ValueError` nếu thiếu khoá.

    `get_llm_provider()` trong `app/llm.py` bắt các lỗi này và rơi về `mock` — gọi hàm
    này trực tiếp mà không bắt lỗi sẽ làm sập tiến trình, đúng ý: thiếu khoá không bao
    giờ được âm thầm "coi như xong".
    """
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Thiếu biến môi trường {API_KEY_ENV} — LLM đám mây không được bật."
        )
    model = os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
    return ClaudeProvider(api_key=api_key, model=model)
