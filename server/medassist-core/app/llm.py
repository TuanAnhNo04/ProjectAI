"""Cổng gọi LLM — **Day 20: chỉ có provider giả (mock), không có provider thật**.

## Vì sao có interface mà KHÔNG có adapter thật?

`AGENTS.md` mục 2 cấm gửi dữ liệu người dùng ra ngoài máy. Adapter LLM thật tức là
gửi **câu hỏi y tế + trích dẫn thuốc** tới nhà cung cấp. Điều đó chỉ được phép khi
**cả ba** điều kiện sau đã có:

1. người dùng **đồng ý rõ ràng** trong UI,
2. đã **khử định danh** dữ liệu trước khi gửi,
3. đã ghi vào `docs/PRIVACY.md`.

Cả ba đều chưa có. Vì vậy hôm nay:

- ✅ `LLMProvider` (interface) để route không phụ thuộc hãng cụ thể.
- ✅ `MockLLMProvider` — **tất định, offline, không đọc biến môi trường, không mở
  socket**.
- ❌ **Không** có `OpenAIProvider`/`GeminiProvider`, **không** có API key.
- ❌ **Không** mã nào trong repo gọi `requests`/`httpx`/`urllib` ra internet.

`get_llm_provider()` mặc định trả mock. Khi nào có adapter thật thì đổi hàm này,
**không sửa route**.

⚠️ `MockLLMProvider` **không** bịa nội dung y khoa: nó chỉ ghép lại câu trả lời từ
những đoạn văn bản truy hồi được truyền vào, kèm mã trích dẫn. Không có đoạn nào
thì trả `refused` — xem `app/agent.py`.
"""

from __future__ import annotations

from typing import Protocol

from app.schemas import LLMMessage, LLMResponse


class LLMProvider(Protocol):
    """Hợp đồng tối thiểu mà `app/agent.py` cần.

    ⚠️ Cố tình **rất hẹp**: chỉ `generate()`. Không `embed()`, không `stream()` —
    thêm sau khi thật sự cần, và mỗi lần thêm phải chống được việc gửi dữ liệu
    người dùng đi nơi khác.
    """

    #: Tên định danh để log/ghi vào response (KHÔNG ghi nội dung sinh ra).
    name: str

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Sinh câu trả lời. `temperature` mặc định 0 để lặp lại cho cùng đầu vào."""
        ...


class MockLLMProvider:
    """Provider giả: **tất định, offline, không side-effect**.

    ⚠️ Đây KHÔNG phải "LLM đơn giản hoá cho nhanh". Đây là chốt chặn an toàn: một
    LLM thật sẽ tự bịa câu trả lời y khoa khi không có căn cứ — đúng thứ mà
    `AGENTS.md` cấm. Provider giả **từ chối sinh** nếu không có đoạn trích dẫn.
    """

    name = "mock"

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Ghép câu trả lời từ ngữ cảnh đã đưa vào.

        `max_tokens`/`temperature` được **chấp nhận nhưng không dùng** — giữ đúng
        chữ ký để sau này thay adapter thật mà không sửa call site.
        """
        del max_tokens, temperature  # cố ý bỏ qua: provider giả không ngẫu nhiên

        context = self._extract_context(messages)
        if not context:
            # Không có căn cứ -> KHÔNG trả lời. Đây là hành vi an toàn, không
            # phải giới hạn kỹ thuật.
            return LLMResponse(
                text=None,
                provider=self.name,
                model="mock-extractive",
                refused=True,
                refusal_reason="no_evidence",
            )

        lines = [f"- {text} [{cite}]" for cite, text in context]
        return LLMResponse(
            text="\n".join(lines),
            provider=self.name,
            model="mock-extractive",
            refused=False,
            citations=[cite for cite, _ in context],
        )

    def _extract_context(
        self, messages: list[LLMMessage]
    ) -> list[tuple[str, str]]:
        """Bóc `(mã trích dẫn, nội dung)` ra khỏi ngữ cảnh trong prompt.

        ⚠️ Nội dung trong ngữ cảnh là **dữ liệu không tin cậy** (CSV do người khác
        soạn). Dấu `<<<` bị lọc khỏi cả mã lẫn nội dung để một đoạn dữ liệu độc
        hại **không thể tự khai báo thêm "nguồn" giả** hay thoát ra ngoài vùng
        đánh dấu.
        """
        found: list[tuple[str, str]] = []
        for message in messages:
            raw = message.content
            while self._CTX_START in raw:
                raw = raw.split(self._CTX_START, 1)[1]
                if self._CTX_END not in raw:
                    # Ngữ cảnh bị cắt cụt -> không tin, coi như không có căn cứ.
                    return []
                block, raw = raw.split(self._CTX_END, 1)
                cite, _, body = block.partition("|")
                found.append((_sanitize_id(cite), _sanitize_text(body.strip())))
        return found

#: Đánh dấu ranh giới ngữ cảnh trong prompt. Văn bản nằm giữa hai dấu này là
    #: **DỮ LIỆU**, không phải lệnh — xem `app/prompt.py`.
    _CTX_START = "<<<NGUON_THAM_CHIEU>>>"
    _CTX_END = "<<<HET_NGUON>>>"


def _sanitize_id(value: str) -> str:
    """Giữ mã trích dẫn ở dạng an toàn: chữ/số/gạch, tối đa 32 ký tự."""
    cleaned = "".join(c for c in value.strip() if c.isalnum() or c in "_-.")
    return cleaned[:32] or "unknown"


def _sanitize_text(value: str) -> str:
    """Làm sạch văn bản nguồn trước khi đưa ra câu trả lời."""
    return value.replace("<<<", "").replace(">>>", "").strip()


#: Provider dùng chung cho toàn tiến trình. Provider giả **không giữ trạng thái**
#: nên một instance duy nhất là đủ và cho kết quả giống nhau mọi lần.
_default_provider: MockLLMProvider = MockLLMProvider()


def get_llm_provider() -> LLMProvider:
    """Provider hiện tại — **luôn là mock**.

    ⚠️ Không đọc biến môi trường, không đăng ký provider khác. Khi được phép gắp
    adapter thật (sau khi đủ 3 điều kiện ở `AGENTS.md` mục 2), chỗ duy nhất cần
    sửa là hàm này.
    """
    return _default_provider
