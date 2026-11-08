"""Adapter LLM **chạy cục bộ** qua Ollama (`http://127.0.0.1:11434`).

## Vì sao adapter này được phép (và adapter đám mây thì chưa)

`AGENTS.md` mục 2 cấm gửi dữ liệu người dùng **ra ngoài máy** khi chưa đủ đồng ý trong
UI + khử định danh + ghi `docs/PRIVACY.md`. Ollama chạy ngay trên máy này và chỉ nhận
kết nối loopback, nên dữ liệu **không rời máy**. Để giữ đúng điều đó, file này:

- chỉ chấp nhận địa chỉ `127.0.0.1` / `localhost` / `::1` — địa chỉ khác bị từ chối
  ngay lúc khởi tạo (`validate_loopback_url`), không có cách "cấu hình" ra ngoài;
- `trust_env=False`: không đi qua `HTTP_PROXY`/`HTTPS_PROXY` của môi trường (proxy sẽ
  đưa lưu lượng loopback đi chỗ khác);
- `follow_redirects=False`: một máy chủ cục bộ bị chiếm không thể chuyển hướng ta ra ngoài;
- **không** ghi log nội dung prompt/câu trả lời, không ghi chi tiết lỗi (`AGENTS.md` mục 3).

## Không tin đầu ra của mô hình

Mô hình nhỏ có thể viết thứ không có trong nguồn. Adapter này **không** tự bảo đảm điều
đó — lớp bảo đảm là `verify_grounding()` trong `app/agent.py` (và trong orchestrator):
câu nào không bám nguồn đã trích dẫn thì bị bỏ, quay về câu trả lời trích nguyên văn.

Mọi lỗi (Ollama chưa chạy, hết thời gian, JSON lạ) được trả về dạng `refused=True` —
**không** ném lỗi ra route, và **không** làm app chết.

Bật bằng: `MEDASSIST_LLM=ollama` (mặc định vẫn là mock). Tuỳ chọn:
`MEDASSIST_OLLAMA_MODEL` (mặc định `qwen2.5:3b`), `MEDASSIST_OLLAMA_URL` (chỉ loopback).
"""

from __future__ import annotations

import functools
import os
from urllib.parse import urlparse

import httpx

from app.schemas import LLMMessage, LLMResponse

DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:3b"
URL_ENV = "MEDASSIST_OLLAMA_URL"
MODEL_ENV = "MEDASSIST_OLLAMA_MODEL"

#: Chỉ các tên máy này được coi là "trên máy này".
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

#: Mô hình nhỏ hay viết lời dẫn/kết luận thừa; `verify_grounding` sẽ loại cả câu trả lời
#: nếu một dòng không bám nguồn, nên nhắc rõ định dạng để tỉ lệ đạt cao hơn.
_FORMAT_HINT = (
    "\n\nĐỊNH DẠNG ĐẦU RA (bắt buộc): chép NGUYÊN VĂN cụm từ trong nguồn, KHÔNG dịch, "
    "KHÔNG diễn giải, KHÔNG nhắc lại câu hỏi. Mỗi ý một dòng, dạng "
    "'- <chép nguyên văn từ nguồn> [mã]'. "
    "Ví dụ: nguồn 'R1|Pain relief, fever reduction' -> trả lời đúng một dòng: "
    "- Pain relief, fever reduction [R1]. "
    "Không thêm lời dẫn, lời khuyên hay kết luận nào khác."
)


def validate_loopback_url(url: str) -> str:
    """Trả URL chuẩn hoá nếu trỏ về máy này, ngược lại ném `ValueError`."""
    parsed = urlparse(url.strip())
    if parsed.scheme != "http":
        raise ValueError("Ollama chỉ được gọi qua http trên máy này.")
    if (parsed.hostname or "") not in _LOOPBACK_HOSTS:
        raise ValueError("Ollama phải chạy trên máy này (127.0.0.1/localhost).")
    if parsed.username or parsed.password or parsed.path not in ("", "/"):
        raise ValueError("Địa chỉ Ollama không được kèm tài khoản hay đường dẫn.")
    return url.strip().rstrip("/")


def _refused(model: str, reason: str) -> LLMResponse:
    return LLMResponse(
        text=None, provider="ollama", model=model, refused=True, refusal_reason=reason
    )


class OllamaProvider:
    """`LLMProvider` gọi Ollama cục bộ. Tất định ở mức có thể (`temperature=0`)."""

    name = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = validate_loopback_url(base_url)
        self.model = model
        # `client` chỉ để test thay transport; URL vẫn đã được kiểm tra ở trên.
        self._client = client or httpx.Client(
            timeout=timeout, trust_env=False, follow_redirects=False
        )

    def generate(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> LLMResponse:
        payload_messages = [{"role": m.role, "content": m.content} for m in messages]
        if payload_messages and payload_messages[0]["role"] == "system":
            payload_messages[0]["content"] += _FORMAT_HINT

        body = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            # Giữ model trong bộ nhớ: nạp lại model 3B trên CPU mất cả chục giây.
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = self._client.post(f"{self._base_url}/api/chat", json=body)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("nội dung phải là chuỗi")
            text = content.strip()
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            # Cố ý nuốt chi tiết: thông điệp lỗi có thể chứa nội dung request.
            return _refused(self.model, "provider_unavailable")

        if not text:
            return _refused(self.model, "no_evidence")
        return LLMResponse(text=text, provider=self.name, model=self.model, refused=False)


@functools.lru_cache(maxsize=4)
def _cached_provider(url: str, model: str) -> OllamaProvider:
    return OllamaProvider(base_url=url, model=model)


def build_ollama_provider() -> OllamaProvider:
    """Dựng provider từ biến môi trường. Ném `ValueError` nếu URL không phải loopback."""
    url = os.environ.get(URL_ENV, "").strip() or DEFAULT_URL
    model = os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
    validate_loopback_url(url)  # kiểm tra trước khi cache
    return _cached_provider(url, model)
