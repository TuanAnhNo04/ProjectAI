"""Test adapter LLM cục bộ (Ollama) — **không có kết nối mạng nào**: mọi lời gọi đi qua
`httpx.MockTransport`, và dữ liệu hoàn toàn giả."""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent import REFUSAL_UNSUPPORTED, answer_question
from app.llm import get_llm_provider
from app.llm_ollama import (
    MODEL_ENV,
    URL_ENV,
    OllamaProvider,
    _cached_provider,
    validate_loopback_url,
)
from app.orchestrator import _llm_answer
from app.schemas import CitationResponse, LLMMessage
from test_agent import _meds


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _reply(text: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": text}})

    return handler


MSGS = [
    LLMMessage(role="system", content="He thong"),
    LLMMessage(role="user", content="Cau hoi"),
]


# ------------------------------------------------------------ địa chỉ loopback


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434/", "http://127.0.0.1"],
)
def test_dia_chi_loopback_duoc_chap_nhan(url):
    assert validate_loopback_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://10.0.0.5:11434",
        "http://192.168.1.10:11434",
        "http://127.0.0.1.evil.com:11434",
        "http://user:pw@127.0.0.1:11434",
        "http://127.0.0.1:11434/v1/secret",
        "https://127.0.0.1:11434",
        "ftp://127.0.0.1",
        "",
    ],
)
def test_dia_chi_ngoai_may_bi_tu_choi(url):
    with pytest.raises(ValueError):
        validate_loopback_url(url)


def test_khong_the_khoi_tao_provider_tro_ra_ngoai():
    with pytest.raises(ValueError):
        OllamaProvider(base_url="https://api.example.com")


def test_client_mac_dinh_khong_dung_proxy_khong_theo_redirect():
    provider = OllamaProvider()
    assert provider._client.follow_redirects is False
    assert provider._client.trust_env is False


# ------------------------------------------------------------------ generate


def test_generate_goi_dung_url_va_tham_so():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "- Dau dau [R1]"}})

    provider = OllamaProvider(model="test-model", client=_client(handler))
    result = provider.generate(MSGS, max_tokens=77, temperature=0.0)

    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert seen["body"]["model"] == "test-model"
    assert seen["body"]["stream"] is False
    assert seen["body"]["keep_alive"] == "30m"
    assert seen["body"]["options"] == {"temperature": 0.0, "num_predict": 77}
    assert "ĐỊNH DẠNG ĐẦU RA" in seen["body"]["messages"][0]["content"]
    assert seen["body"]["messages"][1]["content"] == "Cau hoi"
    assert result.provider == "ollama"
    assert result.model == "test-model"
    assert result.refused is False
    assert result.text == "- Dau dau [R1]"


def test_generate_khong_sua_tin_nhan_goc():
    provider = OllamaProvider(client=_client(_reply("- x [R1]")))
    provider.generate(MSGS)
    assert MSGS[0].content == "He thong"


def test_ollama_chua_chay_thi_tu_choi_chu_khong_nem_loi():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    result = OllamaProvider(client=_client(handler)).generate(MSGS)
    assert result.refused is True
    assert result.text is None
    assert result.refusal_reason == "provider_unavailable"


def test_loi_5xx_va_json_la_deu_la_tu_choi():
    for handler in (
        lambda r: httpx.Response(500, text="boom"),
        lambda r: httpx.Response(200, text="khong phai json"),
        lambda r: httpx.Response(200, json={"khong": "co message"}),
        lambda r: httpx.Response(200, json={"message": {"content": 123}}),
    ):
        result = OllamaProvider(client=_client(handler)).generate(MSGS)
        assert result.refused is True, handler


def test_noi_dung_rong_la_tu_choi():
    result = OllamaProvider(client=_client(_reply("   "))).generate(MSGS)
    assert result.refused is True
    assert result.refusal_reason == "no_evidence"


def test_redirect_khong_duoc_theo():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://evil.example.com/x"})

    provider = OllamaProvider()
    provider._client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False
    )
    assert provider.generate(MSGS).refused is True


# ------------------------------------------------------------ chọn provider


def test_mac_dinh_van_la_mock(monkeypatch):
    monkeypatch.delenv("MEDASSIST_LLM", raising=False)
    assert get_llm_provider().name == "mock"


def test_bat_co_ollama(monkeypatch):
    _cached_provider.cache_clear()
    monkeypatch.setenv("MEDASSIST_LLM", "ollama")
    monkeypatch.delenv(URL_ENV, raising=False)
    monkeypatch.setenv(MODEL_ENV, "ten-model-gia")
    provider = get_llm_provider()
    assert provider.name == "ollama"
    assert provider.model == "ten-model-gia"
    _cached_provider.cache_clear()


def test_url_ngoai_may_thi_quay_ve_mock(monkeypatch):
    _cached_provider.cache_clear()
    monkeypatch.setenv("MEDASSIST_LLM", "ollama")
    monkeypatch.setenv(URL_ENV, "https://api.example.com")
    assert get_llm_provider().name == "mock"
    _cached_provider.cache_clear()


# ----------------------------------------- agent: lớp kiểm chứng vẫn canh LLM


class _Fake:
    name = "ollama"

    def __init__(self, text):
        self._text = text

    def generate(self, messages, *, max_tokens=512, temperature=0.0):
        from app.schemas import LLMResponse

        return LLMResponse(text=self._text, provider="ollama", model="fake")


def test_llm_bia_noi_dung_thi_agent_tu_choi():
    r = answer_question(
        "Panadol 500mg dùng để làm gì",
        _meds(),
        provider=_Fake("- Chữa khỏi ung thư hoàn toàn [R1]"),
    )
    assert r.answered is False
    assert r.refusal_reason == REFUSAL_UNSUPPORTED


def test_llm_ma_trich_dan_gia_thi_agent_tu_choi():
    r = answer_question(
        "Panadol 500mg dùng để làm gì",
        _meds(),
        provider=_Fake("- Pain relief, fever reduction [R99]"),
    )
    assert r.answered is False


# ----------------------------------- orchestrator: _llm_answer có phương án lùi


def _cite() -> list[CitationResponse]:
    return [
        CitationResponse(
            id="R1",
            medicine_name="Panadol 500mg",
            field="uses",
            excerpt="Pain relief, fever reduction",
            source="dev-sample",
            is_authoritative=False,
            score=0.5,
        )
    ]


_OLD = "Kết quả:\n- Panadol 500mg (uses): Pain relief, fever reduction [R1]"


def test_llm_that_bam_nguon_thi_duoc_dung():
    out = _llm_answer("q", _OLD, _cite(), _Fake("- Pain relief, fever reduction [R1]"))
    assert out.startswith("Kết quả:")
    assert out.endswith("- Pain relief, fever reduction [R1]")
    assert "Panadol 500mg (uses)" not in out


def test_llm_that_bia_thi_giu_cau_tra_loi_cu():
    assert _llm_answer("q", _OLD, _cite(), _Fake("- Chữa khỏi mọi bệnh [R1]")) == _OLD


def test_llm_that_thieu_ma_trich_dan_thi_giu_cau_tra_loi_cu():
    assert _llm_answer("q", _OLD, _cite(), _Fake("- Pain relief, fever reduction")) == _OLD


def test_llm_loi_thi_giu_cau_tra_loi_cu():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("x")

    provider = OllamaProvider(client=_client(handler))
    assert _llm_answer("q", _OLD, _cite(), provider) == _OLD


def test_mock_thi_khong_doi_gi():
    class _Mock(_Fake):
        name = "mock"

    assert _llm_answer("q", _OLD, _cite(), _Mock("- bat ky [R1]")) == _OLD


def test_khong_co_trich_dan_thi_khong_goi_llm():
    class _Boom(_Fake):
        def generate(self, *a, **k):
            raise AssertionError("không được gọi LLM khi không có nguồn")

    assert _llm_answer("q", "Đã đặt nhắc.", [], _Boom("x")) == "Đã đặt nhắc."
