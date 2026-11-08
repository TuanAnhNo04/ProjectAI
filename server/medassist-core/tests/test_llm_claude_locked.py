"""Adapter Claude (`app/llm_claude.py`) — mọi lần gọi HTTP đều qua `httpx.MockTransport`.

Không có kết nối mạng thật nào trong file này; không có khoá API thật nào được dùng.
"""

from __future__ import annotations

import httpx
import pytest

from app.llm import get_llm_provider
from app.llm_claude import API_URL, ClaudeProvider, build_claude_provider
from app.schemas import LLMMessage


_RealClient = httpx.Client


def _client(handler):
    return _RealClient(transport=httpx.MockTransport(handler))


def _text_response(text: str, stop_reason: str = "end_turn"):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == API_URL
        assert request.headers["x-api-key"] == "fake-test-key-not-real"
        assert "temperature" not in request.content.decode()  # bị loại khỏi API
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
            },
        )

    return handler


MSGS = [
    LLMMessage(role="system", content="He thong"),
    LLMMessage(role="user", content="Cau hoi"),
]


# ------------------------------------------------------------- build_claude_provider


def test_thieu_khoa_nem_runtime_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_claude_provider()


def test_co_khoa_dung_duoc(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key-not-real")
    p = build_claude_provider()
    assert p.name == "claude"
    assert p.model  # có model mặc định


def test_model_doi_duoc_qua_bien_moi_truong(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key-not-real")
    monkeypatch.setenv("MEDASSIST_CLAUDE_MODEL", "mo-hinh-gia-de-test")
    assert build_claude_provider().model == "mo-hinh-gia-de-test"


def test_khoi_tao_truc_tiep_voi_khoa_rong_bi_tu_choi():
    with pytest.raises(ValueError):
        ClaudeProvider(api_key="   ", model="x")


# --------------------------------------------------------------------- generate


def test_generate_goi_dung_url_khong_gui_temperature(monkeypatch):
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="model-gia")
    monkeypatch.setattr(
        httpx, "Client", lambda *a, **k: _client(_text_response("- Tra loi gia [R1]"))
    )
    res = p.generate(MSGS, max_tokens=99, temperature=0.9)
    assert res.refused is False
    assert res.text == "- Tra loi gia [R1]"
    assert res.provider == "claude"
    assert res.model == "model-gia"


def test_stop_reason_refusal_duoc_coi_la_tu_choi(monkeypatch):
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="x")

    def handler(request):
        return httpx.Response(200, json={"content": [], "stop_reason": "refusal"})

    import httpx as httpx_mod
    monkeypatch.setattr(httpx_mod, "Client", lambda *a, **k: _client(handler))
    res = p.generate(MSGS)
    assert res.refused is True
    assert res.refusal_reason == "provider_refused"


def test_loi_mang_khong_nem_chi_tu_choi(monkeypatch):
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="x")

    def handler(request):
        raise httpx.ConnectError("boom")

    import httpx as httpx_mod
    monkeypatch.setattr(httpx_mod, "Client", lambda *a, **k: _client(handler))
    res = p.generate(MSGS)
    assert res.refused is True
    assert res.refusal_reason == "provider_unavailable"


def test_content_khong_phai_mang_la_tu_choi(monkeypatch):
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="x")

    def handler(request):
        return httpx.Response(200, json={"content": "khong phai mang", "stop_reason": "end_turn"})

    import httpx as httpx_mod
    monkeypatch.setattr(httpx_mod, "Client", lambda *a, **k: _client(handler))
    res = p.generate(MSGS)
    assert res.refused is True
    assert res.refusal_reason == "bad_response"


def test_chi_loc_block_text_bo_qua_thinking(monkeypatch):
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="x")

    def handler(request):
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "khong lay cai nay"},
                    {"type": "text", "text": "cau tra loi that"},
                ],
                "stop_reason": "end_turn",
            },
        )

    import httpx as httpx_mod
    monkeypatch.setattr(httpx_mod, "Client", lambda *a, **k: _client(handler))
    res = p.generate(MSGS)
    assert res.text == "cau tra loi that"


def test_khong_bat_dau_bang_user_bi_tu_choi():
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="x")
    res = p.generate([LLMMessage(role="system", content="x")])
    assert res.refused is True
    assert res.refusal_reason == "bad_request"


# ---------------------------------------------------------- get_llm_provider gating


def test_mac_dinh_van_la_mock(monkeypatch):
    monkeypatch.delenv("MEDASSIST_LLM", raising=False)
    assert get_llm_provider().name == "mock"


def test_bat_claude_thieu_khoa_roi_ve_mock(monkeypatch):
    monkeypatch.setenv("MEDASSIST_LLM", "claude")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert get_llm_provider().name == "mock"


def test_bat_claude_co_khoa_dung_duoc(monkeypatch):
    monkeypatch.setenv("MEDASSIST_LLM", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key-not-real")
    assert get_llm_provider().name == "claude"


def test_max_tokens_bi_cat_la_tu_choi_va_co_san_1024(monkeypatch):
    p = ClaudeProvider(api_key="fake-test-key-not-real", model="x")
    seen = {}

    def handler(request):
        import json as _j

        seen["max_tokens"] = _j.loads(request.content)["max_tokens"]
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "cut"}], "stop_reason": "max_tokens"}
        )

    import httpx as httpx_mod
    monkeypatch.setattr(httpx_mod, "Client", lambda *a, **k: _client(handler))
    res = p.generate(MSGS, max_tokens=99)
    assert seen["max_tokens"] == 1024
    assert res.refused and res.refusal_reason == "truncated_max_tokens"
