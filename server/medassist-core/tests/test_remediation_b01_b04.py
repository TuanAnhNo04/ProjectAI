"""Hồi quy cho B-01..B-04, C-01, C-02 (RELEASE_REPORT). Toàn bộ dữ liệu GIẢ, không mạng."""

from __future__ import annotations

import httpx
import pytest

from app.deidentify import deidentify
from app.llm import get_llm_provider
from app.orchestrator import _select_provider_for_request, chat
from app.safety import detect_red_flags
from app.sqlite_repository import SQLiteRepository

_RealClient = httpx.Client


@pytest.fixture()
def claude_env(monkeypatch):
    monkeypatch.setenv("MEDASSIST_LLM", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key-not-real")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}
        )

    monkeypatch.setattr(
        httpx, "Client", lambda *a, **k: _RealClient(transport=httpx.MockTransport(handler))
    )
    return calls


def _repo():
    from app.ingest import load_csv, to_rows

    r = SQLiteRepository(":memory:")
    r.add_medicines(to_rows(load_csv()))
    return r


def test_b01_khong_dong_y_thi_mock(claude_env):
    configured = get_llm_provider()
    assert configured.name == "claude"
    assert _select_provider_for_request(configured, False).name == "mock"
    assert _select_provider_for_request(configured, True).name == "claude"


def test_b01_chat_khong_dong_y_khong_goi_claude(claude_env):
    res = chat("Panadol 500mg dùng để làm gì?", user_id=1, repo=_repo(),
               session_id="session-abc-123", cloud_ai_consent=False)
    assert res.answered
    assert claude_env["n"] == 0


def test_b01_chat_co_dong_y_moi_goi_claude(claude_env):
    chat("Panadol 500mg dùng để làm gì?", user_id=1, repo=_repo(),
         session_id="session-abc-123", cloud_ai_consent=True)
    assert claude_env["n"] >= 1


def test_b01_route_chat_va_ask_khong_ra_claude(claude_env, seeded_client):
    from conftest import register_user

    register_user(seeded_client, "b01user", "matkhau123", "b01@e.com")
    tok = seeded_client.post(
        "/v1/auth/login", json={"username": "b01user", "password": "matkhau123"}
    ).json()["token"]["access_token"]
    h = {"Authorization": f"Bearer {tok}"}
    r = seeded_client.post("/v1/agent/chat", headers=h, json={
        "message": "Panadol 500mg dùng để làm gì?", "session_id": "session-abc-123"})
    assert r.status_code == 200
    r = seeded_client.post("/v1/agent/ask", headers=h, json={"question": "Panadol dùng để làm gì?"})
    assert r.status_code == 200
    assert claude_env["n"] == 0


@pytest.mark.parametrize("msg", [
    "tôi muốn chết", "toi muon chet", "tôi không muốn sống nữa", "toi khong con muon song",
    "tôi tự tử", "tôi tự làm hại bản thân", "nghi bị đột quỵ", "nghi bi dot quy",
    "tôi thở khó", "tho kho qua", "khó thở lắm", "tôi không thở được", "khong tho duoc",
    "tôi không thể thở được", "tôi nghẹt thở", "nghet tho", "thở không nổi", "bị đau tim",
])
def test_b02_red_flag(msg):
    assert detect_red_flags(msg)


@pytest.mark.parametrize("msg", [
    "ngạt mũi", "nghẹt mũi về đêm", "bảo quản nơi khô ráo", "buồn nôn nhẹ",
    "thuốc này uống lúc nào", "tôi muốn biết liều Panadol", "khó uống thuốc viên to",
    "muốn mua thuốc ho", "không thể quên uống thuốc",
])
def test_b02_khong_bao_dong_nham(msg):
    assert detect_red_flags(msg) == []


def test_b02_tu_hai_co_loi_nhan_ho_tro():
    res = chat("tôi muốn chết", user_id=1, repo=_repo(), session_id="session-abc-123")
    txt = " ".join(s.text for s in res.steps if s.kind == "token")
    assert "115" in txt and "không đơn độc" in txt
    res2 = chat("tôi đau tim", user_id=1, repo=_repo(), session_id="session-abc-123")
    assert "không đơn độc" not in " ".join(s.text for s in res2.steps if s.kind == "token")


@pytest.mark.parametrize("src,label", [
    ("sinh 5 tháng 3 năm 1980", "<DATE>"), ("khám ngày 5/3", "<DATE>"),
    ("sinh 1980-03-05", "<DATE>"), ("BHYT DN4797912345678", "<INSURANCE>"),
    ("ở 12 Lê Lợi quận 1", "<ADDRESS>"), ("số 12 đường Lê Lợi", "<ADDRESS>"),
    ("gọi (028) 3822 1234", "<PHONE>"), ("gọi 024 38221234", "<PHONE>"),
    ("bệnh nhân Lê Thị B bị đau", "<NAME>"),
])
def test_b03_che(src, label):
    out = deidentify(src).text
    assert label in out
    assert "1980" not in out and "Lợi" not in out and "4797912345678" not in out
    assert "Thị" not in out and "3822" not in out and "38221234" not in out


@pytest.mark.parametrize("src", [
    "uống 2 viên 500mg ngày 3 lần", "uống 2 viên Paracetamol 500mg ngày 3 lần",
    "mua 2 hộp Panadol Extra", "đường huyết cao có uống được không",
    "Amoxicillin 250mg 3 viên mỗi ngày", "ngày 2 lần sau ăn",
])
def test_b03_khong_che_nham(src):
    assert deidentify(src).text == src


def test_b04_docs_tat_o_production(monkeypatch):
    import importlib

    import app.main as m

    monkeypatch.setenv("MEDASSIST_ENV", "production")
    # Production từ chối khoá mẫu/yếu (khoá của conftest chứa "do-not-use") -> dùng
    # khoá giả đủ mạnh chỉ trong test này.
    monkeypatch.setenv(
        "MEDASSIST_SECRET_KEY", "qa-prod-Zq8vN3rT5yK1wL9pB4mX7cD2hJ6fG0sAe"
    )
    m2 = importlib.reload(m)
    try:
        assert m2.app.docs_url is None and m2.app.redoc_url is None
        assert m2.app.openapi_url is None
    finally:
        monkeypatch.delenv("MEDASSIST_ENV")
        importlib.reload(m)


def test_c01_c02_no_data_nhac_hoi_duoc_si_va_giu_dau():
    res = chat("Thuốc Xyzqwerty là gì?", user_id=1, repo=_repo(), session_id="session-abc-123")
    txt = " ".join(s.text for s in res.steps if s.kind == "token")
    assert "dược sĩ" in txt
    assert "Xyzqwerty" in txt  # C-02: echo bản gốc có dấu
