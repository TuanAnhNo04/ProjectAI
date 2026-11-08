"""Khẩn cấp, đối chiếu tác dụng phụ, health_context, lifestyle — DỮ LIỆU GIẢ hoàn toàn."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.orchestrator import chat
from app.safety import detect_red_flags, detect_symptoms, is_symptom_report
from app.schemas import AgentChatRequest, LifestyleRequest, LLMResponse
from app.sessions import InMemorySessionStore, get_session_store

FAKE_MEDS = [
    {
        "name": "Fakedol 500",
        "composition": "Fakeamol 500mg",
        "uses": "Giảm đau (dữ liệu giả)",
        "side_effects": "Buồn nôn, chóng mặt",
        "source": "fake-test",
    },
    {
        "name": "Testcillin",
        "composition": "Testcillin trihydrate",
        "uses": "Dữ liệu giả",
        "side_effects": "Tiêu chảy, phát ban",
        "source": "fake-test",
    },
    {
        "name": "Mockazin",
        "composition": "Mockazine 10mg",
        "uses": "Dữ liệu giả",
        "side_effects": "Drowsiness, dry mouth",
        "source": "fake-test",
    },
]


@pytest.fixture()
def fake_repo(repo):
    repo.add_medicines(FAKE_MEDS)
    return repo


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    event = None
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and event:
            out.append((event, json.loads(line.split(":", 1)[1].strip())))
            event = None
    return out


def _text(result) -> str:
    return "".join(s.text for s in result.steps if s.kind == "token")


# ------------------------------------------------------------------ red flags


@pytest.mark.parametrize(
    "msg, expected",
    [
        ("Tôi bị khó thở sau khi uống thuốc", "khó thở"),
        ("toi bi kho tho qua", "khó thở"),
        ("Bỗng dưng đau ngực dữ dội", "đau ngực"),
        ("con toi bi co giat", "co giật"),
        ("Mẹ tôi bị ngất xỉu", "ngất xỉu"),
        ("sưng môi và sưng lưỡi", "sưng môi"),
        ("đi ngoai ra mau", "đi ngoài ra máu"),
        ("TÔI BỊ HO RA MÁU", "ho ra máu"),
    ],
)
def test_red_flag_detected(msg, expected):
    assert expected in detect_red_flags(msg)


@pytest.mark.parametrize(
    "msg",
    [
        "Panadol có tác dụng gì?",
        "Tôi bị ngạt mũi",
        "toi bi ngat mui",
        "bao quan noi kho rao",
        "Bảo quản nơi khô ráo",
        "pho bo co ngon khong",
        "Cho tôi xem phần về tác dụng phụ",
        "Uống thuốc ngắt quãng có sao không",
    ],
)
def test_no_false_red_flag(msg):
    assert detect_red_flags(msg) == []


def test_emergency_turn_skips_llm_and_session(fake_repo):
    class ExplodingLLM:
        name = "claude"

        def generate(self, *a, **k):  # pragma: no cover - không được gọi
            raise AssertionError("LLM không được gọi ở luồng khẩn cấp")

    store = InMemorySessionStore()
    result = chat(
        "kho tho va sung moi sau khi uong Fakedol",
        user_id=1,
        repo=fake_repo,
        session=store,
        session_id="sess-emergency-1",
        llm=ExplodingLLM(),
        current_medications=["Fakedol"],
        cloud_ai_consent=True,
    )
    kinds = [s.kind for s in result.steps]
    assert kinds == ["emergency", "token", "done"]
    data = result.steps[0].data
    assert "khó thở" in data["matched"] and "115" in data["message"]
    assert "115" in _text(result)
    assert store.history(1, "sess-emergency-1") == []


def test_emergency_sse_event(fake_repo, client, auth):
    tok = auth.token("capcuu")
    resp = client.post(
        "/v1/agent/chat",
        json={"message": "Tôi bị đau ngực và khó thở", "session_id": "sess-sse-0001"},
        headers=auth.header(tok),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names[0] == "emergency" and names[-1] == "done"
    payload = events[0][1]
    assert set(payload) == {"matched", "message"}
    assert {"đau ngực", "khó thở"} <= set(payload["matched"])
    assert "115" in payload["message"]


# ----------------------------------------------------------- tác dụng phụ


def test_symptom_detection_accent_insensitive():
    assert "buồn nôn" in detect_symptoms("toi bi buon non")
    assert is_symptom_report("Tôi thấy chóng mặt sau khi uống thuốc")
    assert not is_symptom_report("Fakedol có tác dụng gì")


def test_side_effect_cites_only_matching_records(fake_repo):
    result = chat(
        "Tôi bị buồn nôn sau khi uống thuốc",
        user_id=1,
        repo=fake_repo,
        current_medications=["Fakedol 500", "Testcillin", "Khongcothuoc"],
    )
    cites = [s.citation for s in result.steps if s.kind == "citation"]
    assert [c.medicine_name for c in cites] == ["Fakedol 500"]
    assert cites[0].field == "side_effects"
    text = _text(result)
    assert f"[{cites[0].id}]" in text
    assert "KHÔNG khẳng định thuốc là nguyên nhân" in text
    assert "Testcillin" in text and "Khongcothuoc" in text  # liệt kê thuốc không thấy
    low = text.lower()
    for bad in ("do thuốc gây ra", "chắc chắn là do", "nguyên nhân là"):
        assert bad not in low
    assert "không tự ý ngưng thuốc" in low


def test_side_effect_english_source_text(fake_repo):
    result = chat(
        "Tôi bị khô miệng sau khi dùng thuốc",
        user_id=1,
        repo=fake_repo,
        current_medications=["Mockazin"],
    )
    cites = [s.citation for s in result.steps if s.kind == "citation"]
    assert [c.medicine_name for c in cites] == ["Mockazin"]


def test_side_effect_no_match_says_insufficient(fake_repo):
    result = chat(
        "Tôi bị mất ngủ sau khi uống thuốc",
        user_id=1,
        repo=fake_repo,
        current_medications=["Fakedol 500"],
    )
    assert not [s for s in result.steps if s.kind == "citation"]
    assert "Không đủ căn cứ" in _text(result)
    assert result.answered is False


def test_allergy_warning_cited(fake_repo):
    from app.schemas import HealthContext

    result = chat(
        "Tôi bị buồn nôn sau khi uống thuốc",
        user_id=1,
        repo=fake_repo,
        current_medications=["Fakedol 500"],
        health_context=HealthContext(allergies=["fakeamol"], conditions=["benh gia"]),
    )
    cites = [s.citation for s in result.steps if s.kind == "citation"]
    allergy = [c for c in cites if c.field == "composition"]
    assert allergy and f"[{allergy[0].id}]" in _text(result)
    assert "benh gia" not in _text(result)  # không lặp lại bệnh nền


def test_cloud_rephrase_receives_deidentified_question(fake_repo):
    seen: list[str] = []

    class FakeCloud:
        name = "claude"

        def generate(self, messages, *, max_tokens=512, temperature=0.0):
            seen.append(messages[-1].content)
            assert "dược sĩ tư vấn" in messages[0].content
            return LLMResponse(text="Bịa ra [R9]", provider="claude")

    result = chat(
        "Tôi bị buồn nôn sau khi uống thuốc, sđt 0912345678",
        user_id=1,
        repo=fake_repo,
        current_medications=["Fakedol 500"],
        llm=FakeCloud(),
        cloud_ai_consent=True,
    )
    assert seen and "0912345678" not in seen[0]
    # Bản viết lại có mã bịa -> bị loại, giữ bản luật.
    assert "Bịa ra" not in _text(result)


def test_cloud_not_called_without_consent(fake_repo):
    class FakeCloud:
        name = "claude"

        def generate(self, *a, **k):  # pragma: no cover
            raise AssertionError("không được gọi khi chưa đồng ý")

    chat(
        "Tôi bị buồn nôn sau khi uống thuốc",
        user_id=1,
        repo=fake_repo,
        current_medications=["Fakedol 500"],
        llm=FakeCloud(),
    )


# ------------------------------------------------------- health_context validation


def _req(**hc):
    return {"message": "x", "session_id": "sess-12345", "health_context": hc}


def test_health_context_limits():
    AgentChatRequest(**_req(conditions=["a"] * 20, allergies=["b" * 100]))
    with pytest.raises(ValidationError):
        AgentChatRequest(**_req(conditions=["a"] * 21))
    with pytest.raises(ValidationError):
        AgentChatRequest(**_req(allergies=["b" * 101]))
    with pytest.raises(ValidationError):
        AgentChatRequest(**_req(allergies=[""]))
    with pytest.raises(ValidationError):
        AgentChatRequest(**_req(unknown=["x"]))
    assert AgentChatRequest(message="x", session_id="sess-12345").health_context is None


# ------------------------------------------------------------------ lifestyle


def test_lifestyle_requires_auth(client):
    assert client.post("/v1/agent/lifestyle", json={}).status_code == 401


def test_lifestyle_tips_shape_and_no_persistence(fake_repo, client, auth):
    tok = auth.token("loisong")
    before_meds = fake_repo.list_medicines(1, 0).total
    sessions_before = len(get_session_store()._data)
    resp = client.post(
        "/v1/agent/lifestyle",
        json={
            "current_medications": ["Fakedol 500", "Mockazin", "Khongco"],
            "conditions": ["benh gia"],
            "recent_topics": ["mất ngủ gần đây"],
        },
        headers=auth.header(tok),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body) == {"tips", "disclaimer", "generated_by", "unmatched_conditions"}
    assert body["generated_by"] == "rules"
    tips = body["tips"]
    assert 1 <= len(tips) <= 5
    meds = [t for t in tips if t["kind"] == "medication"]
    general = [t for t in tips if t["kind"] == "general"]
    assert {t["citations"][0]["medicine_name"] for t in meds} == {"Fakedol 500", "Mockazin"}
    for t in meds:
        c = t["citations"][0]
        assert set(c) == {"id", "medicine_name", "source", "excerpt", "is_authoritative",
                              "source_url", "retrieved_at", "license", "reviewed_by"}
        assert f"[{c['id']}]" in t["body"]
    for t in general:
        assert t["citations"] == []
        assert not any(ch.isdigit() for ch in t["body"])  # không liều/con số
        assert "benh gia" not in t["body"]
    # Chủ đề "mất ngủ" -> gợi ý ngủ được ưu tiên trong nhóm general.
    assert general[0]["title"] == "Ngủ đủ giấc"
    # Không ghi gì: không thêm thuốc, không lịch sử, không phiên.
    assert fake_repo.list_medicines(1, 0).total == before_meds
    assert len(get_session_store()._data) == sessions_before
    uid = client.get("/v1/me", headers=auth.header(tok))
    if uid.status_code == 200:
        assert fake_repo.count_history(uid.json()["id"]) == 0


def test_lifestyle_validation_limits(client, auth):
    tok = auth.token("loisong2")
    h = auth.header(tok)
    assert client.post("/v1/agent/lifestyle", json={"recent_topics": ["a"] * 6}, headers=h).status_code == 422
    assert client.post("/v1/agent/lifestyle", json={"recent_topics": ["a" * 201]}, headers=h).status_code == 422
    assert client.post("/v1/agent/lifestyle", json={"conditions": ["a"] * 21}, headers=h).status_code == 422
    ok = client.post("/v1/agent/lifestyle", json={}, headers=h)
    assert ok.status_code == 200 and len(ok.json()["tips"]) == 5


def test_lifestyle_cloud_rewrite_rejects_new_numbers(fake_repo):
    from app.lifestyle import build_lifestyle_tips

    class FakeCloud:
        name = "claude"

        def generate(self, messages, *, max_tokens=512, temperature=0.0):
            data = json.loads(messages[-1].content.split("Gợi ý: ", 1)[1].split("\n")[0])
            out = [{"title": d["title"], "body": d["body"] + " Uống 3 viên."} for d in data]
            return LLMResponse(text=json.dumps(out, ensure_ascii=False), provider="claude")

    req = LifestyleRequest(current_medications=["Fakedol 500"], cloud_ai_consent=True)
    res = build_lifestyle_tips(req, fake_repo, provider=FakeCloud())
    assert res.generated_by == "rules"
    assert len(res.tips) <= 5


def test_lifestyle_cloud_never_rewrites_cited_tips(fake_repo):
    """Tip có trích dẫn (medication/diet) giữ NGUYÊN VĂN dù Claude "diễn đạt" thêm lời khuyên
    mới bằng chữ (không có số nên `_rewrite_ok` không bắt được)."""
    from app.lifestyle import build_lifestyle_tips

    class FakeCloud:
        name = "claude"

        def generate(self, messages, *, max_tokens=512, temperature=0.0):
            data = json.loads(messages[-1].content.split("Gợi ý: ", 1)[1].split("\n")[0])
            out = [
                {"title": d["title"], "body": d["body"] + " Hãy cẩn trọng khi lái xe."}
                for d in data
            ]
            return LLMResponse(text=json.dumps(out, ensure_ascii=False), provider="claude")

    req = LifestyleRequest(current_medications=["Fakedol 500"], cloud_ai_consent=True)
    plain = build_lifestyle_tips(
        LifestyleRequest(current_medications=["Fakedol 500"]), fake_repo
    )
    res = build_lifestyle_tips(req, fake_repo, provider=FakeCloud())
    cited_plain = [(t.title, t.body) for t in plain.tips if t.citations]
    cited_cloud = [(t.title, t.body) for t in res.tips if t.citations]
    assert cited_plain and cited_cloud == cited_plain
    # Câu do FakeCloud chèn thêm (viết hoa "Hãy", có dấu chấm) không được xuất hiện.
    assert all("Hãy cẩn trọng khi lái xe." not in t.body for t in res.tips if t.citations)
