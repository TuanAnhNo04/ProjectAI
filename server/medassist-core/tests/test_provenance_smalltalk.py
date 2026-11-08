"""Provenance của trích dẫn + small talk. Dữ liệu hoàn toàn GIẢ."""

from __future__ import annotations

import sqlite3

import pytest

from app.agent import answer_question
from app.import_real import run as import_run
from app.ingest import IngestError, load_csv, load_csv_mapped
from app.orchestrator import chat
from app.provenance import safe_source_url
from app.schemas import CitationResponse, MedicineResponse
from app.smalltalk import smalltalk_intent, smalltalk_reply
from app.sqlite_repository import SQLiteRepository


@pytest.fixture
def allow(monkeypatch):
    monkeypatch.setenv("MEDASSIST_SOURCE_ALLOWED_DOMAINS", "example.org, Gov.Example")


def test_allowlist_empty_default(monkeypatch):
    monkeypatch.delenv("MEDASSIST_SOURCE_ALLOWED_DOMAINS", raising=False)
    assert safe_source_url("https://example.org/a") == ""


def test_allowlist_cases(allow):
    assert safe_source_url("https://example.org/a") == "https://example.org/a"
    assert safe_source_url("https://sub.example.org/a") == "https://sub.example.org/a"
    assert safe_source_url("https://example.org:443/a")
    for bad in (
        "http://example.org/a",
        "https://evilexample.org/a",
        "https://example.org.evil.com/a",
        "https://user@example.org/a",
        "https://user:pw@example.org/a",
        "https://example.org:8443/a",
        "https://evil.com/?u=example.org",
        "javascript:alert(1)",
        "",
    ):
        assert safe_source_url(bad) == "", bad


def test_citation_schema_filters_url(allow):
    c = CitationResponse(id="R1", medicine_name="X", field="uses", excerpt="e",
                         source_url="https://evil.com/x", license=None)
    assert c.source_url == "" and c.license == ""
    c2 = CitationResponse(id="R1", medicine_name="X", field="uses", excerpt="e",
                          source_url="https://example.org/x")
    assert c2.source_url.startswith("https://example.org")


def test_migration_idempotent_on_old_db(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE medicines (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,"
        " normalized_name TEXT NOT NULL, composition TEXT, uses TEXT, side_effects TEXT,"
        " manufacturer TEXT, ingredients TEXT, excellent_review REAL, average_review REAL,"
        " poor_review REAL, source TEXT, is_authoritative INTEGER NOT NULL DEFAULT 0, updated_at TEXT)"
    )
    con.execute("INSERT INTO medicines (name, normalized_name, source) VALUES ('Fake', 'fake', 's')")
    con.commit()
    con.close()
    for _ in range(2):
        repo = SQLiteRepository(db)
        cols = {r["name"] for r in repo._conn.execute("PRAGMA table_info(medicines)")}
        assert {"source_url", "retrieved_at", "license", "reviewed_by"} <= cols
        repo.close()
    repo = SQLiteRepository(db)
    row = repo._conn.execute("SELECT source_url, reviewed_by FROM medicines").fetchone()
    assert row["source_url"] is None and row["reviewed_by"] is None
    repo.close()


def test_sample_data_has_no_provenance():
    for d in load_csv():
        assert d.source_url is None and d.retrieved_at is None
        assert d.license is None and d.reviewed_by is None


def _csv(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("Medicine_Name,Composition,Uses,Side_effects\nFakeMed,fakeine,Giảm đau giả,Buồn nôn giả\n",
                 encoding="utf-8")
    return p


def test_loader_authoritative_needs_reviewer(tmp_path):
    with pytest.raises(IngestError):
        load_csv_mapped(_csv(tmp_path), source="s", authoritative=True)
    with pytest.raises(IngestError):
        load_csv_mapped(_csv(tmp_path), source="s", source_url="http://example.org/x")
    drugs, _ = load_csv_mapped(
        _csv(tmp_path), source="s", authoritative=True, reviewed_by="Dược sĩ giả",
        source_url="https://example.org/x", retrieved_at="2026-10-30", license="CC-FAKE",
    )
    assert drugs[0].reviewed_by == "Dược sĩ giả" and drugs[0].retrieved_at == "2026-10-30"


def test_importer_flags_and_sse_flow(tmp_path, allow, capsys):
    db = tmp_path / "t.db"
    code = import_run([
        "--csv", str(_csv(tmp_path)), "--source", "Nguồn giả", "--db", str(db), "--apply",
        "--source-url", "https://example.org/doc", "--retrieved-at", "2026-10-30",
        "--license", "CC-FAKE", "--authoritative", "--reviewed-by", "Người duyệt giả",
    ])
    assert code == 0
    repo = SQLiteRepository(db)
    meds = repo.candidate_medicines("fakemed giảm đau")
    assert meds and meds[0].source_url == "https://example.org/doc"
    assert meds[0].reviewed_by == "Người duyệt giả"
    from app.retrieval import build_citations

    hits = build_citations(meds, "FakeMed giảm đau giả")
    assert hits and hits[0].citation.source_url == "https://example.org/doc"
    assert hits[0].citation.license == "CC-FAKE"
    repo.close()


def test_importer_authoritative_without_reviewer(tmp_path, capsys):
    code = import_run(["--csv", str(_csv(tmp_path)), "--source", "x", "--authoritative"])
    assert code == 2


def test_citation_sse_has_provenance_keys(monkeypatch):
    c = CitationResponse(id="R1", medicine_name="X", field="uses", excerpt="e")
    assert c.source_url == "" and c.retrieved_at == "" and c.license == "" and c.reviewed_by == ""
    m = MedicineResponse(name="X")
    assert m.source_url is None


@pytest.mark.parametrize("msg", [
    "Xin chào", "chào bạn", "Hello", "alo", "Bạn khỏe không?", "Cảm ơn bạn", "Tạm biệt",
    "Bạn là ai?", "Bạn làm được gì?", "ok", "được rồi", "hôm nay mình hơi buồn",
])
def test_smalltalk_detected(msg):
    assert smalltalk_reply(msg, "s1")


@pytest.mark.parametrize("msg", [
    "chào bạn, Panadol có tác dụng phụ gì", "tôi đau ngực", "hôm nay mình hơi mệt",
    "Panadol là thuốc gì", "x" * 80,
])
def test_not_smalltalk(msg):
    assert smalltalk_intent(msg) is None


def test_smalltalk_variants_deterministic():
    assert smalltalk_reply("xin chào", "a") == smalltalk_reply("xin chào", "a")
    assert len({smalltalk_reply("xin chào", f"s{i}") for i in range(30)}) > 1


def test_smalltalk_chat_steps():
    repo = SQLiteRepository(":memory:")
    res = chat("Xin chào", 1, repo, session_id="s")
    kinds = [s.kind for s in res.steps]
    assert set(kinds) == {"token", "done"} and kinds[-1] == "done"
    assert res.answered and res.mode == "extractive" and res.refusal_reason is None
    em = chat("tôi đau ngực", 1, repo, session_id="s")
    assert any(s.kind == "emergency" for s in em.steps)
    repo.close()
