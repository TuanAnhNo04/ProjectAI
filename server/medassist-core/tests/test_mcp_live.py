"""MCP chạy thật qua tiến trình con (stdio) — bắt lỗi mà test dùng mock không thấy.

Dữ liệu hoàn toàn giả (CSV mẫu), CSDL là file tạm; không có kết nối mạng.
"""

from __future__ import annotations

import pytest

from app.ingest import load_csv, to_rows
from app.mcp_tools import McpTool, _server_env
from app.sqlite_repository import SQLiteRepository


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    path = tmp_path / "mcp-live.db"
    repo = SQLiteRepository(path)
    try:
        repo.add_medicines(to_rows(load_csv()))
    finally:
        repo.close()
    monkeypatch.setenv("MEDASSIST_DB", str(path))
    return path


def test_server_env_truyen_db_nhung_khong_truyen_khoa_bi_mat(seeded_db, monkeypatch):
    monkeypatch.setenv("MEDASSIST_SECRET_KEY", "khoa-gia-khoa-gia-khoa-gia-khoa-gia-01")
    env = _server_env()
    assert env["MEDASSIST_DB"] == str(seeded_db)
    assert "MEDASSIST_SECRET_KEY" not in env


def test_server_env_khong_co_db_thi_khong_them_bien(monkeypatch):
    monkeypatch.delenv("MEDASSIST_DB", raising=False)
    assert "MEDASSIST_DB" not in _server_env()


def test_search_drug_qua_mcp_that_tim_thay_thuoc(seeded_db):
    tool = McpTool("search_drug", "Tìm thuốc")
    result = tool.run({"query": "Panadol"}, repo=None, user_id=0)  # type: ignore[arg-type]
    assert result.ok is True, result.reason
    assert result.citations, "MCP server phải thấy DB thuốc, không phải DB :memory: rỗng"
    assert any("Panadol" in c.medicine_name for c in result.citations)


def test_check_interaction_qua_mcp_that_van_tu_choi(seeded_db):
    tool = McpTool("check_interaction", "Kiểm tra tương tác")
    result = tool.run(
        {"drugs": ["Panadol", "Amoxicillin"]}, repo=None, user_id=0  # type: ignore[arg-type]
    )
    assert result.ok is False  # chưa có nguồn tương tác đã kiểm chứng
