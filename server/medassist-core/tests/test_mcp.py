"""Test MCP server + adapter — Day 28.

## Phạm vi

1. MCP server: tool functions trả đúng format ToolResult-as-dict.
2. MCP tool adapter: ``McpTool`` implement ``Tool`` protocol.
3. ``mcp_registry()``: đủ 5 tool, 2 qua MCP + 3 trực tiếp.
4. ``_parse_mcp_result()``: chuyển MCP response thành ``ToolResult``.

## ⚠️ Không test MCP transport end-to-end ở đây

Test subprocess stdio cần ``mcp`` package cài đặt + thời gian khởi động process.
Chạy end-to-end bằng MCP Inspector (xem ``docs/ROADMAP.md`` Day 28 DoD).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest


def test_mcp_server_search_drug_co_du_truong():
    """MCP ``search_drug`` trả dict có đủ 5 trường chuẩn."""
    from app.mcp_server import search_drug

    result = search_drug(query="notexist_xyz")
    assert isinstance(result, dict)
    for key in ("ok", "data", "citations", "reason", "user_message"):
        assert key in result, f"Thiếu trường '{key}'"


def test_mcp_server_search_drug_khong_tim_thay():
    """``search_drug`` trả ``ok=False`` khi không tìm thấy thuốc."""
    from app.mcp_server import search_drug

    result = search_drug(query="thuoc_khong_ton_tai_abc")
    assert result["ok"] is False
    assert result["reason"] == "no_data"


def test_mcp_server_search_drug_thieu_query():
    """``search_drug`` trả ``ok=False`` khi query rỗng."""
    from app.mcp_server import search_drug

    result = search_drug(query="")
    assert result["ok"] is False
    assert result["reason"] == "bad_args"


def test_mcp_server_check_interaction_luon_tu_choi():
    """``check_interaction`` luôn từ chối — chưa có nguồn kiểm chứng."""
    from app.mcp_server import check_interaction

    result = check_interaction(drugs=["warfarin", "panadol"])
    assert result["ok"] is False
    assert result["reason"] == "no_data"
    assert "dược sĩ" in result["user_message"].lower() or "bác sĩ" in result["user_message"].lower()


def test_mcp_server_check_interaction_ghi_ten_thuoc():
    """Tool ghi lại tên thuốc người dùng hỏi (dữ liệu do chính họ nhập)."""
    from app.mcp_server import check_interaction

    result = check_interaction(drugs=["aspirin", "ibuprofen"])
    msg = result["user_message"]
    assert "aspirin" in msg
    assert "ibuprofen" in msg


# ---- MCP Tool Adapter ----


def test_mcp_tool_co_name_va_description():
    """``McpTool`` cung cấp đủ trường cho ``Tool`` protocol."""
    from app.mcp_tools import McpTool

    tool = McpTool(name="search_drug", description="Tìm thuốc.")
    assert tool.name == "search_drug"
    assert tool.description == "Tìm thuốc."


def test_mcp_tool_mcp_name_override():
    """``mcp_tool_name`` cho phép tên MCP khác tên registry."""
    from app.mcp_tools import McpTool

    tool = McpTool(name="my_search", description="x", mcp_tool_name="search_drug")
    assert tool._mcp_tool_name == "search_drug"


# ---- _parse_mcp_result ----


@dataclass
class _FakeContent:
    text: str


@dataclass
class _FakeResult:
    isError: bool
    content: list[Any]


def test_parse_mcp_result_ok():
    """Parse kết quả thành công từ MCP."""
    from app.mcp_tools import _parse_mcp_result

    data = {
        "ok": True,
        "data": {"names": ["Paracetamol"], "count": 1},
        "citations": [
            {
                "id": "T1",
                "medicine_name": "Paracetamol",
                "field": "composition",
                "excerpt": "Paracetamol 500mg",
                "source": "drugbank",
                "is_authoritative": True,
            }
        ],
        "reason": None,
        "user_message": None,
    }

    fake = _FakeResult(isError=False, content=[_FakeContent(json.dumps(data))])
    result = _parse_mcp_result(fake)

    assert result.ok is True
    assert result.data["count"] == 1
    assert len(result.citations) == 1
    assert result.citations[0].medicine_name == "Paracetamol"


def test_parse_mcp_result_error():
    """Parse lỗi từ MCP trả ``ok=False``."""
    from app.mcp_tools import _parse_mcp_result

    fake = _FakeResult(isError=True, content=[_FakeContent("Lỗi tool")])
    result = _parse_mcp_result(fake)

    assert result.ok is False
    assert result.reason == "mcp_tool_error"


def test_parse_mcp_result_empty():
    """MCP trả content rỗng → ``ok=False``."""
    from app.mcp_tools import _parse_mcp_result

    fake = _FakeResult(isError=False, content=[])
    result = _parse_mcp_result(fake)

    assert result.ok is False
    assert result.reason == "mcp_empty"


# ---- mcp_registry ----


def test_mcp_registry_co_du_5_tool():
    """``mcp_registry()`` vẫn đăng ký đủ 5 tool (2 MCP + 3 trực tiếp)."""
    from app.mcp_tools import mcp_registry

    registry = mcp_registry()
    names = registry.names()
    assert "search_drug" in names
    assert "check_interaction" in names
    assert "normalize_prescription" in names
    assert "explain_for_patient" in names
    assert "schedule_reminder" in names
    assert len(names) == 5


def test_mcp_registry_search_drug_la_McpTool():
    """``search_drug`` trong MCP registry là ``McpTool``, không phải ``SearchDrugTool``."""
    from app.mcp_tools import McpTool, mcp_registry

    registry = mcp_registry()
    tool = registry.get("search_drug")
    assert isinstance(tool, McpTool)


def test_mcp_registry_explain_van_la_truc_tiep():
    """``explain_for_patient`` vẫn chạy trực tiếp (cần ``repo``)."""
    from app.mcp_tools import McpTool, mcp_registry
    from app.tools import ExplainForPatientTool

    registry = mcp_registry()
    tool = registry.get("explain_for_patient")
    assert isinstance(tool, ExplainForPatientTool)
    assert not isinstance(tool, McpTool)


# ---- Feature flag ----


def test_feature_flag_mac_dinh_la_direct(monkeypatch):
    """Không set ``MEDASSIST_USE_MCP`` → dùng ``default_registry()``."""
    monkeypatch.delenv("MEDASSIST_USE_MCP", raising=False)
    # Re-import to pick up env change
    import importlib
    import app.orchestrator as orch
    monkeypatch.setattr(orch, "_USE_MCP", False)

    from app.mcp_tools import McpTool
    registry = orch._default_or_mcp_registry()
    tool = registry.get("search_drug")
    assert not isinstance(tool, McpTool)


def test_feature_flag_mcp_khi_bat(monkeypatch):
    """``MEDASSIST_USE_MCP=1`` → ``search_drug`` là ``McpTool``."""
    import app.orchestrator as orch
    monkeypatch.setattr(orch, "_USE_MCP", True)

    from app.mcp_tools import McpTool
    registry = orch._default_or_mcp_registry()
    tool = registry.get("search_drug")
    assert isinstance(tool, McpTool)
