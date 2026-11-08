"""`MedAssist` MCP server — Day 28.

Expose `search_drug` và `check_interaction` theo chuẩn MCP (Model Context Protocol),
cho phép **bất kỳ MCP client nào** gọi được — không phụ thuộc app Flutter.

## Chạy

    python -m app.mcp_server              # stdio (dùng với MCP Inspector / IDE)
    python -m app.mcp_server --transport sse  # SSE (dùng với web client)

## Vì sao chỉ expose 2 tool?

- `search_drug` + `check_interaction` là **public, không cần auth** — bất kỳ client
  nào cũng gọi được (dữ liệu thuốc là danh mục công khai).
- `schedule_reminder` + `explain_for_patient` cần `user_id` từ JWT token — không expose
  qua MCP vì client ngoài không có token hệ thống.
- `normalize_prescription` xử lý văn bản người nhập, không tra kho — MCP client tự làm
  được mà không cần kho thuốc.

## Bảo mật (`AGENTS.md`)

- Server chạy **localhost only** (không bind 0.0.0.0).
- Không log nội dung query/tên thuốc — chỉ log trace_id/latency nếu cần.
- `check_interaction` vẫn **từ chối** khi chưa có nguồn kiểm chứng — MCP không thay đổi
  hành vi an toàn của tool.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from app.sqlite_repository import SQLiteRepository
from app.tools import CheckInteractionTool, SearchDrugTool, ToolResult

mcp = MCPServer(
    "MedAssist Tools",
    instructions=(
        "MedAssist cung cấp tra cứu thuốc từ danh mục nội bộ. "
        "Kết quả chỉ mang tính tham khảo — không thay thế tư vấn dược sĩ/bác sĩ."
    ),
)

_repo: SQLiteRepository | None = None


def _get_repo() -> SQLiteRepository:
    """Kho SQLite dùng chung — khởi tạo lần đầu, đóng khi thoát."""
    global _repo
    if _repo is None:
        db_path = os.environ.get("MEDASSIST_DB", ":memory:")
        _repo = SQLiteRepository(db_path)
    return _repo


def _result_to_dict(result: ToolResult) -> dict[str, Any]:
    """Chuyển ``ToolResult`` thành dict JSON-safe cho MCP response."""
    return {
        "ok": result.ok,
        "data": result.data,
        "citations": [
            {
                "id": c.id,
                "medicine_name": c.medicine_name,
                "field": c.field,
                "excerpt": c.excerpt,
                "source": c.source,
                "is_authoritative": c.is_authoritative,
            }
            for c in result.citations
        ],
        "reason": result.reason,
        "user_message": result.user_message,
    }


@mcp.tool()
def search_drug(query: str) -> dict[str, Any]:
    """Tìm thuốc theo tên hoặc hoạt chất trong danh mục nội bộ.

    Args:
        query: Tên thuốc hoặc hoạt chất cần tìm (ví dụ: "paracetamol", "panadol").

    Returns:
        dict chứa ok, data (names + count), citations, reason, user_message.
    """
    tool = SearchDrugTool()
    result = tool.run({"query": query}, _get_repo(), user_id=0)
    return _result_to_dict(result)


@mcp.tool()
def check_interaction(drugs: list[str]) -> dict[str, Any]:
    """Kiểm tra tương tác giữa các thuốc.

    ⚠️ Hiện luôn từ chối vì chưa có nguồn dữ liệu tương tác đã kiểm chứng.
    Khi có nguồn, tool sẽ tự mở lại — không cần sửa code MCP.

    Args:
        drugs: Danh sách tên thuốc cần kiểm tra tương tác.

    Returns:
        dict chứa ok, data, citations, reason, user_message.
    """
    tool = CheckInteractionTool()
    result = tool.run({"drugs": drugs}, _get_repo(), user_id=0)
    return _result_to_dict(result)


if __name__ == "__main__":
    transport = "stdio"
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    mcp.run(transport=transport)
