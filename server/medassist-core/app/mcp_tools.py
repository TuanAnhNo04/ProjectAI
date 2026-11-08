"""MCP client adapter — Day 28. Tool gọi qua MCP thay vì import trực tiếp.

## Kiến trúc

Mỗi ``McpTool`` implement ``Tool`` protocol (``app/tools.py``):

    orchestrator  →  ToolRegistry.run("search_drug", args)
                     └─ McpTool.run(args)
                        └─ MCP client  →  MCP server (stdio)
                                          └─ SearchDrugTool (trực tiếp)

Orchestrator **không đổi một dòng** — nó vẫn gọi ``registry.run(name, args)``.
Khác biệt duy nhất: factory ``mcp_registry()`` đăng ký ``McpTool`` thay vì tool trực
tiếp cho 2 tool public (``search_drug``, ``check_interaction``).

## Transport

Dùng **stdio** — MCP server chạy như subprocess, giao tiếp qua stdin/stdout.
Đơn giản nhất, không cần port, không cần HTTP.

## Bảo mật

- MCP server chạy **cùng máy**, không gửi dữ liệu ra ngoài.
- ``user_id`` cho tool public = 0 (không cần auth).
- Tool cần auth (``schedule_reminder``, ``explain_for_patient``) vẫn chạy trực tiếp —
  không expose qua MCP vì client ngoài không có JWT token.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from app.schemas import CitationResponse
from app.sqlite_repository import SQLiteRepository
from app.tools import (
    ExplainForPatientTool,
    NormalizePrescriptionTool,
    ScheduleReminderTool,
    ToolRegistry,
    ToolResult,
)

_MCP_SERVER_MODULE = "app.mcp_server"


class McpTool:
    """Adapter: gọi MCP server qua stdio, trả về ``ToolResult``.

    Mỗi lần ``run()`` mở 1 phiên MCP ngắn (connect → call_tool → close).
    Chi phí subprocess nhỏ so với I/O mạng — và pattern này chứng minh rằng
    orchestrator hoạt động được với tool ở **process khác**.
    """

    def __init__(self, name: str, description: str, mcp_tool_name: str | None = None):
        self.name = name
        self.description = description
        self._mcp_tool_name = mcp_tool_name or name

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        del repo, user_id
        try:
            return asyncio.run(self._call_mcp(arguments))
        except Exception as exc:
            return ToolResult(
                ok=False,
                reason="mcp_error",
                user_message=f"Không kết nối được MCP server: {type(exc).__name__}",
            )

    async def _call_mcp(self, arguments: dict[str, Any]) -> ToolResult:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", _MCP_SERVER_MODULE],
            env=_server_env(),
        )

        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    self._mcp_tool_name, arguments=arguments
                )

        return _parse_mcp_result(result)


def _server_env() -> dict[str, str]:
    """Môi trường cho tiến trình MCP server.

    ⚠️ `env=None` **không** thừa kế biến của ta: SDK chỉ cấp một tập an toàn (PATH,
    HOME...). Thiếu `MEDASSIST_DB` thì server MCP mở CSDL `:memory:` **rỗng** và mọi
    `search_drug` trả `no_data` — lỗi lặng lẽ, test mock không thấy. Vì vậy truyền
    đúng một biến này, và **không** truyền khoá bí mật JWT (server MCP không cần).
    """
    from mcp.client.stdio import get_default_environment

    env = dict(get_default_environment())
    db_path = os.environ.get("MEDASSIST_DB")
    if db_path:
        env["MEDASSIST_DB"] = db_path
    return env


def _parse_mcp_result(mcp_result: Any) -> ToolResult:
    """Chuyển MCP ``CallToolResult`` thành ``ToolResult`` nội bộ."""
    # SDK mcp 2.x đặt tên `is_error`; bản cũ/test giả dùng `isError` — nhận cả hai.
    # ⚠️ Chỉ nhận một tên thì lỗi thật bị `except Exception` ở `run()` nuốt thành "mcp_error".
    is_error = getattr(mcp_result, "is_error", getattr(mcp_result, "isError", False))
    if is_error:
        error_text = ""
        if mcp_result.content:
            error_text = mcp_result.content[0].text if hasattr(mcp_result.content[0], "text") else ""
        return ToolResult(
            ok=False,
            reason="mcp_tool_error",
            user_message=error_text or "MCP tool trả về lỗi.",
        )

    if not mcp_result.content:
        return ToolResult(ok=False, reason="mcp_empty", user_message="MCP không trả dữ liệu.")

    text = mcp_result.content[0].text if hasattr(mcp_result.content[0], "text") else ""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return ToolResult(ok=True, data={"raw": text})

    citations = [
        CitationResponse(**c) for c in data.get("citations", [])
    ]

    return ToolResult(
        ok=data.get("ok", True),
        data=data.get("data", {}),
        citations=citations,
        reason=data.get("reason"),
        user_message=data.get("user_message"),
    )


def mcp_registry() -> ToolRegistry:
    """Registry dùng MCP cho 2 tool public, trực tiếp cho 3 tool cần auth.

    So với ``default_registry()``:
    - ``search_drug`` → ``McpTool`` (qua MCP server)
    - ``check_interaction`` → ``McpTool`` (qua MCP server)
    - ``normalize_prescription`` → trực tiếp (xử lý text, không cần kho)
    - ``explain_for_patient`` → trực tiếp (cần ``repo`` để truy xuất)
    - ``schedule_reminder`` → trực tiếp (cần ``user_id`` từ token)
    """
    registry = ToolRegistry()

    registry.register(McpTool(
        name="search_drug",
        description="Tìm thuốc theo tên hoặc hoạt chất (qua MCP server).",
    ))
    registry.register(McpTool(
        name="check_interaction",
        description="Kiểm tra tương tác giữa các thuốc (qua MCP server).",
    ))

    for tool in (
        NormalizePrescriptionTool(),
        ExplainForPatientTool(),
        ScheduleReminderTool(),
    ):
        registry.register(tool)

    return registry
