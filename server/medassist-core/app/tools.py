"""Tool của agent — Day 22. **Interface + registry**, không phụ thuộc MCP.

## Vì sao bọc tool sau interface?

`ROADMAP.md` Day 28 sẽ thay bằng MCP server. Nếu orchestrator gọi tool trực tiếp
kiểu `check_interaction(...)` thì ngày đó phải sửa orchestrator. Nên ở đây:

- orchestrator chỉ biết [ToolRegistry];
- tool là [Tool] protocol, chạy qua `registry.run(name, arguments)`;
- thay MCP = viết adapter `McpTool(Tool)` rồi đăng ký vào registry — **không sửa
  orchestrator, không sửa route**.

## ⚠️ `check_interaction` LUÔN từ chối — và đây là hành vi đúng

`ROADMAP.md` yêu cầu test: *"Tôi đang uống warfarin, thêm panadol được không?" →
agent phải gọi tool tương tác và trả lời có căn cứ*.

Ta **gọi tool đúng như yêu cầu**. Nhưng tool trả *"không có nguồn để kiểm chứng"* —
vì máy không có bảng tương tác thuốc, mà `AGENTS.md` mục 3 cấm tự bịa quan hệ tương
tác. Tương tác thuốc là chỗ sai sót **có hậu quả thật** (người bệnh uống hai thuốc
đối kháng).

Nói rõ: **đạt được "gọi đúng tool", chưa đạt được "trả lời có căn cứ"** — vì chưa
tồn tại nguồn căn cứ. Xem `docs/ROADMAP.md` mục Day 22–23.

## Quy tắc chung cho mọi tool

1. `ToolResult` **luôn** mang citation kèm `source` và `is_authoritative` (Đề bài #3).
2. Tool **không bao giờ** tự sinh thông tin không đọc được từ kho.
3. Tool nhận `user_id` từ **token** (`deps.current_user_id`), không nhận từ tham số
   do client gửi — nếu không thì người dùng tự khai id người khác → IDOR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.ingest import has_interaction_data
from app.schemas import CitationResponse, MedicineResponse
from app.sqlite_repository import SQLiteRepository

#: Mã kết quả — orchestrator/UI xử lý theo mã, **không** theo text.
TOOL_NO_DATA = "no_data"
TOOL_REFUSED = "refused"
TOOL_BAD_ARGS = "bad_args"


@dataclass(frozen=True)
class ToolResult:
    """Kết quả một lần gọi tool.

    ⚠️ `data` là **dữ liệu đã kiểm chứng được** — orchestrator chỉ được diễn giải từ
    đây, không tự thêm thông tin.
    """

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    citations: list[CitationResponse] = field(default_factory=list)
    reason: str | None = None
    #: Lời nhắc cho người dùng khi tool không có dữ liệu. Luôn hướng về chuyên gia:
    #: đây là ứng dụng tham khảo, không phải bác sĩ.
    user_message: str | None = None


class Tool(Protocol):
    """Hợp đồng một tool.

    ⚠️ `arguments` là dict thô **do planner tạo ra**, chưa kiểm tra kiểu. Mỗi tool tự
    validate và trả `ToolResult(ok=False, reason=TOOL_BAD_ARGS)` — planner là mô hình,
    không phải nơi tin cậy.
    """

    name: str
    description: str

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        """Chạy tool. **Không** ném exception ra ngoài — lỗi thành `ToolResult`."""
        ...


def cite_from(med: MedicineResponse, field_name: str, value: str, idx: int) -> CitationResponse:
    """Dựng citation từ bản ghi thuốc — `source` **luôn** lấy từ kho."""
    return CitationResponse(
        id=f"T{idx}",
        medicine_name=med.name,
        field=field_name,
        excerpt=" ".join(value.split())[:400],
        source=med.source,
        is_authoritative=med.is_authoritative,
        source_url=med.source_url or "",
        retrieved_at=med.retrieved_at or "",
        license=med.license or "",
        reviewed_by=med.reviewed_by or "",
    )


class SearchDrugTool:
    """Tìm thuốc theo tên hoặc hoạt chất.

    ⚠️ Cố ý **chỉ** trả về thứ có trong kho. Không có kho -> `TOOL_NO_DATA`, không
    suy đoán từ trí nhớ mô hình.
    """

    name = "search_drug"
    description = "Tìm thuốc theo tên hoặc hoạt chất trong danh mục nội bộ."

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        del user_id  # tra cứu công khai, không phụ thuộc người gọi
        query = str(arguments.get("query", "")).strip()
        if not query:
            return ToolResult(
                ok=False,
                reason=TOOL_BAD_ARGS,
                user_message="Thiếu tên hoặc hoạt chất cần tìm.",
            )

        # Thử có dấu trước; không ra thì thử bỏ khoảng trắng ("panadol extra").
        found: list[MedicineResponse] = []
        for candidate in (query, query.replace(" ", "")):
            found = repo.search_medicines(candidate, 5, 0).results
            if found:
                break

        if not found:
            # Hướng 2: chỉ khi chủ dự án đã cấu hình nguồn; mặc định trả rỗng, không gọi mạng.
            from app import live_sources

            found = live_sources.lookup(query)

        if not found:
            return ToolResult(
                ok=False,
                reason=TOOL_NO_DATA,
                user_message=(
                    f"Không tìm thấy thuốc nào khớp với '{query}' trong danh mục hiện có."
                ),
            )

        return ToolResult(
            ok=True,
            data={"names": [m.name for m in found], "count": len(found)},
            citations=[
                cite_from(m, "composition", m.composition or m.name, i + 1)
                for i, m in enumerate(found)
            ],
        )


class CheckInteractionTool:
    """Kiểm tra tương tác giữa các thuốc.

    ⚠️⚠️ **Luôn từ chối ở thời điểm hiện tại.**

    Máy không có bảng tương tác thuốc đã kiểm chứng. `AGENTS.md` mục 3 cấm tự bịa
    dữ liệu tương tác — đây là chỗ sai sót **có hậu quả thật**: người bệnh dùng hai
    thuốc đối kháng.

    [has_interaction_data] là **cổng chặn duy nhất** cho phép mở lại tool này.
    """

    name = "check_interaction"
    description = "Kiểm tra tương tác giữa hai thuốc (chưa có nguồn — luôn từ chối)."

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        del repo, user_id
        drugs = [str(d).strip() for d in (arguments.get("drugs") or []) if str(d).strip()]

        if not has_interaction_data():
            return ToolResult(
                ok=False,
                reason=TOOL_NO_DATA,
                # ⚠️ Nêu lại tên thuốc đã gửi: đây là dữ liệu **do chính người dùng
                # nhập**, không phải dữ liệu bệnh nhân từ kho — đưa lại cho họ xem
                # là hợp lệ và giúp họ biết hệ thống đã hiểu đúng chưa.
                user_message=(
                    "Chưa có nguồn dữ liệu tương tác thuốc đã kiểm chứng"
                    + (f" (bạn đang hỏi về: {', '.join(drugs)})" if drugs else "")
                    + ". Tương tác thuốc là chỗ sai sót có hậu quả thật, nên tôi không "
                    "đoán bừa. Hãy hỏi dược sĩ hoặc bác sĩ."
                ),
            )
        # Nhánh dưới đây **không** chạm tới cho tới khi có nguồn thật.
        return ToolResult(ok=False, reason=TOOL_NO_DATA, user_message="Chưa hỗ trợ.")

        source=med.source,
        is_authoritative=med.is_authoritative,
#: Đơn vị liều thường gặp trong đơn thuốc Việt Nam — dùng để **bỏ khoảng trắng
#: thừa** ("500 mg" -> "500mg"), KHÔNG phải để diễn giải đơn vốn đã quen thuộc.
_UNIT_RE = re.compile(r"(\d)\s*(mg|mcg|g|ml|iu|vòng|viên)\b", re.IGNORECASE)


class NormalizePrescriptionTool:
    """Chuẩn hoá **hình thức** chữ viết trong đơn thuốc.

    ⚠️ Cố ý **không diễn giải** đơn thuốc. Chỉ: bỏ khoảng trắng thừa, gộp đơn vị
    liều, bỏ dòng trống. Không suy đoán thuốc nào trong ảnh, không đoán hàm lượng.
    Nhiệm vụ suy luận là của người dùng và bác sĩ.
    """

    name = "normalize_prescription"
    description = "Chuẩn hoá chữ viết trong đơn thuốc (không diễn giải)."

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        del repo, user_id
        raw = str(arguments.get("text", "")).strip()
        if not raw:
            return ToolResult(
                ok=False,
                reason=TOOL_BAD_ARGS,
                user_message="Chưa có nội dung đơn thuốc để chuẩn hoá.",
            )

        lines = [" ".join(line.split()) for line in raw.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        normalized = _UNIT_RE.sub(r"\1\2", cleaned)

        return ToolResult(
            ok=True,
            data={
                "original": cleaned,
                "normalized": normalized,
                "changed": normalized != cleaned,
            },
            # ⚠️ Đây là **văn bản do người dùng nhập**, không phải dữ liệu từ kho ->
            # `source=None`, `is_authoritative=False`. Tuyệt đối không gắn nguồn giả.
            citations=[
                CitationResponse(
                    id="P1",
                    medicine_name="(văn bản người dùng nhập)",
                    field="normalized_text",
                    excerpt=normalized[:400],
                    source=None,
                    is_authoritative=False,
                )
            ],
        )


class ExplainForPatientTool:
    """Diễn giải thông tin thuốc cho người bệnh, chỉ dựa trên nguồn đã có.

    ⚠️ Công cụ này **không tự diễn giải**. Nó chỉ gom các trích dẫn đã truy hồi được;
    việc diễn giải là của orchestrator — vì diễn giải bằng mô hình mà không có nguồn
    thì đúng nghĩa là **bịa**.
    """

    name = "explain_for_patient"
    description = "Diễn giải thông tin thuốc cho người bệnh, chỉ dựa trên nguồn đã có."

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        del user_id  # tra cứu công khai — nhưng `repo` thì **dùng**, đừng xoá
        query = str(arguments.get("query", "")).strip()
        shown = str(arguments.get("display") or query).strip()  # C-02: echo bản gốc có dấu
        if not query:
            return ToolResult(
                ok=False,
                reason=TOOL_BAD_ARGS,
                user_message="Chưa biết cần diễn giải thuốc nào.",
            )

        from app.retrieval import build_citations  # import cục bộ tránh vòng lặp

        # ⚠️ Không dùng `list_medicines(100, 0)`: chỉ thấy 100 thuốc đầu theo tên.
        # Chấm điểm trên câu GỐC: `query` đã bỏ từ chức năng ("dụng"), nên cụm hỏi
        # "tác dụng" không còn nhận ra được và "tác" khớp nhầm "tặc"/"tắc".
        question = shown or query
        hits = build_citations(repo.candidate_medicines(query), question, top_k=5)
        if not hits:
            from app import live_sources  # Hướng 2 (mặc định tắt)

            hits = build_citations(live_sources.lookup(query), question, top_k=5)
        if not hits:
            return ToolResult(
                ok=False,
                reason=TOOL_NO_DATA,
                user_message=(
                    f"Không tìm thấy thông tin trong nguồn hiện có về '{shown}'. "
                    "Hãy hỏi dược sĩ hoặc bác sĩ."
                ),
            )

        return ToolResult(
            ok=True,
            data={"query": query, "sources_found": len(hits)},
            citations=[h.citation for h in hits],
        )


class ScheduleReminderTool:
    """Ghi nhắc uống thuốc vào danh sách của **chính người gọi**.

    ⚠️ `user_id` lấy từ token, **không** từ `arguments`. Nếu lấy từ tham số thì
    người dùng gửi `{"user_id": 1}` là ghi vào danh sách của người khác → IDOR.
    """

    name = "schedule_reminder"
    description = "Ghi nhắc uống thuốc vào danh sách của người đang đăng nhập."

    def run(
        self, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        if user_id <= 0:
            return ToolResult(ok=False, reason=TOOL_BAD_ARGS, user_message="Chưa xác thực.")

        medicine = str(arguments.get("medicine", "")).strip()
        if not medicine:
            return ToolResult(
                ok=False,
                reason=TOOL_BAD_ARGS,
                user_message="Chưa cho biết nhắc uống thuốc nào.",
            )

        store = getattr(repo, "add_reminder", None)
        if store is None:  # kho chưa hỗ trợ -> nói thật, không giả vờ đã lưu
            return ToolResult(
                ok=False,
                reason=TOOL_NO_DATA,
                user_message="Tính năng nhắc uống thuốc chưa được bật.",
            )

        item_id = store(user_id, medicine)
        return ToolResult(
            ok=True,
            data={"reminder_id": item_id, "medicine": medicine},
            user_message=f"Đã ghi nhắc uống {medicine}.",
        )


class ToolRegistry:
    """Điểm vào **duy nhất** của orchestrator tới các tool.

    Day 28 thay MCP chỉ cần đăng ký thêm tool vào đây (xem [register]); orchestrator
    và route **không đổi một dòng**.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Đăng ký tool. Ghi đè nếu trùng tên — có kiểm soát, không im lặng."""
        if not getattr(tool, "name", ""):
            raise ValueError("Tool phải có `name`.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        """Danh sách tên tool — đưa vào prompt để planner biết có thể gọi gì."""
        return sorted(self._tools)

    def describe(self) -> str:
        """Mô tả tool cho planner, **không** kèm nội dung dữ liệu."""
        return "\n".join(
            f"- {name}: {self._tools[name].description}" for name in self.names()
        )

    def run(
        self, name: str, arguments: dict[str, Any], repo: SQLiteRepository, user_id: int
    ) -> ToolResult:
        """Chạy tool theo tên, **không** để exception lọt ra ngoài.

        ⚠️ Bọc `except Exception` ở đây là cố ý: orchestrator gọi tool, một tool lỗi
        (kho hỏng, dữ liệu lạ) không được làm sập cả lượt trả lời — phải hoá thành
        `ToolResult(ok=False)` để agent còn trả lời được "tôi không kiểm tra được".
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                reason=TOOL_BAD_ARGS,
                user_message=f"Không có công cụ '{name}'.",
            )
        try:
            result = tool.run(arguments, repo, user_id)
        except Exception:
            # ⚠️ KHÔNG log chi tiết lỗi — có thể chứa dữ liệu người dùng (AGENTS.md mục 3).
            return ToolResult(
                ok=False,
                reason=TOOL_REFUSED,
                user_message="Công cụ tra cứu gặp sự cố, tôi không kiểm tra được.",
            )
        return result


def default_registry() -> ToolRegistry:
    """Registry đủ 5 tool ROADMAP yêu cầu."""
    registry = ToolRegistry()
    for tool in (
        SearchDrugTool(),
        CheckInteractionTool(),
        NormalizePrescriptionTool(),
        ExplainForPatientTool(),
        ScheduleReminderTool(),
    ):
        registry.register(tool)
    return registry

