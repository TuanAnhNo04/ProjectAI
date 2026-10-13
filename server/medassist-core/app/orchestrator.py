"""Orchestrator: planner -> gọi tool -> trả lời **có căn cứ** (Day 22).

## Nguyên tắc: LLM **không** quyết có nên trả lời

```
câu hỏi + lịch sử hội thoại
  -> planner chọn tool          (mock, luật xác định — xem [plan])
  -> registry.run(tool, args)   (app/tools.py)
  -> ghép câu trả lời TỪ KẾT QUẢ TOOL
  -> kiểm chứng grounding       (app/agent.py)
  -> phát sự kiện SSE
```

Planner ở đây là **luật xác định**, không phải LLM. Vì sao:

- `MockLLMProvider` không có khả năng "gọi tool" (nó chỉ trích nguyên văn);
- nếu để LLM tự chọn tool, ta đã dựng lại đúng lỗ hổng mà Day 21 đã chặn.

Khi có LLM thật, chỗ cần đổi là hàm [plan] — phần còn lại giữ nguyên.

## ⚠️ Nếu tool không có dữ liệu -> agent **nói ra điều đó**

Không có cách nào quyết định "không đủ căn cứ" mà không so với kết quả tool. Vì vậy
mọi câu trả lời đều dựng từ `ToolResult`; không có tool nào trả `ok=True` với nội
dung bịa thì agent không có gì để nói.

Đây là điểm khác biệt quan trọng với cách làm phổ biến ("LLM tự quyết"): ở đó, khi
thiếu dữ liệu, LLM **bịa** — vì không có gì buộc nó phải dừng lại.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.agent import REFUSAL_OUT_OF_SCOPE, _OUT_OF_SCOPE_RE
from app.llm import LLMProvider
from app.prompt import looks_like_injection as _looks_like_injection
from app.schemas import CitationResponse
from app.sqlite_repository import SQLiteRepository
from app.tools import (
    TOOL_BAD_ARGS,
    TOOL_NO_DATA,
    ToolRegistry,
    ToolResult,
    default_registry,
)
from app.textnorm import content_tokens, normalize

#: Số lượt tối đa giữ trong một phiên hội thoại. Bỏ giới hạn thì một phiên dài sẽ
#: làm prompt phình vô hạn — chi phí và rủi ro tăng theo lịch sử.
MAX_HISTORY_TURNS = 10

#: Ngưỡng cần có từ khoá thật để coi là "tra cứu thuốc". Không có thì coi như câu
#: hỏi mơ hồ -> không gọi tool, hỏi lại cho rõ.
MIN_DRUG_TOKENS = 1

if TYPE_CHECKING:  # pragma: no cover - chỉ để trình biết kiểu cho type checker
    from app.sessions import SessionStore


@dataclass
class ChatTurn:
    """Một lượt hội thoại đã lưu — **thuộc `user_id`**, không chia sẻ giữa người."""

    user_id: int
    session_id: str
    role: str
    content: str
    created_at: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass
class AgentStep:
    """Một bước của orchestrator — sinh ra sự kiện SSE tương ứng."""

    kind: str  # "tool" | "citation" | "token" | "done" | "refusal"
    text: str = ""
    tool_name: str | None = None
    citation: CitationResponse | None = None
    reason: str | None = None


@dataclass
class ChatResult:
    """Kết quả một lượt chat."""

    steps: list[AgentStep] = field(default_factory=list)
    answered: bool = False
    refusal_reason: str | None = None
    tool_called: str | None = None


# --------------------------------------------------------------- planner

#: ⭐ **Mọi mẫu khớp đều viết trên chuỗi ĐÃ CHUẨN HOÁ** (không dấu, chữ thường).
#:
#: Vì sao không viết regex có dấu: rất dễ sai mà rất khó thấy. Đo thực tế —
#: `t[uươ]ng` trông đúng nhưng chỉ khớp **một** ký tự giữa, trong khi "tương" có
#: **hai** (`ư`, `ơ`), nên "tương tác" **không** khớp. Lỗi kiểu này im lặng: câu hỏi
#: nguy hiểm nhất rơi nhầm sang nhánh khác, còn test kiểu "chức năng chạy" vẫn xanh.
#:
#: `normalize()` rồi khớp ASCII là cách ít chỗ sai nhất — cùng nguyên tắc với
#: `app/textnorm.py`: chuẩn hoá **trước**, khớp **sau**.
_INTERACTION_PATTERNS = (
    "tuong tac",
    "uong chung",
    "cung voi",
    "tranh chap",
    "hoat dong",
    "drug interaction",
    "khong nhau",
    "duoc khong",
    "hop le",
    "an toan",
)

#: "thêm" là dấu hiệu đi kèm, **không đủ** để kết luận là hỏi tương tác
#: ("thêm thông tin cho tôi" không phải).
_INTERACTION_SIGNALS = ("them ", "bo sung", "them vao")

_PRESCRIPTION_PATTERNS = ("don thuoc", "chuan hoa don", "chuan hoa chu", "toa don")

_REMINDER_PATTERNS = (
    "nhac toi",
    "nhac uong",
    "nhac dung",
    "dat nhac",
    "set reminder",
    "remind me",
    "schedule_reminder",
    "nhac thuoc",
)


def is_interaction_question(text: str) -> bool:
    """Câu này có hỏi về **tương tác/tương khắc** thuốc không?

    ⚠️ Đây là phân loại **nguy hiểm nhất**: sai ở đây thì người dùng nhận câu trả
    lời không dành cho câu hỏi đó. Vì vậy thiên về nhận thành công hơn bỏ sót.
    """
    flat = normalize(text)
    if any(p in flat for p in _INTERACTION_PATTERNS):
        return True
    return any(s in flat for s in _INTERACTION_SIGNALS)


def is_prescription_question(text: str) -> bool:
    """Câu này có liên quan tới **đơn thuốc** không?"""
    flat = normalize(text)
    return any(p in flat for p in _PRESCRIPTION_PATTERNS)


def is_reminder_request(text: str) -> bool:
    """Người dùng có **yêu cầu đặt nhắc** không?"""
    flat = normalize(text)
    return any(p in flat for p in _REMINDER_PATTERNS)


def _extract_drugs(text: str, repo: SQLiteRepository, limit: int = 4) -> list[str]:
    """Tìm tên thuốc có **thật trong kho** trong câu nói.

    ⚠️ Chỉ trả về tên **đã kiểm chứng là tồn tại**. Nếu lấy ngẫu nhiên các từ làm
    "tên thuốc" thì `check_interaction` sẽ nhận tham số bịa — và nếu sau này tool
    này có dữ liệu thật, nó sẽ kiểm tra sai cặp thuốc. Tên sai còn nguy hiểm hơn
    là không kiểm tra.
    """
    found: list[str] = []
    seen: set[str] = set()
    for word in content_tokens(text):
        if word in seen:
            continue
        # `search_medicines` đã chuẩn hoá dấu nên gõ không dấu vẫn khớp.
        hits = repo.search_medicines(word, 1, 0).results
        if hits:
            name = hits[0].name
            if name.lower() not in seen:
                seen.add(name.lower())
                found.append(name)
        if len(found) >= limit:
            break
    return found


def plan(message: str, repo: SQLiteRepository) -> tuple[str | None, dict[str, Any]]:
    """Chọn tool + tham số cho một lượt chat.

    ⚠️ Trả `(None, {})` nghĩa là **không gọi tool nào**. Trường hợp này agent sẽ hỏi
    lại cho rõ chứ không đoán — đoán tool sai tốn kém hơn hỏi lại.

    Vì sao không để LLM chọn: planner là nơi quyết định hệ thống có đi vào dữ liệu y
    tế nào. Để mô hình tự chọn là trao quyền đó cho thứ không được kiểm chứng. Luật
    ở đây có thể đọc, kiểm thử và tranh luận — mô hình thì không.
    """
    text = message.strip()

    # 1. Tương tác thuốc — kiểm tra TRƯỚC vì nguy hiểm nhất.
    if is_interaction_question(text):
        return "check_interaction", {"drugs": _extract_drugs(text, repo)}

    # 2. Nhắc uống thuốc.
    if is_reminder_request(text):
        drugs = _extract_drugs(text, repo)
        return "schedule_reminder", {"medicine": drugs[0] if drugs else ""}

    # 3. Đơn thuốc.
    if is_prescription_question(text):
        return "normalize_prescription", {"text": text}

    # 4. Tra cứu / giải thích thuốc: cần ít nhất một từ khoá trong kho.
    tokens = content_tokens(text)
    if tokens:
        return "explain_for_patient", {"query": " ".join(tokens)}

    return None, {}


# --------------------------------------------------------------- trả lời


def _compose(tool_name: str, result: ToolResult) -> tuple[str, list[CitationResponse]]:
    """Dựng câu trả lời **chỉ từ** kết quả tool.

    ⚠️ Không có mệnh đề nào ở đây được tự thêm. Mọi câu đều là `user_message` của
    tool hoặc nội dung citation đã kiểm chứng.
    """
    if not result.ok:
        return (
            result.user_message
            or "Tôi không kiểm tra được thông tin này. Hãy hỏi dược sĩ hoặc bác sĩ.",
            [],
        )

    lines: list[str] = []
    if result.user_message:
        lines.append(result.user_message)

    # Tool có citation -> liệt kê đoạn trích. Đây là nơi ROADMAP yêu cầu "trả lời
    # có căn cứ": người dùng đọc được ngay đoạn nào nói gì.
    for cite in result.citations:
        if not cite.excerpt:
            continue
        lines.append(
            f"- {cite.medicine_name} ({cite.field}): {cite.excerpt} [{cite.id}]"
        )

    if not lines:
        # Tool trả ok=True nhưng rỗng -> coi như không có căn cứ. Không đoán bù.
        return (
            "Tôi không tìm được thông tin đủ tin cậy để trả lời. "
            "Hãy hỏi dược sĩ hoặc bác sĩ.",
            [],
        )
    return "\n".join(lines), result.citations


def _abort(reason: str, text: str = "") -> ChatResult:
    """Kết thúc lượt ngay ở cổng đầu (chủ đề ngoài phạm vi / injection / rỗng).

    ⚠️ ⚠️ **Luôn kèm bước `done`.** Hợp đồng ở `lib/domain/ports/agent_client.dart`
    ghi rõ: *"Sự kiện luôn kết thúc bằng `Done` hoặc `Failure`"*. Thiếu `done` thì
    màn hình chat treo vĩnh viễn với con trỏ nhấp nháy — và test "tool chạy được"
    sẽ **không** bắt được, vì lỗi nằm ở luồng sự kiện chứ không phải logic tool.
    """
    steps = [AgentStep(kind="refusal", reason=reason, text=text)]
    if text:
        steps.append(AgentStep(kind="token", text=text))
    steps.append(AgentStep(kind="done"))
    return ChatResult(steps=steps, answered=False, refusal_reason=reason)


def chat(
    message: str,
    user_id: int,
    repo: SQLiteRepository,
    session: "SessionStore | None" = None,
    session_id: str = "",
    registry: ToolRegistry | None = None,
    llm: LLMProvider | None = None,
) -> ChatResult:
    """Xử lý một lượt chat và trả về **các bước** để phát SSE.

    Args:
        message: tin nhắn của người dùng.
        user_id: lấy từ **token** — không bao giờ từ body (nếu không thì IDOR).
        repo: kho dữ liệu.
        session: nơi lưu lịch sử (xem `app/sessions.py`).
        session_id: khoá phiên. ⚠️ **Tách khỏi `session`** vì `session_id` là dữ liệu
            của *lượt chat này*, không phải thuộc tính của kho — trộn hai thứ khiến
            kho phải biết "phiên hiện tại", mà kho thì phục vụ **mọi** người dùng.
        registry: bộ tool; mặc định lấy registry 5 tool.
        llm: provider; hiện vẫn **mock**, không dùng để quyết định gì.

    ⚠️ Hàm này **không** ghi log nội dung tin nhắn/ngữ cảnh (`AGENTS.md` mục 3, D4).
    """
    tools = registry or default_registry()
    text = message.strip()

    # ---- 0. Lớp phòng thủ sớm nhất: chủ đề ngoài phạm vi + prompt injection.
    if not text:
        return _abort("empty_message", "Bạn chưa gõ nội dung. Hãy hỏi về một loại thuốc cụ thể.")
    if _OUT_OF_SCOPE_RE.search(text) or _looks_like_injection(text):
        # ⚠️ Nội dung lượt này **không** đưa vào câu trả lời — nếu in lại chính câu
        # hỏi có dấu hiệu injection thì UI sẽ hiển thị lại payload đó cho người dùng.
        return _abort(REFUSAL_OUT_OF_SCOPE)

    # ---- 1. Planner chọn tool.
    tool_name, args = plan(text, repo)

    if tool_name is None:
        return _abort(
            "need_clarification",
            "Bạn muốn tra cứu thuốc nào, hay cần mình chuẩn hoá đơn thuốc, "
            "kiểm tra tương tác, hay đặt nhắc uống thuốc?",
        )

    # ---- 2. Chạy tool.
    result: ToolResult = tools.run(tool_name, args, repo, user_id)

    steps: list[AgentStep] = [
        AgentStep(kind="tool", tool_name=tool_name, text=_safe_args(tool_name, args))
    ]

    # ---- 3. Ghép câu trả lời TỪ KẾT QUẢ TOOL (không tự thêm gì).
    answer, citations = _compose(tool_name, result)

    for cite in citations:
        steps.append(AgentStep(kind="citation", citation=cite))

    if not result.ok:
        steps.append(AgentStep(kind="refusal", reason=result.reason or TOOL_NO_DATA))
        steps.append(AgentStep(kind="token", text=answer))
        steps.append(AgentStep(kind="done"))
        return ChatResult(
            steps=steps,
            answered=False,
            refusal_reason=result.reason or TOOL_NO_DATA,
            tool_called=tool_name,
        )

    # ⚠️ `ok=True` mà **không** có citation và không phải tool ghi dữ liệu -> không
    # có căn cứ để kiểm chứng -> từ chối thay vì trả lời trần.
    if not citations and tool_name not in _WRITE_ONLY_TOOLS:
        steps.append(AgentStep(kind="refusal", reason=TOOL_NO_DATA))
        steps.append(
            AgentStep(
                kind="token",
                text="Tôi không tìm được nguồn đủ tin cậy để trả lời. Hãy hỏi dược sĩ.",
            )
        )
        steps.append(AgentStep(kind="done"))
        return ChatResult(
            steps=steps,
            answered=False,
            refusal_reason=TOOL_NO_DATA,
            tool_called=tool_name,
        )

    # ---- 4. Chia nhỏ để UI hiện dần (khớp `TokenDelta` phía Dart).
    for chunk in _chunk_text(answer):
        steps.append(AgentStep(kind="token", text=chunk))
    steps.append(AgentStep(kind="done"))

    if session is not None and session_id:
        # ⚠️ Lưu **cả hai** chiều lượt. Chỉ lưu khi có trả lời — như vậy lượt bị
        # từ chối không đẩy lịch sử đi xa và giữ phiên gọn với câu hỏi thật.
        session.append(user_id, session_id, "user", text)
        session.append(user_id, session_id, "assistant", answer)

    return ChatResult(steps=steps, answered=True, tool_called=tool_name)


#: Tool ghi dữ liệu — không cần citation vì kết quả là hành động, không phải lời
#: khuyên y tế. ⚠️ Danh sách này phải **ngắn và có kiểm soát**: thêm tool vào đây là
#: đang nới điều kiện "phải có căn cứ". Mỗi mục cần lý do riêng.
_WRITE_ONLY_TOOLS = frozenset({"schedule_reminder"})


def _safe_args(tool_name: str, args: dict[str, Any]) -> str:
    """Mô tả tham số để gửi UI — **không** gửi nguyên văn.

    ⚠️ `ToolCallStarted.arguments` ở Dart ghi rõ "đã che (không gửi mật khẩu/token ra
    log)". `text` của `normalize_prescription` chứa **nội dung đơn thuốc** — dữ liệu
    sức khoẻ. Nên ở đây chỉ gửi *số ký tự*, không gửi nội dung.
    """
    if tool_name == "normalize_prescription":
        return f"({len(str(args.get('text', '')))} ký tự)"
    if tool_name == "check_interaction":
        drugs = args.get("drugs") or []
        return f"({len(drugs)} thuốc)"
    if tool_name == "schedule_reminder":
        return f"({len(str(args.get('medicine', '')))} ký tự)"
    return ""


def _chunk_text(text: str, size: int = 80) -> list[str]:
    """Chia văn bản thành các đoạn nhỏ cho `TokenDelta`.

    Chia theo ranh giới từ để không cắt giữa chữ.
    """
    words = text.split()
    if not words:
        return [text] if text else []
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if current and length + len(word) + 1 > size:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

