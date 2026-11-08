"""`/v1/agent/ask` — hỏi đáp về thuốc (Day 21).

## Phân quyền

Bắt buộc token qua `Depends(current_user_id)`. Endpoint này **không** đọc/ghi dữ
liệu lưu trữ theo người dùng (chưa có memory chat), nhưng vẫn bắt buộc đăng nhập
vì: (1) câu hỏi là **dữ liệu sức khoẻ**, không nên cho người lạ gửi tới server
mỗ lúc; (2) sau này gắp memory/SSE thì đã có sẵn ranh giới phân quyền.

## Log

⚠️ **Không** ghi nội dung câu hỏi, tên thuốc hay nội dung trả lời vào log
(`AGENTS.md` mục 3, quyết định D4). Route này hiện **không** log gì cả — đó là hành
vi đúng, không phải thiếu sót. Khi cần log, dùng `app/logging_safe.py` (chỉ cho
phép `trace_id`/latency/mã lỗi/route).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agent import answer_question
from app.deps import current_user_id, get_repository
from app.errors import CODE_NOT_FOUND, APIError
from app.orchestrator import chat as chat_turn
from app.observability import audit, get_usage_tracker
from app.schemas import (
    AgentAnswerResponse,
    AgentAskRequest,
    AgentChatRequest,
    LifestyleRequest,
    LifestyleResponse,
    UsageResponse,
)
from app.lifestyle import build_lifestyle_tips
from app.sessions import get_session_store
from app.sqlite_repository import SQLiteRepository

router = APIRouter(tags=["agent"])

MAX_LIMIT = 100


@router.post(
    "/agent/ask",
    response_model=AgentAnswerResponse,
    summary="Hỏi đáp về thuốc (chỉ dùng nguồn nội bộ, có trích dẫn)",
)
def ask(
    body: AgentAskRequest,
    _user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> AgentAnswerResponse:
    """Trả lời câu hỏi về thuốc.

    ⚠️ Nguồn dữ liệu là danh mục thuốc **đã nạp trong kho**. Provider LLM mặc định
    là **mock, offline** (`app/llm.py`) — không có lệnh gửi dữ liệu ra ngoài.

    Câu hỏi liên quan chẩn đoán/liều dùng/tương tác thuốc sẽ bị từ chối
    (`refusal_reason="out_of_scope"`); câu hỏi không có nguồn sẽ bị từ chối
    (`"no_evidence"`). App phải hiển thị `disclaimer` và tôn trọng `answered=False`.
    """
    # ⚠️ Không dùng `list_medicines(100, 0)`: chỉ thấy 100 thuốc đầu, vô dụng với danh mục thật.
    medicines = repo.candidate_medicines(body.question)
    # Endpoint này không có cờ đồng ý -> luôn mock (không bao giờ ra Claude, B-01).
    from app.llm import _default_provider

    return answer_question(
        body.question, medicines, top_k=body.top_k, provider=_default_provider
    )



@router.get(
    "/agent/usage",
    response_model=UsageResponse,
    summary="Số liệu sử dụng của chính người đang đăng nhập",
)
def my_usage(user_id: int = Depends(current_user_id)) -> UsageResponse:
    """Xem số liệu sử dụng **của mình**.

    ⚠️ ROADMAP Day 24 ghi "log chi phí token theo user". Ta **cố ý không** ghi vào
    log: log ở nơi nhiều người đọc và tồn tại lâu; gắn `user_id` vào đó là tạo bảng
    liên kết vĩnh viễn giữa danh tính và hành vi tìm thuốc (`AGENTS.md` mục 1).
    Số liệu nằm trong RAM và chỉ trả cho **chính người đó** qua đây.

    Với provider **mock** nên `cost_usd` luôn bằng 0 — ta đo độ dài để sẵn sàng
    khi đổi sang provider thật.
    """
    return UsageResponse(**get_usage_tracker().get(user_id))

# ============================ Hội thoại SSE (Day 22) =======================
#
# ⚠️ **Tên sự kiện SSE khớp 1-1 với `lib/domain/ports/agent_client.dart`.** Đổi tên
# ở đây mà không sửa Dart thì app rơi im lặng — mà "rơi im lặng" là kiểu hỏng tệ
# nhất: không có lỗi nào để bắt.
#
#   TokenDelta      -> token
#   ToolCallStarted -> tool_call
#   Citation        -> citation
#   Done            -> done
#   Failure         -> failure
#   (mới) Emergency -> emergency  {"matched": [...], "message": "..."}


def _sse(event: str, data: dict[str, Any]) -> str:
    """Dựng một khung SSE.

    ⚠️ Văn bản có thể chứa dấu xuống dòng (câu hỏi/đáp) — mà SSE coi `\n\n` là **kết
    thúc event**. `json.dumps` tự escape newline thành `\n` bên trong chuỗi JSON
    nên ở đây an toàn. Đừng "tối ưu" bằng cách ghép tay.
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _event_error(reason: str, message: str) -> str:
    """Sự kiện `failure` — luôn **kết thúc** luồng.

    ⚠️ Vì sao lỗi phải là *event* chứ không phải exception: Dart đã ghi rõ — lỗi
    giữa luồng phải tới dưới dạng `Failure`, nếu stream văng exception thì UI mất
    **toàn bộ** tin nhắn đã nhận.
    """
    return _sse("failure", {"kind": "server_error", "message": message, "reason": reason})


@router.post(
    "/agent/chat",
    summary="Hội thoại với agent (SSE: token + tool + citation)",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
def chat_stream(
    body: AgentChatRequest,
    _user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> StreamingResponse:
    """Mở luồng trả lời cho một lượt chat.

    ⚠️ Phân quyền: `user_id` lấy từ **token**; `session_id` từ body nhưng **không**
    được tin là của người gọi — `SessionStore` lọc theo cặp `(user_id, session_id)`.

    ⚠️ Không log nội dung tin nhắn/ngữ cảnh (`AGENTS.md` mục 3, D4).
    """
    user_id = _user_id

    def _generate() -> Iterator[str]:
        output_chars = 0
        try:
            result = chat_turn(
                body.message,
                user_id=user_id,
                repo=repo,
                session=get_session_store(),
                session_id=body.session_id,
                current_medications=body.current_medications,
                cloud_ai_consent=body.cloud_ai_consent,
                health_context=body.health_context,
            )
        except Exception:
            # ⚠️ Không log chi tiết — có thể chứa dữ liệu sức khoẻ của người dùng.
            yield _event_error("internal", "Lỗi hệ thống khi xử lý yêu cầu.")
            return

        for step in result.steps:
            if step.kind == "tool" and step.tool_name:
                yield _sse("tool_call", {"name": step.tool_name, "arguments": step.text})
            elif step.kind == "citation" and step.citation:
                c = step.citation
                yield _sse(
                    "citation",
                    {
                        "documentId": c.id,
                        "quote": c.excerpt,
                        "title": c.medicine_name,
                        "source": c.source,
                        "isAuthoritative": c.is_authoritative,
                        "sourceUrl": c.source_url,
                        "retrievedAt": c.retrieved_at,
                        "license": c.license,
                        "reviewedBy": c.reviewed_by,
                    },
                )
            elif step.kind == "token":
                yield _sse("token", {"text": step.text})
                output_chars += len(step.text)
            elif step.kind == "refusal":
                # Sự kiện riêng: UI hiện cảnh báo ngay, không phải chờ đọc hết text.
                yield _sse("refusal", {"reason": step.reason or ""})
            elif step.kind == "emergency" and step.data:
                # Sự kiện riêng: UI hiện banner khẩn cấp + nút gọi 115 ngay lập tức.
                yield _sse("emergency", step.data)
            elif step.kind == "symptom" and step.data:
                yield _sse("symptom_noted", step.data)
            elif step.kind == "done":
                yield _sse("done", {"sessionId": body.session_id, "mode": result.mode})

        # ⚠️ Đếm **độ dài ký tự**, không đếm số lượt tool/citation: đây là thứ duy
        # nhất tương đương chi phí khi đổi provider. Không ghi nội dung vào đâu.
        get_usage_tracker().record(
            user_id, "/v1/agent/chat", len(body.message), output_chars
        )

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            # Proxy có thể giữ bộ đệm -> client không thấy token cho tới khi hết.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete(
    "/agent/session/{session_id}",
    status_code=204,
    summary="Xoá lịch sử hội thoại của người đang đăng nhập",
)
def delete_session(
    session_id: str,
    user_id: int = Depends(current_user_id),
) -> None:
    """Xoá phiên **thuộc về mình**.

    ⚠️ `session_id` tồn tại nhưng thuộc người khác thì cũng trả 404 — đúng như
    "không tồn tại", để không lộ sự tồn tại của dữ liệu người khác.

    Đây là endpoint **bắt buộc** cho quyền được quên của người dùng (quyết định `D5`).
    """
    if not get_session_store().clear(user_id, session_id):
        raise APIError(status_code=404, code=CODE_NOT_FOUND, message="Không có phiên nào.")
    audit("session_deleted", "ok")


@router.post(
    "/agent/lifestyle",
    response_model=LifestyleResponse,
    summary="Gợi ý lối sống (stateless, có trích dẫn cho mục về thuốc)",
)
def lifestyle(
    body: LifestyleRequest,
    _user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> LifestyleResponse:
    """Tối đa 5 gợi ý. Mục `medication` chỉ lấy từ bản ghi thuốc (có trích dẫn); mục
    `general` từ danh sách viết sẵn, không có khẳng định về bệnh hay liều.

    ⚠️ **Stateless**: không ghi phiên, lịch sử, số liệu sử dụng hay log nội dung.
    """
    return build_lifestyle_tips(body, repo)
