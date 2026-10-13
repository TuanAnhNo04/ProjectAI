"""Định dạng lỗi thống nhất cho toàn bộ API.

Mọi lỗi trả về đều có **cùng một hình dạng**, kể cả lỗi do FastAPI/Pydantic
sinh ra (422, 404 của route...). Nhờ vậy client chỉ cần đọc 3 trường:

    {"code": "validation_error", "message": "...", "detail": {...}}

`code` là **mã ổn định** để client xử lý logic (không so sánh `message`),
`message` là câu tiếng Anh ngắn để hiển thị/log, `detail` là thông tin kỹ thuật
tùy chọn.

⚠️ `detail` **KHÔNG BAO GIỜ** được chứa mật khẩu, token hay nội dung y tế —
xem `AGENTS.md` mục 3.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Mã lỗi ổn định (contract với client — đổi tên là phá vỡ client).
CODE_UNAUTHORIZED = "unauthorized"
CODE_FORBIDDEN = "forbidden"
CODE_NOT_FOUND = "not_found"
CODE_VALIDATION_ERROR = "validation_error"
CODE_CONFLICT = "conflict"
CODE_RATE_LIMITED = "rate_limited"
CODE_SERVER_ERROR = "server_error"


def error_body(
    code: str,
    message: str,
    detail: Any | None = None,
) -> dict[str, Any]:
    """Dựng thân lỗi theo format thống nhất."""
    return {"code": code, "message": message, "detail": detail}


class APIError(HTTPException):
    """Lỗi nghiệp vụ do server chủ động ném.

    Dùng thay cho `HTTPException` thường để **bắt buộc** đi qua
    `error_body` — không thể quên `code`.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        detail: Any | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code
        self.message = message
        self.extra_detail = detail


def register_error_handlers(app: Any) -> None:
    """Gắn handler để **mọi** lỗi đều ra đúng format."""

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.extra_detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # `exc.errors()` có thể chứa `input` (giá trị người dùng gửi). Bỏ nó đi
        # để không vô tình phản chiếu mật khẩu/email trong phản hồi.
        safe = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", "")}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=error_body(CODE_VALIDATION_ERROR, "Dữ liệu gửi lên không hợp lệ.", safe),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        # ⚠️ Bắt `StarletteHTTPException` (lớp cha), **không** phải
        # `fastapi.HTTPException` (lớp con). Route 404 của Starlette ném lớp cha,
        # nên handler đăng ký cho lớp con sẽ **không bao giờ** chạy — lúc đó mọi
        # lỗi 404 vẫn ra `{"detail": "Not Found"}` thiếu `code`.
        code = {
            401: CODE_UNAUTHORIZED,
            403: CODE_FORBIDDEN,
            404: CODE_NOT_FOUND,
        }.get(
            exc.status_code,
            CODE_SERVER_ERROR if exc.status_code >= 500 else CODE_VALIDATION_ERROR,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, str(exc.detail)),
        )