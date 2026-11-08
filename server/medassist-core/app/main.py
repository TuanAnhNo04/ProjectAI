"""Điểm khởi chạy FastAPI của `medassist-core`.

Chạy local:
```powershell
uvicorn app.main:app --reload --port 8000
```
Sau đó mở http://127.0.0.1:8000/docs để xem tài liệu tự sinh.

⚠️ **Chỉ phục vụ `localhost`.** Server này chứa dữ liệu sức khoẻ và tài khoản;
không bao giờ bind `0.0.0.0` trên máy dev (xem `AGENTS.md` mục 3).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from app.deps import get_repository
from app.errors import register_error_handlers
from app.jwt_tokens import validate_config
from app.observability import access_log, current_trace_id, new_trace_id
from app.privacy import apply_retention
from app.routers import agent, auth, data, health, news

# ⚠️ FAIL-FAST (quyết định N2): kiểm tra cấu hình NGAY khi module nạp.
# Nếu `MEDASSIST_SECRET_KEY` thiếu/ngắn thì server **không khởi động** thay vì
# chạy với khoá yếu rồi hỏng lúc có request đầu tiên.
validate_config()

# ⚠️ Access log mặc định của uvicorn ghi CẢ query string: `GET /v1/search?name=panadol`
# — tức là từ khoá thuốc người dùng tra (dữ liệu sức khoẻ) nằm trong log (quyết định D4).
# Ta đã có access log an toàn riêng (middleware `register_trace_middleware`: chỉ route
# template + trace_id), nên tắt cái của uvicorn dù server được chạy bằng cách nào.
logging.getLogger("uvicorn.access").disabled = True

API_PREFIX = "/v1"

DESCRIPTION = """
Backend cho ứng dụng MedAssist.

**Mọi lỗi trả về cùng một hình dạng**: `{"code", "message", "detail"}`.

⚠️ Thông tin thuốc chỉ mang tính tham khảo, không thay thế chẩn đoán hoặc đơn
thuốc của bác sĩ.
"""

@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Lúc khởi động: dọn lịch sử quá hạn theo `MEDASSIST_HISTORY_RETENTION_DAYS`."""
    repo = get_repository()
    apply_retention(repo)
    # Deploy (Railway): CSDL trên volume mới tạo thì rỗng -> nạp danh mục MẪU một lần.
    if os.environ.get("MEDASSIST_SEED_IF_EMPTY", "").strip() == "1":
        from app.seed import seed_if_empty

        seed_if_empty(repo)
    yield


_IS_PROD = os.environ.get("MEDASSIST_ENV", "").strip().lower() in {"production", "prod"}

app = FastAPI(
    title="MedAssist Core",
    version=health.SERVICE_VERSION,
    description=DESCRIPTION,
    lifespan=_lifespan,
    # Lịch sử thay đổi API: `/v1` nên client Flutter chỉ cần đổi `API_BASE_URL`.
    # B-04: production không phơi tài liệu API.
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
    openapi_url=None if _IS_PROD else "/openapi.json",
)

register_error_handlers(app)

app.include_router(health.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(data.router, prefix=API_PREFIX)
app.include_router(agent.router, prefix=API_PREFIX)
app.include_router(news.router, prefix=API_PREFIX)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    """Chỉ báo đường dẫn tài liệu."""
    return {"docs": "/docs", "openapi": "/openapi.json", "health": f"{API_PREFIX}/health"}


def register_trace_middleware(app: Any) -> None:
    """Gắn middleware sinh `trace_id` + ghi log truy cập (Day 24).

    ⚠️ Ba điều cố ý ở đây:

    1. **Không log URL thật.** Chỉ log *route template* — vì query string của
       `/v1/search?name=panadol` là dữ liệu thuốc người dùng đang tra (xem
       `observability.py`).
    2. **Không log thân request.** Body chứa mật khẩu, email, tin nhắn sức khoẻ.
    3. **Lỗi cũng phải có log.** Đường `finally` đảm bảo request lỗi vẫn được đo;
       không thì chính những request đáng ngờ nhất lại không có dấu vết.
    """

    @app.middleware("http")
    async def _trace(request: Any, call_next: Any) -> Any:
        trace_id = new_trace_id()
        token = current_trace_id.set(trace_id)
        started = time.perf_counter()
        status_code = 500
        error_code: str | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            # Trả `trace_id` cho client để khi người dùng báo lỗi thì có mã tra
            # được đúng request đó. Chỉ request **thành công** mới gắn.
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception:
            error_code = "unhandled_exception"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            access_log(
                trace_id=trace_id,
                route=request.url.path,
                latency_ms=elapsed_ms,
                status_code=status_code,
                error_code=error_code,
            )
            current_trace_id.reset(token)


# ⚠️ Gọi **sau** khi định nghĩa ở trên — Python phải thấy hàm trước khi gọi.
register_trace_middleware(app)


def export_openapi(target: str | Path = "openapi.json") -> Path:
    """Ghi đặc tả OpenAPI ra file để client sinh code hoặc review hợp đồng.

    Dùng: `python -m app.main` hoặc `python -c "from app.main import
    export_openapi; export_openapi()"`.
    """
    path = Path(target)
    path.write_text(
        json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    written = export_openapi()
    print(f"OpenAPI written to {written}")