"""`/v1/health` — kiểm tra server còn sống.

Endpoint công khai (không cần token) để app Flutter ping được khi khởi động.
Chỉ trả trạng thái, **không** lộ phiên bản dependency hay cấu hình nội bộ.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])

SERVICE_NAME = "medassist-core"
SERVICE_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse, summary="Kiểm tra server")
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        time=datetime.now(timezone.utc),
    )