"""`/v1/search`, `/v1/analyze`, `/v1/history` — dữ liệu nghiệp vụ.

## Phân quyền (điểm quan trọng nhất của route này)
Mọi endpoint đọc/ghi dữ liệu người dùng đều:
1. bắt buộc token qua `Depends(current_user_id)`;
2. truyền `user_id` **đã lấy từ token** xuống kho — **không** nhận `user_id`
   từ query/body (nếu nhận thì người dùng tự khai id người khác → IDOR).

Vì vậy không endpoint nào cho phép đọc lịch sử của người khác, kể cả khi biết
đúng `id` của bản ghi.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.deps import current_user_id, get_repository
from app.errors import CODE_NOT_FOUND, APIError
from app.schemas import (
    AnalyzeResponse,
    HistoryCreateRequest,
    HistoryResponse,
    SearchResponse,
)
from app.sqlite_repository import SQLiteRepository

router = APIRouter(tags=["data"])

#: Trần `limit` để 1 request không kéo cả bảng về (rủi ro DoS).
MAX_LIMIT = 100


def _page(limit: int, offset: int) -> tuple[int, int]:
    return min(max(limit, 1), MAX_LIMIT), max(offset, 0)


# --- Tìm kiếm (công khai) ---------------------------------------------------


@router.get("/search", response_model=SearchResponse, summary="Tìm thuốc theo tên")
def search(
    name: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=20, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    repo: SQLiteRepository = Depends(get_repository),
) -> SearchResponse:
    """Tìm theo tên thuốc; **không** bất buộc đăng nhập.

    Tên được chuẩn hoá (bỏ dấu, hạ chữ thường) nên gõ "Panadol", "panadol" hay
    không dấu đều ra kết quả như nhau.
    """
    limit, offset = _page(limit, offset)
    return repo.search_medicines(name, limit, offset)


# --- Phân tích ảnh (công khai ở Day 15) -------------------------------------


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Nhận diện thuốc từ ảnh (placeholder Day 15)",
)
def analyze(
    repo: SQLiteRepository = Depends(get_repository),
    _user_id: int = Depends(current_user_id),
) -> AnalyzeResponse:
    """Placeholder — Day 15 chưa gắt mô hình nhận diện thật.

    ⚠️ Endpoint này **không** nhận/gửi ảnh đi nơi khác. Day 26 sẽ gắp
    nhận diện on-device phía app; nếu sau này gọi LLM/OCR bên ngoài thì phải
    có **sự đồng ý của người dùng + khử định danh + `docs/PRIVACY.md`** theo
    `AGENTS.md` mục 2.
    """
    raise APIError(
        status_code=501,
        code="not_implemented",
        message="Nhận diện ảnh chưa được bật. Hãy dùng tìm kiếm theo tên.",
    )


# --- Lịch sử (bắt buộc đăng nhập) ------------------------------------------


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Lịch sử của người đang đăng nhập",
)
def list_history(
    limit: int = Query(default=20, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> HistoryResponse:
    """Chỉ trả lịch sử **của chính người gọi**.

    `user_id` đến từ token; người dùng không có cách nào xem lịch sử người khác.
    """
    limit, offset = _page(limit, offset)
    items = repo.list_history(user_id, limit, offset)
    return HistoryResponse(history=items, total=repo.count_history(user_id))


@router.delete(
    "/history/{item_id}",
    status_code=204,
    summary="Xoá một bản ghi lịch sử của người đang đăng nhập",
)
def delete_history(
    item_id: int,
    user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> None:
    """Xoá bản ghi **thuộc về mình**.

    Nếu `item_id` tồn tại nhưng thuộc người khác thì cũng trả 404 — đúng như
    "không tồn tại", để không lộ sự tồn tại của dữ liệu người khác.
    """
    if not repo.delete_history(user_id, item_id):
        raise APIError(
            status_code=404,
            code=CODE_NOT_FOUND,
            message="Không tìm thấy bản ghi.",
        )


@router.post(
    "/history",
    response_model=HistoryResponse,
    status_code=201,
    summary="Ghi một bản ghi lịch sử (dùng khi app phân tích ảnh)",
)
def create_history(
    body: HistoryCreateRequest,
    user_id: int = Depends(current_user_id),
    repo: SQLiteRepository = Depends(get_repository),
) -> HistoryResponse:
    """Ghi bản ghi cho người đang đăng nhập và trả về lịch sử mới nhất.

    `filepath` là đường dẫn **trên server**; app không gửi ảnh lên ở Day 15.
    """
    repo.add_history(user_id, body.filepath, body.medicines)
    items = repo.list_history(user_id, 20, 0)
    return HistoryResponse(history=items, total=repo.count_history(user_id))