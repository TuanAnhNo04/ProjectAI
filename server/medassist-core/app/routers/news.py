"""Tin tức y tế — `GET /v1/news` (Day 31).

Công khai (không cần token): đây là tin báo chí, không phải dữ liệu người dùng.

⚠️ Endpoint này **không ghi log** câu truy vấn hay bài nào được đọc. Thói quen đọc tin y tế
cũng là dữ liệu sức khoẻ (`AGENTS.md` mục 3): biết một người đọc bài về bệnh gì là suy ra
được tình trạng của họ.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.news import MAX_ITEMS, get_news, verified_sources
from app.schemas import NewsArticleResponse, NewsResponse

router = APIRouter(tags=["news"])


@router.get(
    "/news",
    response_model=NewsResponse,
    summary="Tin tức thuốc và y tế (tiêu đề + trích đoạn + nguồn)",
)
def list_news(
    limit: int = Query(default=20, ge=1, le=MAX_ITEMS),
    refresh: bool = Query(default=False, description="Bỏ qua cache, tải lại từ nguồn"),
) -> NewsResponse:
    """Danh sách bài tin.

    - Chưa đặt `MEDASSIST_NEWS_FEEDS` -> trả **bài mẫu** kèm `is_sample=true`; server
      không gọi ra ngoài. App phải hiện rõ đây là nội dung mẫu.
    - Chỉ trả **tiêu đề, trích đoạn ngắn, nguồn, link** — không bao giờ toàn văn bài báo.
    - `failed_sources` cho biết nguồn nào vừa tải hỏng, để app nói thật với người dùng
      thay vì im lặng hiện thiếu tin.
    """
    articles, is_sample, failed = get_news(force=refresh)
    return NewsResponse(
        articles=[
            NewsArticleResponse(
                id=a.id,
                title=a.title,
                summary=a.summary,
                url=a.url,
                source=a.source,
                published_at=a.published_at,
                image_url=a.image_url,
            )
            for a in articles[:limit]
        ],
        total=len(articles),
        is_sample=is_sample,
        failed_sources=failed,
        verified_sources=verified_sources(),
        disclaimer=(
            "Tin tức do báo chí đăng tải, không phải tư vấn y khoa. "
            "Hãy hỏi dược sĩ hoặc bác sĩ trước khi áp dụng."
        ),
    )
