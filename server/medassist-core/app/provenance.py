"""Nguồn gốc (provenance) của trích dẫn: kiểm tra an toàn đường dẫn nguồn.

Chỉ phát `source_url` khi: https, không userinfo, cổng vắng/443, và host nằm trong
danh sách `MEDASSIST_SOURCE_ALLOWED_DOMAINS` (mặc định RỖNG -> không bao giờ phát link
cho tới khi chủ dự án cấu hình). Khớp = host bằng mục hoặc là tên miền con của mục.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

ALLOWED_ENV = "MEDASSIST_SOURCE_ALLOWED_DOMAINS"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def allowed_source_domains() -> list[str]:
    raw = os.environ.get(ALLOWED_ENV) or ""
    return [d.strip().lower().lstrip(".") for d in raw.split(",") if d.strip()]


def safe_source_url(url: str | None) -> str:
    """Trả URL nếu an toàn, ngược lại chuỗi rỗng."""
    if not url or not isinstance(url, str):
        return ""
    url = url.strip()
    if not url or len(url) > 2048 or any(c.isspace() or ord(c) < 32 for c in url):
        return ""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return ""
    if parts.scheme != "https" or parts.username is not None or parts.password is not None:
        return ""
    if "@" in parts.netloc or port not in (None, 443):
        return ""
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return ""
    for dom in allowed_source_domains():
        if host == dom or host.endswith("." + dom):
            return url
    return ""


def clean_text(value: str | None, limit: int = 200) -> str | None:
    text = (value or "").strip()
    return text[:limit] or None


def clean_date(value: str | None) -> str | None:
    """Ngày ISO (YYYY-MM-DD...) hoặc None — không đoán/bịa."""
    text = (value or "").strip()
    return text[:32] if _DATE_RE.match(text) else None
