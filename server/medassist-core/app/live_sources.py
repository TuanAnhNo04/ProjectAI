"""Truy hồi trực tiếp từ nguồn y khoa ĐÃ DUYỆT khi kho cục bộ không có thuốc (Hướng 2).

⚠️ MẶC ĐỊNH TẮT: chưa cấu hình `MEDASSIST_LIVE_SOURCES` thì không bao giờ gọi mạng.
Chỉ chủ dự án thêm nguồn (AGENTS.md mục 2a) — mỗi mục phải khớp một dòng trong
`docs/TRUSTED_SOURCES.md`. Agent/AI không tự thêm nguồn.

Ràng buộc an toàn (kiểm bằng test):
- Chỉ HTTPS GET, chỉ đọc; không cookie, không header định danh, không body.
- Chỉ **tên thuốc đã chuẩn hoá** (a-z, 0-9, khoảng trắng, `.`/`-`; 2–40 ký tự) được đưa vào URL.
  Câu hỏi, hồ sơ, thuốc đang dùng, mã người dùng KHÔNG BAO GIỜ rời server.
- Host cố định theo mẫu URL đã duyệt; không theo chuyển hướng (redirect = bỏ).
- Giới hạn thời gian, dung lượng, tần suất; có bộ nhớ đệm (kể cả kết quả rỗng).
- Nội dung tải về là DỮ LIỆU: bỏ ký tự điều khiển, cắt độ dài, rồi đi qua đúng bộ
  truy hồi + kiểm tra bám nguồn như dữ liệu cục bộ.
- Mọi bản ghi gắn nguồn/ngày lấy/giấy phép/người duyệt; `is_authoritative` chỉ khi có `reviewed_by`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import date
from urllib.parse import quote, urlsplit

from app.schemas import MedicineResponse

SOURCES_ENV = "MEDASSIST_LIVE_SOURCES"

_FETCH_TIMEOUT = 5.0
_MAX_BYTES = 256 * 1024
_MAX_FIELD_CHARS = 800
_CACHE_TTL = 24 * 3600
_NEG_TTL = 3600
_CACHE_MAX = 256
_MAX_PER_MINUTE = 20
_FIELDS = ("composition", "uses", "side_effects")
_DRUG_RE = re.compile(r"^[a-z0-9][a-z0-9 .\-]{1,39}$")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

_log = logging.getLogger("medassist")


@dataclass(frozen=True)
class LiveSource:
    name: str
    url_template: str
    host: str
    fields: dict[str, str]
    owner: str
    license: str
    reviewed_by: str | None


_cache: dict[tuple[str, str], tuple[float, list[MedicineResponse]]] = {}
_calls: deque[float] = deque()


def reset_cache() -> None:
    _cache.clear()
    _calls.clear()


def normalize_drug(name: str | None) -> str | None:
    """Tên thuốc an toàn để đưa vào URL, hoặc None nếu không đạt."""
    if not name:
        return None
    text = unicodedata.normalize("NFKD", name.replace("đ", "d").replace("Đ", "D"))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text if _DRUG_RE.match(text) else None


def configured_sources() -> list[LiveSource]:
    """Đọc cấu hình; mục thiếu thông tin duyệt (giấy phép, chủ sở hữu, ngày xác nhận) bị bỏ."""
    raw = os.environ.get(SOURCES_ENV)
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except ValueError:
        return []
    out: list[LiveSource] = []
    for it in items if isinstance(items, list) else []:
        src = _parse_entry(it)
        if src:
            out.append(src)
    return out


def _parse_entry(it: object) -> LiveSource | None:
    if not isinstance(it, dict):
        return None
    tpl = str(it.get("url_template") or "").strip()
    if tpl.count("{drug}") != 1 or len(tpl) > 1024:
        return None
    probe = tpl.replace("{drug}", "x")
    try:
        parts = urlsplit(probe)
        port = parts.port
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if (
        parts.scheme != "https" or not host or port not in (None, 443)
        or parts.username is not None or parts.password is not None
        or "{drug}" in tpl.split("://", 1)[-1].split("/", 1)[0]  # không cho tên thuốc vào host
    ):
        return None
    name = str(it.get("name") or "").strip()
    owner = str(it.get("owner") or "").strip()
    lic = str(it.get("license") or "").strip()
    approved_on = str(it.get("approved_on") or "").strip()
    if not (name and owner and lic and approved_on):
        return None
    fields = {
        k: str(v) for k, v in (it.get("fields") or {}).items()
        if k in _FIELDS and isinstance(v, str) and v
    }
    if not fields:
        return None
    reviewed = str(it.get("reviewed_by") or "").strip() or None
    return LiveSource(name, tpl, host, fields, owner, lic, reviewed)


def enabled() -> bool:
    return bool(configured_sources())


def _rate_ok() -> bool:
    now = time.monotonic()
    while _calls and now - _calls[0] > 60:
        _calls.popleft()
    if len(_calls) >= _MAX_PER_MINUTE:
        return False
    _calls.append(now)
    return True


def _http_get(url: str) -> tuple[int, str, bytes]:
    """(status, content-type, body). Tách riêng để test thay thế, không gọi mạng thật."""
    import httpx

    with httpx.Client(
        timeout=_FETCH_TIMEOUT, follow_redirects=False, trust_env=False
    ) as c:
        r = c.get(url, headers={"User-Agent": "MedAssist/1.0 (do an hoc tap)"})
        return r.status_code, r.headers.get("content-type", ""), r.content


def _dig(data: object, path: str) -> str | None:
    cur = data
    for key in path.split("."):
        if isinstance(cur, list):
            cur = cur[0] if cur else None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    if isinstance(cur, list):
        cur = "; ".join(str(x) for x in cur[:10])
    if cur is None or isinstance(cur, (dict, bool)):
        return None
    text = _CTRL_RE.sub(" ", str(cur))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_FIELD_CHARS] or None


def _fetch_one(src: LiveSource, drug: str) -> MedicineResponse | None:
    url = src.url_template.replace("{drug}", quote(drug, safe=""))
    if urlsplit(url).hostname != src.host:  # phòng thủ: host không được đổi
        return None
    try:
        status, ctype, body = _http_get(url)
    except Exception:
        # ⚠️ Cố ý nuốt chi tiết: thông điệp lỗi có thể chứa URL.
        return None
    if status != 200 or "json" not in ctype.lower() or len(body) > _MAX_BYTES:
        return None
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return None
    values = {k: _dig(data, p) for k, p in src.fields.items()}
    if not any(values.values()):
        return None
    return MedicineResponse(
        name=drug.title(),
        composition=values.get("composition"),
        uses=values.get("uses"),
        side_effects=values.get("side_effects"),
        source=f"{src.name} (truy hồi trực tiếp)",
        is_authoritative=bool(src.reviewed_by),
        source_url=url,
        retrieved_at=date.today().isoformat(),
        license=src.license,
        reviewed_by=src.reviewed_by,
    )


def lookup(drug_name: str | None) -> list[MedicineResponse]:
    """Tra một tên thuốc ở các nguồn đã duyệt. Tắt/lỗi/không có -> danh sách rỗng."""
    sources = configured_sources()
    drug = normalize_drug(drug_name)
    if not sources or not drug:
        return []
    out: list[MedicineResponse] = []
    for src in sources:
        key = (src.name, drug)
        hit = _cache.get(key)
        if hit and time.monotonic() < hit[0]:
            out.extend(hit[1])
            continue
        if not _rate_ok():
            _log.warning("live source rate limited")
            continue
        med = _fetch_one(src, drug)
        found = [med] if med else []
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = (time.monotonic() + (_CACHE_TTL if found else _NEG_TTL), found)
        out.extend(found)
    return out
