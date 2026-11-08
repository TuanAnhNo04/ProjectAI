"""Tin tức y tế — server đọc RSS, lọc, rồi trả cho app (Day 31).

## Vì sao server đọc chứ không để app đọc thẳng

1. **App không bao giờ gọi ra ngoài.** Mọi lưu lượng đi qua backend của ta, nên `AGENTS.md`
   mục 2 được giữ: thiết bị người dùng chỉ nói chuyện với một địa chỉ duy nhất.
2. **Gỡ nguồn được ngay.** Báo yêu cầu dừng -> sửa một biến môi trường, không cần phát hành
   lại app.
3. **Lọc tập trung.** Đo thật (28/10/2026): feed "chuyên mục y tế" của một báo thực chất trả
   về feed TRANG CHỦ — item đầu là tin hình sự và kết quả xổ số. Cắm thẳng vào app y tế là
   hiện tin đó cho người dùng. Lọc phải nằm ở một chỗ.

## Mặc định TẮT

Không đặt `MEDASSIST_NEWS_FEEDS` thì **không có request nào ra ngoài** và endpoint trả về
bộ bài mẫu kèm `is_sample=True`. Đây là trạng thái an toàn: demo luôn chạy, kể cả khi mất
mạng, và không ai vô tình bật tính năng gọi mạng.

## Điều khoản nguồn (đã kiểm chứng 28/10/2026)

- **FDA** (`fda.gov`): public domain, dùng tự do.
- **Báo Việt Nam** (VnExpress, Tuổi Trẻ, Dân trí): điều khoản RSS chỉ nói "miễn phí cho cá
  nhân và tổ chức **phi lợi nhuận**", bắt buộc ghi nguồn, và báo có quyền yêu cầu ngừng bất
  cứ lúc nào. **Không có câu nào cho phép rõ ràng ứng dụng bên thứ ba.** Dùng cho đồ án học
  tập thì hợp tinh thần đó; phát hành thương mại thì phải xin phép.
- **MedlinePlus**: RSS ghi rõ "personal use only" -> KHÔNG dùng trong app.

Vì vậy ta chỉ hiện **tiêu đề + trích đoạn ngắn + tên nguồn + link bài gốc**, không bao giờ
lưu hay hiện toàn văn.
"""

from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from xml.etree import ElementTree

#: Biến môi trường: danh sách feed, ngăn cách bằng dấu phẩy.
#: Dạng mỗi mục: `Tên nguồn|https://...` (thiếu tên thì lấy theo tên miền).
FEEDS_ENV = "MEDASSIST_NEWS_FEEDS"
#: Số giây giữ kết quả trong RAM trước khi tải lại (mặc định 15 phút).
TTL_ENV = "MEDASSIST_NEWS_TTL"
DEFAULT_TTL = 900

#: Độ dài tối đa của trích đoạn. Cố ý ngắn: ta **trích dẫn**, không đăng lại bài.
MAX_SUMMARY = 200
#: Số bài tối đa trả về.
MAX_ITEMS = 40
#: Giây chờ mỗi feed. Ngắn để một báo chậm không treo cả endpoint.
FETCH_TIMEOUT = 8.0

#: Chỉ cho phép http/https và **không** cho địa chỉ nội bộ — chặn SSRF: nếu không, ai đặt
#: được biến môi trường cũng khiến server tự gọi vào mạng nội bộ của chính nó.
_BLOCKED_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal", "169.254.169.254"}
)

#: Từ khoá loại bài KHÔNG phải y tế (đo thật trên feed "chuyên mục" bị trả nhầm trang chủ).
_OFF_TOPIC = re.compile(
    r"\b(xo so|vietlott|ket qua xo so|bong da|ty so|chuyen nhuong|giet nguoi|cuop|an mang|"
    r"tai nan giao thong|khoi to|bat tam giam|ma tuy|lottery|football)\b",
    re.IGNORECASE,
)
#: Dấu hiệu quảng cáo thuốc trá hình / bài PR.
_ADVERT = re.compile(
    r"(tai tro|bai pr|quang cao|dat mua|gia chi|cam ket khoi|hotline|lien he ngay|"
    r"nha thuoc .* phan phoi|mua ngay|uu dai|khuyen mai|sponsored)",
    re.IGNORECASE,
)
#: Link tiếp thị liên kết.
_AFFILIATE = re.compile(r"(utm_|affiliate|/ref=|clickid)", re.IGNORECASE)


def _strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt để lọc từ khoá không phụ thuộc dấu."""
    import unicodedata

    folded = text.replace("Đ", "D").replace("đ", "d")
    return "".join(
        c for c in unicodedata.normalize("NFD", folded) if not unicodedata.combining(c)
    )


@dataclass
class NewsArticle:
    """Một bài tin. Cố ý **không** có trường nội dung đầy đủ."""

    id: str
    title: str
    summary: str
    url: str
    source: str
    published_at: str | None = None
    image_url: str | None = None
    #: True = bài mẫu kèm theo app, không phải tin thật.
    is_sample: bool = False


@dataclass
class _Feed:
    name: str
    url: str


@dataclass
class _Cache:
    articles: list[NewsArticle] = field(default_factory=list)
    fetched_at: float = 0.0
    #: Feed nào lỗi lần gần nhất (chỉ tên nguồn, không có chi tiết lỗi).
    failed: list[str] = field(default_factory=list)


_cache = _Cache()


# --------------------------------------------------------------------------- bài mẫu

#: Bài mẫu để demo chạy được khi chưa bật nguồn thật hoặc mất mạng.
#: ⚠️ Đây là **nội dung tự soạn mang tính nhắc nhở chung**, không phải tin tức thật và
#: không chứa khẳng định y khoa cụ thể nào (liều dùng, hiệu quả điều trị).
SAMPLE_ARTICLES: list[NewsArticle] = [
    NewsArticle(
        id="sample-1",
        title="Đọc kỹ hướng dẫn trước khi dùng thuốc",
        summary=(
            "Tờ hướng dẫn đi kèm hộp thuốc có thành phần, chỉ định và chống chỉ định. "
            "Khi chưa rõ, hãy hỏi dược sĩ thay vì tự suy đoán."
        ),
        url="",
        source="MedAssist",
        is_sample=True,
    ),
    NewsArticle(
        id="sample-2",
        title="Giữ danh sách thuốc đang dùng",
        summary=(
            "Ghi lại các thuốc đang dùng giúp bác sĩ và dược sĩ phát hiện nguy cơ tương tác. "
            "Mục Hồ sơ sức khoẻ trong ứng dụng lưu danh sách này ngay trên máy của bạn."
        ),
        url="",
        source="MedAssist",
        is_sample=True,
    ),
    NewsArticle(
        id="sample-3",
        title="Không tự ý dùng lại đơn thuốc cũ",
        summary=(
            "Đơn thuốc được kê cho một lần khám và một tình trạng cụ thể. "
            "Triệu chứng giống nhau không có nghĩa nguyên nhân giống nhau."
        ),
        url="",
        source="MedAssist",
        is_sample=True,
    ),
    NewsArticle(
        id="sample-4",
        title="Bảo quản thuốc đúng cách",
        summary=(
            "Phần lớn thuốc cần nơi khô ráo, tránh ánh nắng. Một số loại cần giữ lạnh. "
            "Kiểm tra hạn dùng trước mỗi lần uống."
        ),
        url="",
        source="MedAssist",
        is_sample=True,
    ),
]


# ------------------------------------------------------------------------ cấu hình

def parse_feeds(raw: str | None) -> list[_Feed]:
    """Đọc cấu hình feed. URL không hợp lệ bị **bỏ im lặng** — thà thiếu tin còn hơn gọi bậy."""
    feeds: list[_Feed] = []
    for chunk in (raw or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, sep, url = chunk.partition("|")
        if not sep:
            name, url = "", chunk
        url = url.strip()
        if not _is_safe_url(url):
            continue
        feeds.append(_Feed(name=name.strip() or (urlparse(url).hostname or "Nguồn"), url=url))
    return feeds


def _is_safe_url(url: str) -> bool:
    """Chỉ http/https, có tên miền, và không trỏ vào chính máy chủ (chống SSRF)."""
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    host = p.hostname.lower()
    if host in _BLOCKED_HOSTS or host.endswith(".local") or host.endswith(".internal"):
        return False
    # Dải IP nội bộ dạng số.
    if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.)", host):
        return False
    return True


# ------------------------------------------------- cổng tin tức đã kiểm chứng (allowlist)

#: Biến môi trường: danh sách tên miền đăng ký được phép, ngăn cách bằng dấu phẩy.
ALLOWED_DOMAINS_ENV = "MEDASSIST_NEWS_ALLOWED_DOMAINS"
#: Mặc định: chỉ cơ quan y tế công chính thức. ⚠️ Chủ dự án phải tự kiểm chứng/chỉnh danh sách.
DEFAULT_ALLOWED_DOMAINS: tuple[str, ...] = (
    "fda.gov",
    "who.int",
    "cdc.gov",
    "nih.gov",
    "moh.gov.vn",
    "dav.gov.vn",
)


def allowed_domains() -> list[str]:
    """Danh sách tên miền được phép (chữ thường). Biến rỗng/không hợp lệ -> dùng mặc định."""
    raw = os.environ.get(ALLOWED_DOMAINS_ENV)
    out: list[str] = []
    for part in (raw or "").split(","):
        d = part.strip().lower().lstrip("*").strip(".")
        if d and re.fullmatch(r"[a-z0-9]([a-z0-9.-]*[a-z0-9])?", d) and d not in out:
            out.append(d)
    return out or list(DEFAULT_ALLOWED_DOMAINS)


def verified_sources() -> list[str]:
    """Các tên miền đã kiểm chứng đang áp dụng — để app hiện "Nguồn đã kiểm chứng"."""
    return allowed_domains()


def is_allowed_url(url: str | None) -> bool:
    """True chỉ khi URL là https, không userinfo, cổng 443/vắng, host == hoặc là subdomain
    của một mục trong allowlist (so khớp theo nhãn, không theo chuỗi con)."""
    if not url or any(c.isspace() or ord(c) < 32 for c in url) or "\\" in url:
        return False
    try:
        p = urlparse(url)
        port = p.port
    except ValueError:
        return False
    if p.scheme != "https" or not p.hostname or "@" in p.netloc:
        return False
    if p.username is not None or p.password is not None:
        return False
    if port not in (None, 443):
        return False
    host = p.hostname.lower().rstrip(".")
    if not _is_safe_url(url):
        return False
    return any(host == d or host.endswith("." + d) for d in allowed_domains())


def _upgrade_http(url: str) -> str:
    """`http://host/...` -> `https://host/...`; link khác giữ nguyên."""
    if url and url[:7].lower() == "http://":
        return "https://" + url[7:]
    return url


def _sanitize_links(article: NewsArticle) -> NewsArticle:
    """Link/ảnh không thuộc allowlist bị xoá; bài vẫn giữ dạng chỉ-văn-bản.

    Feed cũ (vd FDA) ghi link `http://`: nâng lên `https://` rồi mới kiểm tra — chỉ
    giữ nếu bản https vẫn qua allowlist. Không bao giờ trả link http cho app.
    """
    article.url = _upgrade_http(article.url)
    article.image_url = _upgrade_http(article.image_url) if article.image_url else None
    if article.url and not is_allowed_url(article.url):
        article.url = ""
    if article.image_url and not is_allowed_url(article.image_url):
        article.image_url = None
    return article


def configured_feeds() -> list[_Feed]:
    return parse_feeds(os.environ.get(FEEDS_ENV))


def _ttl() -> int:
    try:
        return max(60, int(os.environ.get(TTL_ENV, DEFAULT_TTL)))
    except ValueError:
        return DEFAULT_TTL


# -------------------------------------------------------------------------- parse

_TAG_RE = re.compile(r"<[^>]+>")
_IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)


def _clean_text(raw: str | None, limit: int = MAX_SUMMARY) -> str:
    """Bỏ thẻ HTML, giải mã ký tự, gộp khoảng trắng, cắt ở ranh giới từ."""
    if not raw:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", raw))
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _parse_date(raw: str | None) -> str | None:
    """Đọc ngày đăng. Mỗi báo một kiểu nên phải thử nhiều định dạng, và **không được ném lỗi**.

    Đo thật: một báo trả `10/28/2026 5:08:00 PM` (không phải RFC 822) và `GMT+7` thay vì
    `+0700`. Một parser cứng sẽ làm hỏng cả feed.
    """
    if not raw:
        return None
    raw = raw.strip().replace("GMT+7", "+0700").replace("GMT", "+0000")
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            continue
    return None


_NS = {"atom": "http://www.w3.org/2005/Atom", "dc": "http://purl.org/dc/elements/1.1/"}


def _text_of(node, *paths: str) -> str | None:
    """Thử lần lượt các đường dẫn, RSS thường rồi Atom.

    ⚠️ Phải truyền `_NS` vào `find()`: không truyền thì ElementTree không hiểu tiền tố
    `atom:` và lặng lẽ trả `None` — không ném lỗi, nên feed Atom trông như "parse được"
    nhưng mọi bài đều rỗng. Lỗi này chỉ lộ ra khi thử bằng một feed Atom thật.
    """
    for p in paths:
        found = node.find(p, _NS if ":" in p else None)
        if found is not None and (found.text or "").strip():
            return found.text
    return None


def parse_feed(xml_text: str, source: str) -> list[NewsArticle]:
    """Đọc RSS 2.0 hoặc Atom. XML hỏng -> trả danh sách rỗng, không ném lỗi."""
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return []

    items = root.findall(".//item") or root.findall(".//atom:entry", _NS)

    articles: list[NewsArticle] = []
    for item in items:
        title = _clean_text(_text_of(item, "title", "atom:title"), 180)
        if not title:
            continue

        link = _text_of(item, "link")
        if not link:  # Atom: <link href="..."/>
            el = item.find("atom:link", _NS)
            link = el.get("href") if el is not None else None
        link = (link or "").strip()

        raw_desc = _text_of(item, "description", "atom:summary", "atom:content") or ""
        summary = _clean_text(raw_desc)

        image = None
        enc = item.find("enclosure")
        if enc is not None and (enc.get("type") or "").startswith("image/"):
            image = enc.get("url")
        if not image:
            # Nhiều báo nhét ảnh vào CDATA của description thay vì enclosure.
            m = _IMG_RE.search(raw_desc)
            if m:
                image = m.group(1)
        if image and not _is_safe_url(image):
            image = None

        guid = _text_of(item, "guid", "atom:id") or link or title
        articles.append(
            NewsArticle(
                id=_clean_text(guid, 200),
                title=title,
                summary=summary,
                url=link,
                source=source,
                published_at=_parse_date(_text_of(item, "pubDate", "atom:updated", "dc:date")),
                image_url=image,
            )
        )
    return articles


# --------------------------------------------------------------------------- lọc

def is_relevant(article: NewsArticle) -> bool:
    """Giữ bài hay bỏ? Thà bỏ nhầm còn hơn hiện tin xổ số trong app y tế."""
    blob = _strip_accents(f"{article.title} {article.summary}")
    if _OFF_TOPIC.search(blob):
        return False
    if _ADVERT.search(blob):
        return False
    if article.url and _AFFILIATE.search(article.url):
        return False
    # Trích đoạn có số điện thoại hoặc giá tiền -> gần như chắc là quảng cáo.
    if re.search(r"\b0\d{8,10}\b|\b\d{1,3}(\.\d{3})+\s*(d|vnd|dong)\b", blob, re.IGNORECASE):
        return False
    return True


# -------------------------------------------------------------------------- tải

def _fetch(feed: _Feed) -> list[NewsArticle]:
    """Tải một feed. Mọi lỗi -> danh sách rỗng (không làm hỏng các feed khác)."""
    import httpx

    try:
        with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True, trust_env=False) as c:
            r = c.get(feed.url, headers={"User-Agent": "MedAssist/1.0 (+do an hoc tap)"})
            r.raise_for_status()
            # Chuyển hướng: mọi chặng (kể cả chặng cuối) phải còn trong allowlist.
            for hop in [*r.history, r]:
                if not is_allowed_url(str(hop.url)):
                    return []
            return parse_feed(r.text, feed.name)
    except Exception:
        # ⚠️ Cố ý nuốt chi tiết: thông điệp lỗi có thể chứa URL kèm tham số.
        return []


def get_news(force: bool = False) -> tuple[list[NewsArticle], bool, list[str]]:
    """`(danh sách bài, có phải bài mẫu không, tên các nguồn lỗi)`.

    Có cache trong RAM để không gọi báo mỗi lần người dùng mở màn hình tin tức.
    """
    feeds = configured_feeds()
    if not feeds:
        return list(SAMPLE_ARTICLES), True, []

    now = time.monotonic()
    if not force and _cache.articles and (now - _cache.fetched_at) < _ttl():
        return _cache.articles, False, _cache.failed

    collected: list[NewsArticle] = []
    failed: list[str] = []
    for feed in feeds:
        if not is_allowed_url(feed.url):
            failed.append(feed.name)  # chỉ tên, không URL
            continue
        got = [a for a in map(_sanitize_links, _fetch(feed)) if is_relevant(a)]
        if not got:
            failed.append(feed.name)
        collected.extend(got)

    # Bỏ trùng theo link (nhiều feed cùng đăng lại một bài).
    seen: set[str] = set()
    unique: list[NewsArticle] = []
    for a in collected:
        key = a.url or a.id
        if key in seen:
            continue
        seen.add(key)
        unique.append(a)

    unique.sort(key=lambda a: a.published_at or "", reverse=True)
    unique = unique[:MAX_ITEMS]

    if not unique:
        # Mọi nguồn đều hỏng -> giữ bản cache cũ nếu có, không trả màn hình trống.
        if _cache.articles:
            return _cache.articles, False, failed
        return list(SAMPLE_ARTICLES), True, failed

    _cache.articles, _cache.fetched_at, _cache.failed = unique, now, failed
    return unique, False, failed


def reset_cache() -> None:
    """Dùng trong test."""
    _cache.articles, _cache.fetched_at, _cache.failed = [], 0.0, []
