"""Tin tức y tế (`app/news.py`, `/v1/news`) — KHÔNG gọi mạng thật.

Mọi lần `httpx.get` được thay bằng `httpx.MockTransport`; dữ liệu RSS đều tự tạo giả.
"""

from __future__ import annotations

import httpx
import pytest

from app import news as news_mod
from app.news import (
    SAMPLE_ARTICLES,
    NewsArticle,
    _is_safe_url,
    get_news,
    is_relevant,
    parse_feed,
    parse_feeds,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    reset_cache()
    monkeypatch.delenv(news_mod.FEEDS_ENV, raising=False)
    # Test cũ dùng tên miền giả *.example.com -> cho phép riêng trong test.
    monkeypatch.setenv(news_mod.ALLOWED_DOMAINS_ENV, "example.com")
    yield
    reset_cache()


# ------------------------------------------------------------- cấu hình & SSRF


def test_khong_co_bien_moi_truong_thi_khong_co_feed_nao():
    assert parse_feeds(None) == []
    assert parse_feeds("") == []


def test_doc_ten_va_url():
    feeds = parse_feeds("Báo Gia|https://bao-gia.example.com/rss.xml")
    assert len(feeds) == 1
    assert feeds[0].name == "Báo Gia"
    assert feeds[0].url == "https://bao-gia.example.com/rss.xml"


def test_thieu_ten_thi_lay_ten_mien():
    feeds = parse_feeds("https://bao-gia.example.com/rss.xml")
    assert feeds[0].name == "bao-gia.example.com"


def test_nhieu_feed_cach_nhau_bang_phay():
    feeds = parse_feeds("A|https://a.example.com/r.xml, B|https://b.example.com/r.xml")
    assert [f.name for f in feeds] == ["A", "B"]


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/rss.xml",
        "http://127.0.0.1:8000/rss.xml",
        "http://169.254.169.254/latest/meta-data",
        "http://192.168.1.1/rss",
        "http://10.0.0.5/rss",
        "http://172.16.0.1/rss",
        "ftp://bao.example.com/rss.xml",
        "file:///etc/passwd",
        "not a url",
        "",
    ],
)
def test_chan_dia_chi_noi_bo_va_sai_dinh_dang(url):
    """Chống SSRF: ai đặt được biến môi trường cũng không khiến server tự gọi vào chính nó."""
    assert _is_safe_url(url) is False


def test_dia_chi_hop_le_duoc_chap_nhan():
    assert _is_safe_url("https://bao-gia.example.com/rss.xml") is True


def test_feed_tro_vao_noi_bo_bi_loai_am_tham():
    feeds = parse_feeds("Xau|http://localhost/rss.xml,Tot|https://ok.example.com/rss.xml")
    assert [f.name for f in feeds] == ["Tot"]


# ------------------------------------------------------------------- parse RSS


RSS_OK = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Panadol thu hoi lo san xuat loi</title>
  <description><![CDATA[Co quan quan ly thuoc thong bao thu hoi. <img src="https://img.example.com/a.jpg"/>]]></description>
  <link>https://bao.example.com/panadol-thu-hoi</link>
  <pubDate>Wed, 28 Oct 2026 10:00:00 +0700</pubDate>
  <guid>bao-1</guid>
</item>
<item>
  <title>Ket qua xo so mien Bac hom nay</title>
  <description>Giai dac biet...</description>
  <link>https://bao.example.com/xo-so</link>
  <pubDate>Wed, 28 Oct 2026 09:00:00 +0700</pubDate>
</item>
</channel></rss>"""

RSS_US_DATE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item>
  <title>Canh bao thuoc gia</title>
  <description>Noi dung</description>
  <link>https://bao.example.com/canh-bao</link>
  <pubDate>10/28/2026 5:08:00 PM</pubDate>
</item>
</channel></rss>"""

ATOM_OK = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
  <title>FDA recall notice</title>
  <summary>Some drug recalled</summary>
  <link href="https://fda.example.com/recall"/>
  <updated>2026-10-27T12:00:00Z</updated>
  <id>fda-1</id>
</entry>
</feed>"""


def test_doc_duoc_rss_2_0():
    arts = parse_feed(RSS_OK, "Báo Giả")
    assert len(arts) == 2
    assert arts[0].title == "Panadol thu hoi lo san xuat loi"
    assert arts[0].source == "Báo Giả"
    assert arts[0].url == "https://bao.example.com/panadol-thu-hoi"
    assert arts[0].image_url == "https://img.example.com/a.jpg"
    assert arts[0].published_at and arts[0].published_at.startswith("2026-10-28")


def test_doc_duoc_atom():
    arts = parse_feed(ATOM_OK, "FDA")
    assert len(arts) == 1
    assert arts[0].title == "FDA recall notice"
    assert arts[0].url == "https://fda.example.com/recall"


def test_xml_hong_tra_rong_khong_nem_loi():
    assert parse_feed("<rss><channel><item><title>", "X") == []
    assert parse_feed("", "X") == []
    assert parse_feed("khong phai xml", "X") == []


def test_ngay_kieu_my_khong_lam_crash():
    arts = parse_feed(RSS_US_DATE, "Báo Giả")
    assert arts[0].published_at and arts[0].published_at.startswith("2026-10-28")


def test_anh_khong_an_toan_bi_bo():
    rss = RSS_OK.replace("https://img.example.com/a.jpg", "http://169.254.169.254/img.jpg")
    arts = parse_feed(rss, "X")
    assert arts[0].image_url is None


def test_trich_doan_bi_cat_va_khong_con_the_html():
    rss = f"""<rss><channel><item><title>T</title>
      <description><![CDATA[{"x" * 500}]]></description>
      <link>https://e.com/1</link></item></channel></rss>"""
    arts = parse_feed(rss, "X")
    assert len(arts[0].summary) <= news_mod.MAX_SUMMARY + 1
    assert "<" not in arts[0].summary


# ---------------------------------------------------------------------- lọc


@pytest.mark.parametrize(
    "title,summary,url,expect",
    [
        ("Panadol thu hoi", "An toan thuoc", "https://e.com/1", True),
        ("Ket qua xo so", "Giai dac biet", "https://e.com/2", False),
        ("Tin bong da", "Ty so tran dau", "https://e.com/3", False),
        ("Nha thuoc uu dai", "Mua ngay gia chi 99k, hotline 0912345678", "https://e.com/4", False),
        ("Thuoc moi", "Thong tin", "https://e.com/5?utm_source=fb", False),
        ("Vu an mang o Ha Noi", "Khoi to bi can", "https://e.com/6", False),
    ],
)
def test_loc_bai_khong_phai_y_te(title, summary, url, expect):
    a = NewsArticle(id="1", title=title, summary=summary, url=url, source="X")
    assert is_relevant(a) is expect


def test_loc_khong_phan_biet_dau_tieng_viet():
    a = NewsArticle(id="1", title="Kết quả xổ số hôm nay", summary="", url="https://e.com", source="X")
    assert is_relevant(a) is False


# --------------------------------------------------------------- get_news (cache)


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_khong_dat_bien_moi_truong_tra_bai_mau(monkeypatch):
    arts, is_sample, failed = get_news()
    assert is_sample is True
    assert failed == []
    assert [a.id for a in arts] == [a.id for a in SAMPLE_ARTICLES]


def test_bai_mau_khong_co_khang_dinh_lieu_dung_cu_the():
    """Bài mẫu không được chứa số liều/hàm lượng cụ thể — tránh bị hiểu là tư vấn y khoa."""
    import re

    for a in SAMPLE_ARTICLES:
        assert not re.search(r"\d+\s*(mg|ml|viên|lần/ngày)", a.title + a.summary)


def test_bat_nguon_that_goi_dung_url_va_loc(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, text=RSS_OK)

    monkeypatch.setattr(news_mod, "_fetch", lambda feed: parse_feed(RSS_OK, feed.name))
    monkeypatch.setenv(news_mod.FEEDS_ENV, "Báo Gia|https://bao-gia.example.com/rss.xml")
    arts, is_sample, failed = get_news()
    assert is_sample is False
    assert failed == []
    assert len(arts) == 1  # bài xổ số đã bị lọc
    assert arts[0].source == "Báo Gia"


def test_feed_loi_duoc_bao_nhung_khong_lam_hong_feed_khac(monkeypatch):
    def fake_fetch(feed):
        if feed.name == "Hỏng":
            return []  # _fetch nuốt lỗi, trả rỗng
        return parse_feed(RSS_OK, feed.name)

    monkeypatch.setattr(news_mod, "_fetch", fake_fetch)
    monkeypatch.setenv(
        news_mod.FEEDS_ENV,
        "Hỏng|https://hong.example.com/rss.xml,Tốt|https://tot.example.com/rss.xml",
    )
    arts, is_sample, failed = get_news()
    assert is_sample is False
    assert failed == ["Hỏng"]
    assert len(arts) == 1
    assert arts[0].source == "Tốt"


def test_moi_nguon_deu_hong_lan_dau_thi_tra_bai_mau(monkeypatch):
    monkeypatch.setattr(news_mod, "_fetch", lambda feed: [])
    monkeypatch.setenv(news_mod.FEEDS_ENV, "X|https://x.example.com/rss.xml")
    arts, is_sample, failed = get_news()
    assert is_sample is True
    assert failed == ["X"]


def test_bo_trung_theo_link_giua_nhieu_nguon(monkeypatch):
    def fake_fetch(feed):
        return [
            NewsArticle(id="1", title="Panadol", summary="s", url="https://e.com/bai-1", source=feed.name)
        ]

    monkeypatch.setattr(news_mod, "_fetch", fake_fetch)
    monkeypatch.setenv(news_mod.FEEDS_ENV, "A|https://a.example.com/r.xml,B|https://b.example.com/r.xml")
    arts, _, _ = get_news()
    assert len(arts) == 1


def test_dung_cache_trong_ttl_khong_goi_lai(monkeypatch):
    calls = []

    def fake_fetch(feed):
        calls.append(1)
        return parse_feed(RSS_OK, feed.name)

    monkeypatch.setattr(news_mod, "_fetch", fake_fetch)
    monkeypatch.setenv(news_mod.FEEDS_ENV, "X|https://x.example.com/rss.xml")
    get_news()
    get_news()
    assert len(calls) == 1  # lần 2 dùng cache


def test_refresh_bo_qua_cache(monkeypatch):
    calls = []

    def fake_fetch(feed):
        calls.append(1)
        return parse_feed(RSS_OK, feed.name)

    monkeypatch.setattr(news_mod, "_fetch", fake_fetch)
    monkeypatch.setenv(news_mod.FEEDS_ENV, "X|https://x.example.com/rss.xml")
    get_news()
    get_news(force=True)
    assert len(calls) == 2


def test_so_bai_toi_da(monkeypatch):
    many = [
        NewsArticle(id=str(i), title=f"Thuoc so {i}", summary="s", url=f"https://e.com/{i}", source="X")
        for i in range(news_mod.MAX_ITEMS + 10)
    ]
    monkeypatch.setattr(news_mod, "_fetch", lambda feed: many)
    monkeypatch.setenv(news_mod.FEEDS_ENV, "X|https://x.example.com/rss.xml")
    arts, _, _ = get_news()
    assert len(arts) == news_mod.MAX_ITEMS


# ------------------------------------------------------------- _fetch thật (giả transport)


_RealHttpxClient = httpx.Client  # capture TRƯỚC khi test nào patch httpx.Client


class _FakeClientFactory:
    """Thay `httpx.Client(...)` bằng bản dùng `MockTransport`.

    ⚠️ Phải giữ tham chiếu tới lớp `Client` **gốc** (`_RealHttpxClient`) từ trước khi patch —
    gọi `httpx.Client(...)` ở đây sau khi đã patch sẽ tự gọi lại chính nó và đệ quy vô hạn
    cho tới khi `RecursionError` bị `except Exception` trong `_fetch` nuốt mất, khiến test
    "qua" với danh sách rỗng mà không ai nhận ra request chưa từng thật sự chạy.
    """

    def __init__(self, handler):
        self._handler = handler

    def __call__(self, *a, **k):
        return _RealHttpxClient(transport=httpx.MockTransport(self._handler))


def test_fetch_dung_mock_transport_khong_goi_mang_that(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://bao-gia.example.com/rss.xml"
        return httpx.Response(200, text=RSS_OK)

    monkeypatch.setattr(httpx, "Client", _FakeClientFactory(handler))
    feed = news_mod._Feed(name="Báo Gia", url="https://bao-gia.example.com/rss.xml")
    arts = news_mod._fetch(feed)
    # _fetch KHÔNG lọc (is_relevant chạy ở get_news) -> cả 2 bài trong RSS_OK đều có mặt.
    assert len(arts) == 2
    assert arts[0].title == "Panadol thu hoi lo san xuat loi"


def test_fetch_loi_mang_tra_rong_khong_nem(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "Client", _FakeClientFactory(handler))
    feed = news_mod._Feed(name="X", url="https://x.example.com/rss.xml")
    assert news_mod._fetch(feed) == []


# ------------------------------------------------------------------------ API


def test_endpoint_khong_can_token(client):
    r = client.get("/v1/news")
    assert r.status_code == 200
    body = r.json()
    assert body["is_sample"] is True
    assert len(body["articles"]) >= 1
    assert body["disclaimer"]


def test_endpoint_gioi_han_limit(client):
    r = client.get("/v1/news?limit=1")
    assert r.status_code == 200
    assert len(r.json()["articles"]) == 1


def test_endpoint_tu_choi_limit_ngoai_khoang(client):
    assert client.get("/v1/news?limit=0").status_code == 422
    assert client.get("/v1/news?limit=1000").status_code == 422


def test_bai_mau_khong_co_url_that(client):
    r = client.get("/v1/news")
    for a in r.json()["articles"]:
        assert a["url"] == ""


# ------------------------------------------------- cổng tin tức đã kiểm chứng (allowlist)

from app.news import ALLOWED_DOMAINS_ENV, DEFAULT_ALLOWED_DOMAINS, allowed_domains, is_allowed_url


@pytest.fixture
def default_list(monkeypatch):
    monkeypatch.delenv(ALLOWED_DOMAINS_ENV, raising=False)


def test_danh_sach_mac_dinh(default_list):
    assert allowed_domains() == list(DEFAULT_ALLOWED_DOMAINS)
    assert set(DEFAULT_ALLOWED_DOMAINS) == {
        "fda.gov", "who.int", "cdc.gov", "nih.gov", "moh.gov.vn", "dav.gov.vn",
    }


def test_env_ghi_de_danh_sach(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, " Foo.org , *.bar.vn ,")
    assert allowed_domains() == ["foo.org", "bar.vn"]
    assert is_allowed_url("https://www.foo.org/a")
    assert not is_allowed_url("https://fda.gov/a")


def test_env_rong_dung_mac_dinh(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, " , ")
    assert allowed_domains() == list(DEFAULT_ALLOWED_DOMAINS)


@pytest.mark.parametrize(
    "url",
    [
        "https://fda.gov/a",
        "https://www.fda.gov/drugs/x?y=1",
        "https://WWW.FDA.GOV/a",
        "https://who.int/news",
        "https://dav.gov.vn:443/x",
        "https://sub.moh.gov.vn/x",
    ],
)
def test_allowlist_chap_nhan(default_list, url):
    assert is_allowed_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://evilfda.gov/a",  # lookalike (không có dấu chấm)
        "https://fda.gov.evil.com/a",  # allowlist làm tiền tố
        "https://notfda.gov/a",
        "https://fda.gov@evil.com/a",  # userinfo
        "https://user:pw@fda.gov/a",  # userinfo ngay cả khi host đúng
        "https://evil.com/fda.gov",
        "https://evil.com/?u=https://fda.gov",
        "https://fda.gov:8443/a",  # cổng lạ
        "https://fda.gov:80/a",
        "http://fda.gov/a",  # không https
        "ftp://fda.gov/a",
        "javascript:alert(1)",
        "https://fda.gov\\@evil.com/a",
        "https://fda.gov /a",
        "",
        "https://localhost/a",
    ],
)
def test_allowlist_tu_choi(default_list, url):
    assert not is_allowed_url(url)


def _rss(link):
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel><item>'
        "<title>Canh bao thuoc</title><description>Noi dung</description>"
        f"<link>{link}</link><guid>g1</guid></item></channel></rss>"
    )


def test_feed_ngoai_allowlist_bi_bo_va_bao_ten_thoi(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, "fda.gov")
    called = []
    monkeypatch.setattr(news_mod, "_fetch", lambda f: called.append(f.name) or [])
    monkeypatch.setenv(news_mod.FEEDS_ENV, "Xau|https://evil.example.org/rss.xml?k=secret")
    _, _, failed = get_news()
    assert called == []  # không hề gọi tới
    assert failed == ["Xau"]
    assert "evil" not in "".join(failed)


def test_link_ngoai_allowlist_bi_xoa_bai_van_giu(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, "fda.gov")
    monkeypatch.setenv(news_mod.FEEDS_ENV, "FDA|https://www.fda.gov/rss.xml")
    monkeypatch.setattr(
        news_mod, "_fetch", lambda f: parse_feed(_rss("https://evil.example.org/x"), f.name)
    )
    arts, is_sample, _ = get_news()
    assert is_sample is False
    assert len(arts) == 1 and arts[0].url == "" and arts[0].title


def test_link_http_duoc_nang_https_ngoai_allowlist_van_bi_xoa(monkeypatch):
    """App không bao giờ nhận link http: feed FDA ghi `http://` -> nâng lên https
    (vẫn phải qua allowlist); http ngoài allowlist vẫn bị xoá."""
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, "fda.gov")
    monkeypatch.setenv(news_mod.FEEDS_ENV, "FDA|https://www.fda.gov/rss.xml")
    for link, expect in [
        ("http://www.fda.gov/a", "https://www.fda.gov/a"),
        ("https://www.fda.gov/a", "https://www.fda.gov/a"),
        ("http://evil.example.org/a", ""),
        ("http://www.fda.gov:8080/a", ""),
    ]:
        reset_cache()
        monkeypatch.setattr(news_mod, "_fetch", lambda f, link=link: parse_feed(_rss(link), f.name))
        arts, _, _ = get_news(force=True)
        assert arts[0].url == expect


def test_anh_ngoai_allowlist_bi_xoa(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, "fda.gov")
    monkeypatch.setenv(news_mod.FEEDS_ENV, "FDA|https://www.fda.gov/rss.xml")
    xml = _rss("https://www.fda.gov/a").replace(
        "<guid>", '<enclosure url="https://tracker.example.org/p.jpg" type="image/jpeg"/><guid>'
    )
    monkeypatch.setattr(news_mod, "_fetch", lambda f: parse_feed(xml, f.name))
    arts, _, _ = get_news()
    assert arts[0].image_url is None and arts[0].url


def test_redirect_ra_ngoai_allowlist_bi_bo(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, "fda.gov")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.fda.gov":
            return httpx.Response(302, headers={"location": "https://evil.example.org/rss.xml"})
        return httpx.Response(200, text=_rss("https://www.fda.gov/a"))

    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **k: _RealHttpxClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )
    assert news_mod._fetch(news_mod._Feed("FDA", "https://www.fda.gov/rss.xml")) == []


def test_redirect_trong_allowlist_van_duoc(monkeypatch):
    monkeypatch.setenv(ALLOWED_DOMAINS_ENV, "fda.gov")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.fda.gov":
            return httpx.Response(301, headers={"location": "https://feeds.fda.gov/rss.xml"})
        return httpx.Response(200, text=_rss("https://www.fda.gov/a"))

    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **k: _RealHttpxClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )
    assert len(news_mod._fetch(news_mod._Feed("FDA", "https://www.fda.gov/rss.xml"))) == 1


def test_response_co_verified_sources(client, monkeypatch):
    monkeypatch.delenv(ALLOWED_DOMAINS_ENV, raising=False)
    body = client.get("/v1/news").json()
    assert body["verified_sources"] == list(DEFAULT_ALLOWED_DOMAINS)
    assert "failed_sources" in body and "is_sample" in body
