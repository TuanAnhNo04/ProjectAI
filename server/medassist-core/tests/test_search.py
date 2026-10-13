"""Test tìm kiếm thuốc — đặc biệt là chuẩn hoá không dấu."""

from __future__ import annotations

from app.sqlite_repository import normalize

SEED = [
    {"name": "Paracetamol 500mg", "composition": "Paracetamol", "uses": "Giảm đau"},
    {"name": "Ibuprofen 400mg", "composition": "Ibuprofen", "uses": "Chống viêm"},
    {"name": "Amoxicillin 250mg", "composition": "Amoxicillin", "uses": "Kháng sinh"},
]


def _seed(repo) -> None:
    repo.add_medicines(SEED)


def test_tim_theo_ten_chinh_xac(client, repo):
    _seed(repo)

    body = client.get("/v1/search", params={"name": "Paracetamol"}).json()

    assert body["total"] == 1
    assert body["results"][0]["name"] == "Paracetamol 500mg"
    assert body["query"] == "Paracetamol"


def test_tim_khong_dau_van_ra_ket_qua(client, repo):
    """Người dùng Việt hay gõ không dấu -> phải ra kết quả."""
    _seed(repo)

    for query in ("paracetamol", "PARACETAMOL", "Para"):
        assert client.get("/v1/search", params={"name": query}).json()["total"] >= 1


def test_tim_theo_hoa_chat(client, repo):
    _seed(repo)

    body = client.get("/v1/search", params={"name": "ibuprofen"}).json()
    assert body["results"][0]["name"] == "Ibuprofen 400mg"


def test_tim_khong_co_ket_qua(client, repo):
    _seed(repo)

    body = client.get("/v1/search", params={"name": "thuoc-khong-ton-tai"}).json()

    assert body["results"] == []
    assert body["total"] == 0


def test_search_khong_can_token(client, repo):
    """Tìm kiếm là công khai — app gọi được trước khi đăng nhập."""
    _seed(repo)
    assert client.get("/v1/search", params={"name": "para"}).status_code == 200


def test_pagination(client, repo):
    _seed(repo)

    page1 = client.get("/v1/search", params={"name": "m", "limit": 2, "offset": 0}).json()
    page2 = client.get("/v1/search", params={"name": "m", "limit": 2, "offset": 2}).json()

    assert len(page1["results"]) == 2
    assert page1["total"] == 3  # total là TỔNG, không phải số của trang
    assert len(page2["results"]) == 1
    # Không trùng nhau giữa 2 trang.
    assert {r["name"] for r in page1["results"]}.isdisjoint(
        {r["name"] for r in page2["results"]}
    )


def test_limit_bi_chan_tran(client):
    """`limit` vượt trần -> 422 thay vì kéo cả bảng về (rủi ro DoS)."""
    assert client.get("/v1/search", params={"name": "a", "limit": 10000}).status_code == 422


def test_ten_hoi_rong_bi_loi(client):
    assert client.get("/v1/search", params={"name": ""}).status_code == 422


def test_ket_qua_co_nguon(client, repo):
    """Dữ liệu thuốc phải kèm `source` — app không được hiện thông tin
    không truy được nguồn (quy tắc `AGENTS.md` về nguồn y khoa)."""
    repo.add_medicines([{"name": "Panadol", "source": "WHO-formulary"}])

    item = client.get("/v1/search", params={"name": "Panadol"}).json()["results"][0]

    assert item["source"] == "WHO-formulary"
    assert item["updated_at"] is not None


def test_normalize_bo_dau_va_ha_chu_thuong():
    assert normalize("Paracetamol 500mg") == "paracetamol 500mg"
    assert normalize("Đau Mỗi") == "dau moi"
    assert normalize("  A   B  ") == "a b"