"""Day 18–19 — ingest danh muc thuoc va tim kiem theo TEN + HOAT CHAT.

Kiem chung quy tac du lieu: ket qua **phai** co `source`, va co co
`is_authoritative` de app/agent biet day co phai nguon lam sang hay khong.
"""

from __future__ import annotations

import pytest

from app.ingest import (
    DEV_SOURCE_LABEL,
    has_interaction_data,
    load_csv,
    split_ingredients,
    to_rows,
)
from app.sqlite_repository import normalize


@pytest.fixture()
def seeded(repo):
    """Nạp CSV dev thật vào kho in-memory."""
    drugs = load_csv()
    repo.add_medicines(to_rows(drugs))
    return repo, drugs


# ---------------------------------------------------------------- Ingest ----


def test_csv_duoc_doc():
    drugs = load_csv()
    assert len(drugs) > 0
    assert all(d.name for d in drugs)


def test_tach_hoat_chat():
    assert split_ingredients("Paracetamol 500mg, Vitamin C") == [
        "paracetamol 500mg",
        "vitamin c",
    ]
    assert split_ingredients("Menthol; Eucalyptol") == ["menthol", "eucalyptol"]
    assert split_ingredients(None) == []
    assert split_ingredients("") == []


def test_du_lieu_dev_KHONG_phai_nguon_lam_sang(seeded):
    """Quy tac AGENTS.md: khong được bia nguon y khoa."""
    _repo, drugs = seeded
    assert all(d.is_authoritative is False for d in drugs)
    assert all(d.source.startswith("dev-sample:") for d in drugs)


def test_khong_co_du_lieu_tuong_tac_thuoc():
    """Khong co nguon de kiem chung -> phai tro False de agent tu choi."""
    assert has_interaction_data() is False


def test_file_csv_khong_ton_tai_tra_ve_rong(tmp_path):
    assert load_csv(tmp_path / "khong-co.csv") == []


# ---------------------------------------------------------------- Search ----


def test_tim_theo_ten(seeded, client):
    body = client.get("/v1/search", params={"name": "Panadol"}).json()

    assert body["total"] >= 1
    assert body["results"][0]["name"] == "Panadol 500mg"


def test_tim_khong_dau_van_ra(seeded, client):
    """Người Việt gõ không dấu vẫn ra kết quả."""
    for query in ("panadol", "PANADOL", "Paracetamol"):
        assert client.get("/v1/search", params={"name": query}).json()["total"] >= 1


def test_tim_theo_hoat_chat(seeded, client):
    """Tìm theo HOẠT CHẤT, không chỉ tên thuốc."""
    body = client.get("/v1/search", params={"name": "amoxicillin"}).json()

    names = [r["name"] for r in body["results"]]
    assert any("Amoxicillin" in n for n in names)


def test_tim_theo_hoat_chat_khong_dau(seeded, client):
    """'paracetamol' là hoạt chất -> ra được nhiều nhãn thương hiệu."""
    body = client.get("/v1/search", params={"name": "paracetamol"}).json()

    names = [r["name"] for r in body["results"]]
    assert len(names) > 1, "phải ra nhiều thuốc chứa paracetamol"
    assert any("Panadol" in n for n in names)
    assert any("Efferalgan" in n for n in names)


def test_ket_qua_co_nguon_va_co_thoi_gian(seeded, client):
    """DoD Day 18: mọi kết quả phải kèm source + updated_at."""
    for item in client.get("/v1/search", params={"name": "panadol"}).json()["results"]:
        assert item["source"] is not None
        assert item["updated_at"] is not None
        assert item["is_authoritative"] is False


def test_nguon_cu_the_loai_chi_dinh_vao_file(seeded, client):
    """`source` phải trỏ về đúng file đã nạp, không phải nguồn bịa."""
    item = client.get("/v1/search", params={"name": "panadol"}).json()["results"][0]
    assert item["source"] == "dev-sample: medicines.dev.csv"


def test_hoat_chat_duoc_chuan_hoa(seeded, client):
    item = client.get("/v1/search", params={"name": "panadol"}).json()["results"][0]
    assert "paracetamol 500mg" in item["ingredients"]


def test_pagination(seeded, client):
    page1 = client.get("/v1/search", params={"name": "a", "limit": 2, "offset": 0}).json()
    page2 = client.get("/v1/search", params={"name": "a", "limit": 2, "offset": 2}).json()

    assert len(page1["results"]) <= 2
    assert {r["name"] for r in page1["results"]}.isdisjoint(
        {r["name"] for r in page2["results"]}
    )


def test_normalize_van_dung_cho_hoat_chat():
    assert normalize("Vitamin C") == "vitamin c"


# ------------------------------------------- Danh sách kiểm thử tay (DoD) ---

#: ROADMAP Day 19: "gõ không dấu 'panadol', 'para 500' đều ra đúng thuốc".
MANUAL_CASES = [
    ("panadol", "Panadol 500mg"),
    ("para 500", "Para 500"),
    ("efferalgan", "Efferalgan"),
    ("amoxicillin", "Amoxicillin 500mg"),
    ("decolgen", "Decolgen"),
    ("strepsils", "Strepsils"),
    ("hapacol", "Hapacol 500"),
]


@pytest.mark.parametrize("query,expected", MANUAL_CASES)
def test_kiem_thu_tay(query, expected, seeded, client):
    """≥90% ca kiểm thử tay phải ra đúng thuốc và có nguồn."""
    body = client.get("/v1/search", params={"name": query}).json()
    names = [r["name"] for r in body["results"]]
    assert expected in names, f"'{query}' không ra '{expected}' (ra: {names})"
    for item in body["results"]:
        assert item["source"] is not None