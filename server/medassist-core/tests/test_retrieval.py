"""Truy hồi căn cứ (retrieval) — Day 20."""

from __future__ import annotations

import pytest

from app.ingest import load_csv, to_rows
from app.retrieval import (
    CITABLE_FIELDS,
    MIN_RELEVANCE_SCORE,
    build_citations,
    has_authoritative_evidence,
)
from app.schemas import MedicineResponse


def _medicines() -> list[MedicineResponse]:
    """Nạp đúng bộ dữ liệu dev thật (không bịa dữ liệu trong test)."""
    import json

    out = []
    for row in to_rows(load_csv()):
        row = dict(row)
        row["ingredients"] = json.loads(row["ingredients"])
        out.append(MedicineResponse(**row))
    return out


# --------------------------------------------------------------- trích dẫn


def test_nguong_loc_bien_tinh_bang_du_lieu():
    """⭐ Ngưỡng `MIN_RELEVANCE_SCORE` phải tách được 2 nhóm câu hỏi.

    Vì sao test này quan trọng: nếu ai đó đổi bộ dữ liệu dev mà không đo lại,
    ngưỡng có thể trở nên vô nghĩa (hoặc chặn hết, hoặc cho qua tất cả).
    """
    meds = _medicines()

    related = build_citations(meds, "Panadol 500mg dùng để làm gì")
    assert related, "câu hỏi liên quan phải tìm được căn cứ"
    assert max(r.score for r in related) > 0.3

    unrelated = build_citations(meds, "thuoc khong ton tai abcxyz")
    assert unrelated == [], "câu hỏi không liên quan phải không tìm thấy căn cứ"

    assert MIN_RELEVANCE_SCORE > 0


@pytest.mark.parametrize(
    "question",
    [
        "bạn bị gì?",          # "ban" khớp "phát ban" -> phải bị chặn
        "của bạn là gì?",
        "giúp gi",
        "có hai thuốc uống chung không",  # câu hỏi y khoa, không có nguồn
        "việc làm ăn uống gì",
        "uống một lúc mấy viên",
    ],
)
def test_cau_hoi_khong_lien_ran_khong_tim_thay_can_cu(question):
    """⭐ False positive nguy hiểm nhất là từ bất hàm trùng tình cờ.

    Không có `MIN_TERM_COVERAGE`, câu "bạn bị gì?" khớp mỗi chữ "phát ban" và
    đạt điểm 0.50 — hệ thống trả lời lệch hẳn câu hỏi mà vẫn tưởng có căn cứ.
    """
    assert build_citations(_medicines(), question) == []


def test_hoi_chung_ve_tac_dung_van_tim_thay():
    """Câu hỏi chung về "tác dụng" **nên** ra nguồn — không chặn quá tay."""
    hits = build_citations(_medicines(), "tác dụng phụ của panadol")
    assert hits
    assert any("panadol" in h.citation.medicine_name.lower() for h in hits)


def test_tim_theo_ten_thuoc():
    hits = build_citations(_medicines(), "Panadol 500mg")
    assert hits
    assert hits[0].citation.medicine_name == "Panadol 500mg"


def test_tim_theo_hoat_chat():
    """Hỏi bằng **hoạt chất** vẫn ra thuốc chứa hoạt chất đó."""
    hits = build_citations(_medicines(), "paracetamol")
    assert hits
    names = {h.citation.medicine_name for h in hits}
    assert any("Paracetamol" in n for n in names)


def test_gop_khong_dau_van_tim_thay():
    meds = _medicines()
    assert build_citations(meds, "thuoc ho bao thanh") == build_citations(
        meds, "Thuốc ho Bảo Thanh"
    )


def test_ma_trich_dan_duy_nhat_tang_dan():
    """Mỗi đoạn một mã riêng — app hiển thị được trích dẫn lồng nhau."""
    hits = build_citations(_medicines(), "panadol", top_k=5)
    ids = [h.citation.id for h in hits]
    assert ids == sorted(ids, key=lambda x: int(x[1:]))
    assert len(set(ids)) == len(ids)


def test_khong_trich_dan_nguon_khong_duoc_phep():
    """Chỉ trích `CITABLE_FIELDS` — `manufacturer` cố ý bị loại."""
    fields = {f for f, _ in CITABLE_FIELDS}
    assert "manufacturer" not in fields

    hits = build_citations(_medicines(), "panadol GSK", top_k=5)
    assert all(h.citation.field in fields for h in hits)


def test_giu_nguon_va_co_canh_bao():
    """Mọi trích dẫn phải có `source` và cờ `is_authoritative` (Đề bài #3)."""
    for h in build_citations(_medicines(), "panadol"):
        assert h.citation.source, "thiếu nguồn thì app không truy được"
        assert h.citation.is_authoritative is False


def test_giới_hạn_top_k():
    hits = build_citations(_medicines(), "paracetamol", top_k=2)
    assert len(hits) <= 2


def test_top_k_khong_phai_so_am():
    """`top_k=0` phải ra rỗng chứ không trả hết bảng."""
    assert build_citations(_medicines(), "paracetamol", top_k=0) == []


def test_cau_hoi_rong_khong_loi():
    assert build_citations(_medicines(), "") == []


def test_danh_muc_rong():
    assert build_citations([], "panadol") == []


def test_diem_luon_trong_0_1():
    for h in build_citations(_medicines(), "panadol", top_k=10):
        assert 0.0 <= h.score <= 1.0


def test_excerpt_khong_ba_cat_cum():
    """Cắt phải dừng ở ranh giới từ, không cắt giữa chữ."""
    for h in build_citations(_medicines(), "hieu anethol qua", top_k=10):
        if h.citation.excerpt.endswith("..."):
            assert not h.citation.excerpt[:-3].endswith(" ")


# --------------------------------------------------- nguồn không authoritative


def test_du_lieu_dev_khong_bao_gi_la_căn_cứ_manh():
    """⭐ Dữ liệu dev KHÔNG authoritative -> không được coi là căn cứ đủ tin cậy."""
    hits = build_citations(_medicines(), "panadol")
    assert hits, "vẫn trích được để mô tả"
    assert has_authoritative_evidence(hits) is False


def test_nguong_authoritative_dung_nguon_ingest():
    """Hai module phải dùng **cùng** một ngưỡng, không để lệch âm thầm."""
    from app.ingest import MIN_AUTHORITATIVE_SCORE

    hits = build_citations(_medicines(), "panadol")
    fake = [
        h.__class__(
            citation=h.citation.model_copy(update={"is_authoritative": True}),
            score=0.5,
        )
        for h in hits
    ]
    # 0.5 < MIN_AUTHORITATIVE_SCORE (0.8) -> vẫn False
    assert has_authoritative_evidence(fake) is False
    strong = [
        h.__class__(
            citation=h.citation.model_copy(update={"is_authoritative": True}),
            score=MIN_AUTHORITATIVE_SCORE,
        )
        for h in hits
    ]
    assert has_authoritative_evidence(strong) is True


def test_rong_khong_co_căn_cứ():
    assert has_authoritative_evidence([]) is False
