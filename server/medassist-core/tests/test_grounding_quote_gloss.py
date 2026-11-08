"""Quy tắc "trích nguyên văn + giải thích ngắn" của `verify_grounding` (dữ liệu giả)."""

from __future__ import annotations

from app.agent import verify_grounding
from app.schemas import CitationResponse


def _c(id_: str, excerpt: str, name: str = "Panadol 500mg") -> CitationResponse:
    return CitationResponse(
        id=id_, medicine_name=name, field="side_effects",
        excerpt=excerpt, source="s", is_authoritative=False,
    )


CITES = [_c("R1", "Paracetamol 500mg"), _c("R3", "Nausea, allergic reactions")]


def test_chap_nhan_trich_nguyen_van_kem_giai_thich_ngan():
    line = '- Tác dụng phụ: "Nausea, allergic reactions" nghĩa là buồn nôn, phản ứng dị ứng [R3]'
    ok, used = verify_grounding(line, CITES)
    assert ok is True and used == ["R3"]


def test_chap_nhan_so_co_trong_ten_thuoc_hoac_nguon():
    ok, _ = verify_grounding('- Thành phần "Paracetamol 500mg" của Panadol 500mg [R1]', CITES)
    assert ok is True


def test_tu_choi_chi_dich_khong_co_nguyen_van():
    ok, _ = verify_grounding("- Có thể gây buồn nôn và phản ứng dị ứng [R3]", CITES)
    assert ok is False


def test_tu_choi_giai_thich_qua_dai():
    gloss = " ".join(["từ"] * 20)
    ok, _ = verify_grounding(f'- "Nausea, allergic reactions" {gloss} [R3]', CITES)
    assert ok is False


def test_tu_choi_con_so_moi():
    ok, _ = verify_grounding('- "Nausea, allergic reactions" khi uống 3 viên mỗi ngày [R3]', CITES)
    assert ok is False


def test_tu_choi_ma_trich_dan_gia():
    ok, _ = verify_grounding('- "Nausea, allergic reactions" buồn nôn [R9]', CITES)
    assert ok is False


def test_tu_choi_trich_nguyen_van_cua_nguon_khac():
    # Dòng trích nguyên văn đoạn của R3 nhưng gắn mã R1 -> không bám đúng nguồn.
    ok, _ = verify_grounding('- "Nausea, allergic reactions" buồn nôn [R1]', CITES)
    assert ok is False


def test_tu_choi_doan_trich_qua_ngan_de_lach():
    ok, _ = verify_grounding('- "ab" thuốc này rất tốt cho mọi bệnh [R2]', [_c("R2", "ab")])
    assert ok is False
