"""Bộ lọc phạm vi (out-of-scope) phải bắt được tiếng Việt CÓ DẤU.

Regex cũ viết trên chuỗi gốc với character class (`[uương]`, `[eê]`...) bị sai:
một class chỉ khớp MỘT ký tự nên "tương", "liều", "chẩn đoán" lọt qua. Bộ lọc
mới khớp trên chuỗi đã `normalize()` (xem `app/textnorm.py`).

Toàn bộ câu hỏi ở đây là dữ liệu GIẢ.
"""

from __future__ import annotations

import pytest

from app.agent import is_out_of_scope

OUT_OF_SCOPE = [
    "Chẩn đoán giúp tôi bệnh gì?",
    "chan doan benh cua toi",
    "Liều dùng của Panadol là bao nhiêu?",
    "Liều dùng cho trẻ em thế nào",
    "Liều cao nhất mỗi ngày là gì",
    "Tương tác giữa hai thuốc này",
    "TƯƠNG TÁC thuốc",
    "drug interaction check",
    "Cấp cứu ngộ độc thuốc",
    "Phụ nữ mang thai uống được không",
    "Đang cho con bú có dùng được không",
    "Tôi có nên ngưng thuốc không",
    "Kê đơn cho tôi",
    "Tôi uống bao nhiêu viên một ngày",
    "dose 5 mg/kg",
]

IN_SCOPE = [
    "Panadol 500mg là thuốc gì?",
    "Cho tôi biết thông tin về Hapacol 500",
    "Thành phần của Efferalgan",
    "Công dụng của thuốc hạ sốt",
    "Tác dụng phụ của Panadol",
    "Hạ sốt",
    "Thuốc dùng để làm gì",
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_cau_hoi_ngoai_pham_vi_bi_chan(question: str) -> None:
    assert is_out_of_scope(question), question


@pytest.mark.parametrize("question", IN_SCOPE)
def test_cau_hoi_tra_cuu_thuong_khong_bi_chan(question: str) -> None:
    assert not is_out_of_scope(question), question


def test_cho_phep_bo_qua_nhom_tuong_tac_cho_luong_chat() -> None:
    """`/chat` định tuyến câu hỏi tương tác sang tool `check_interaction`."""
    question = "Tương tác giữa warfarin và panadol?"
    assert is_out_of_scope(question)
    assert not is_out_of_scope(question, allow_interaction=True)
    # Nhưng nhóm khác vẫn bị chặn dù cho phép tương tác.
    assert is_out_of_scope("Liều dùng của warfarin", allow_interaction=True)
