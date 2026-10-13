"""Chuẩn hoá & tách âm tiết tiếng Việt (Day 20).

Câu hỏi quan trọng nhất của bộ test: **gõ không dấu có ra kết quả giống gõ có
dấu không?** — vì đó là cách người dùng Việt thật sự gõ trên bàn phím mobile.
"""

from __future__ import annotations

import pytest

from app.textnorm import (
    bigrams,
    content_tokens,
    normalize_query,
    query_terms,
    tokenize,
)


# ---------------------------------------------------------------- tách âm tiết


def test_bo_dau_truoc_khi_tach():
    """Bỏ dấu **trước**, tách **sau** — nếu đảo thì "hạ" != "ha"."""
    assert tokenize("Hạ sốt") == ["ha", "sot"]
    assert tokenize("Giảm đau") == ["giam", "dau"]


def test_gop_khoang_trang_thua():
    assert tokenize("  Panadol   500mg  ") == ["panadol", "500mg"]


def test_giu_chu_so():
    assert "500mg" in tokenize("Panadol 500mg")


def test_chu_D_da_dung():
    """`Đ` là chữ riêng, `unicodedata` NFD **không** bỏ được -> cần bảng fold."""
    assert tokenize("Đại hoàn sức khoẻ") == ["dai", "hoan", "suc", "khoe"]


def test_bigram():
    assert bigrams(["ha", "sot"]) == ["ha sot"]
    assert bigrams(["a"]) == []
    assert bigrams([]) == []


# ------------------------------------------- chuẩn hoá trước tokenization (chốt)


@pytest.mark.parametrize(
    ("co_dau", "khong_dau"),
    [
        ("Hạ sốt", "Ha Sot"),
        ("Giảm đau", "Giam dau"),
        ("Đau nóng", "Dau nong"),
        ("Thuốc ho Bảo Thanh", "Thuoc ho Bao Thanh"),
    ],
)
def test_gop_khong_dau_va_co_dau_cho_cung_ket_qua(co_dau, khong_dau):
    """⭐ Bảo đảm chuẩn hoá TRƯỚC tokenization, không phải sau."""
    assert tokenize(co_dau) == tokenize(khong_dau)
    assert query_terms(co_dau) == query_terms(khong_dau)


def test_trau_biet_nguon_giong_nhau():
    assert normalize_query("Hạ Sốt") == normalize_query("ha  sot")


# ------------------------------------------------------------ từ khoá truy vấn


def test_bo_tu_chuc_nang_khoi_truy_van():
    """`thuoc` là từ chức năng -> không mang thông tin phân biệt."""
    assert "thuoc" not in content_tokens("thuoc ho")


def test_khong_bo_so_trong_ten_thuoc():
    """Số là **một phần** của tên thuốc, không phải từ chức năng."""
    assert "500mg" in content_tokens("Panadol 500mg")


def test_query_terms_co_bien_goc_day_du():
    assert query_terms("Hạ sốt") == ["ha", "sot", "ha sot"]


def test_truy_van_rong_tok_khong_loi():
    assert tokenize("") == []
    assert query_terms("") == []
    assert content_tokens("") == []


def test_truy_van_chi_to_chuc_nang():
    """Chỉ từ chức năng -> không từ khoá nào -> agent từ chối, không đoán bừa."""
    assert content_tokens("của là gì") == []


def test_giới_hạn_do_dai_the():
    """Đoạn nguồn dài không được phình vô hạn."""
    assert len(content_tokens("mua thuoc " * 200)) <= 40


def test_chuan_hoa_khong_bien_mat_chu_hoa():
    """Unicode tổ hợp giữ nguyên chữ cái, không vỡ vụn."""
    assert tokenize("Nguyễn Văn A") == ["nguyen", "van", "a"]
