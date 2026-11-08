"""Khử định danh (`app/deidentify.py`) — toàn bộ dữ liệu test là GIẢ, không thật."""

from __future__ import annotations

import pytest

from app.deidentify import deidentify


@pytest.mark.parametrize(
    "text,label",
    [
        ("Liên hệ qua email demo@example.com nhé", "EMAIL"),
        ("Gọi số 0912345678 giúp tôi", "PHONE"),
        ("Gọi số +84912345678 giúp tôi", "PHONE"),
        ("CCCD của tôi là 001099012345", "ID_CARD"),
        ("Xem thêm tại https://benhvien.example.com/ho-so", "URL"),
        ("Server nội bộ 192.168.1.10", "IP"),
        ("Ngày khám 28/10/2026 có kết quả", "DATE"),
    ],
)
def test_bat_duoc_tung_loai_dinh_danh(text, label):
    r = deidentify(text)
    assert f"<{label}>" in r.text
    assert r.redacted_counts.get(label) == 1
    assert r.had_any is True


def test_ten_co_danh_xung_bi_thay_the():
    r = deidentify("Bệnh nhân Nguyễn Văn Hoàng bị sốt")
    assert "<NAME>" in r.text
    assert "Hoàng" not in r.text
    assert r.redacted_counts.get("NAME") == 1


def test_khong_co_dinh_danh_thi_giu_nguyen():
    r = deidentify("Panadol có tác dụng phụ gì")
    assert r.text == "Panadol có tác dụng phụ gì"
    assert r.had_any is False
    assert r.redacted_counts == {}


def test_nhieu_loai_cung_luc():
    r = deidentify(
        "Bệnh nhân Trần Thị Bình, SĐT 0987654321, email binh@example.com, "
        "khám ngày 01/02/2026"
    )
    assert r.redacted_counts.get("NAME") == 1
    assert r.redacted_counts.get("PHONE") == 1
    assert r.redacted_counts.get("EMAIL") == 1
    assert r.redacted_counts.get("DATE") == 1
    assert "Bình" not in r.text
    assert "0987654321" not in r.text


def test_khong_ghi_lai_gia_tri_goc_chi_dem_so_luong():
    """Kiểm tra hợp đồng: redacted_counts là SỐ, không phải danh sách giá trị."""
    r = deidentify("Gọi 0912345678")
    assert isinstance(r.redacted_counts["PHONE"], int)
    assert all(isinstance(v, int) for v in r.redacted_counts.values())


def test_chuoi_rong_khong_nem_loi():
    r = deidentify("")
    assert r.text == ""
    assert r.had_any is False


def test_none_khong_nem_loi():
    r = deidentify(None)  # type: ignore[arg-type]
    assert r.text == ""


def test_cccd_khong_bat_nham_so_dien_thoai_gan_do():
    """Số 10 chữ số (không phải 9 hay 12) không bị ID_CARD bắt nhầm."""
    r = deidentify("Mã đơn hàng 1234567890")
    assert "ID_CARD" not in r.redacted_counts


def test_unicode_nfd_van_bat_duoc_ten():
    """Tên gõ ở dạng NFD (dấu tách rời) vẫn phải khớp sau khi chuẩn hoá NFC."""
    import unicodedata

    nfd_name = unicodedata.normalize("NFD", "Bệnh nhân Lê Thị Hoà bị ho")
    r = deidentify(nfd_name)
    assert "<NAME>" in r.text


def test_khong_bat_qua_nhieu_ten_thuoc_binh_thuong():
    """Câu hỏi bình thường về thuốc không bị hiểu nhầm là chứa định danh."""
    r = deidentify("Paracetamol và Amoxicillin có dùng chung được không")
    assert r.had_any is False
