"""Danh mục LỚN (giả) — bắt lỗi "chỉ nhìn 100 thuốc đầu" mà 10 thuốc mẫu không lộ ra.

Toàn bộ dữ liệu là tên bịa ghép từ âm tiết; không có dữ liệu thật.
"""

from __future__ import annotations

import itertools

import pytest

from app.agent import answer_question
from app.sqlite_repository import SQLiteRepository
from app.tools import ExplainForPatientTool

A = ["Zor", "Qin", "Vel", "Mur", "Tha", "Bru", "Kel", "Dro", "Fyn", "Gal", "Hox", "Jun"]
B = ["ta", "ve", "mi", "ro", "su", "no", "li", "pa", "ke", "do", "zu", "ba"]
C = ["xin", "pol", "zen", "dar", "mex", "tol", "vir", "lan", "rox", "fen", "gom", "sil"]
NAMES = ["".join(p) for p in itertools.product(A, B, C)]  # 1728 tên bịa, toàn chữ cái


def _rows(names):
    return [
        {
            "name": n,
            "composition": f"Hoat chat {n.lower()}",
            "uses": f"Giam trieu chung {n.lower()}",
            "side_effects": "Buon non",
            "manufacturer": "Hang gia",
            "source": "test-fake",
            "is_authoritative": False,
        }
        for n in names
    ]


@pytest.fixture(scope="module")
def big_repo():
    repo = SQLiteRepository(":memory:")
    repo.add_medicines(_rows(NAMES))
    yield repo
    repo.close()


def test_danh_muc_du_lon(big_repo):
    assert big_repo.list_medicines(1, 0).total == len(NAMES) >= 1500


@pytest.mark.parametrize("idx", [0, 150, 900, 1727])
def test_thuoc_o_moi_vi_tri_deu_tim_duoc(big_repo, idx):
    """Thuốc sau vị trí 100 (theo tên) trước đây không bao giờ được truy hồi."""
    name = sorted(NAMES)[idx]
    question = f"{name} dung de lam gi"
    result = answer_question(question, big_repo.candidate_medicines(question))
    assert result.answered is True, result.refusal_reason
    assert any(c.medicine_name == name for c in result.citations)


def test_cach_cu_chi_100_thuoc_dau_bo_sot(big_repo):
    """Chứng minh lỗi cũ: lấy 100 thuốc đầu thì thuốc ở cuối danh mục bị từ chối."""
    name = sorted(NAMES)[1727]
    question = f"{name} dung de lam gi"
    old = answer_question(question, big_repo.list_medicines(100, 0).results)
    assert old.answered is False


def test_ung_vien_gioi_han_va_on_dinh(big_repo):
    cand = big_repo.candidate_medicines("hoat chat giam trieu chung", limit=50)
    assert len(cand) == 50
    assert cand == big_repo.candidate_medicines("hoat chat giam trieu chung", limit=50)


def test_cau_hoi_vo_nghia_khong_co_ung_vien(big_repo):
    assert big_repo.candidate_medicines("") == []
    assert big_repo.candidate_medicines("xyzkhongtontai") == []


def test_cong_cu_explain_tim_duoc_thuoc_o_cuoi_danh_muc(big_repo):
    name = sorted(NAMES)[1700]
    result = ExplainForPatientTool().run({"query": f"{name} dung de lam gi"}, big_repo, user_id=1)
    assert result.ok is True
    assert any(c.medicine_name == name for c in result.citations)


def test_chi_muc_tu_lam_moi_khi_chinh_no_ghi():
    repo = SQLiteRepository(":memory:")
    try:
        repo.add_medicines(_rows(NAMES[:10]))
        assert repo.candidate_medicines(f"{NAMES[0]} la gi")
        assert repo.candidate_medicines("Newdrugzz la gi") == []
        repo.add_medicines(_rows(["Newdrugzz"]))
        assert repo.candidate_medicines("Newdrugzz la gi"), "phải thấy thuốc vừa thêm"
        repo.delete_medicines_by_source("test-fake")
        assert repo.candidate_medicines("Newdrugzz la gi") == []
    finally:
        repo.close()


def test_chi_muc_tu_lam_moi_khi_tien_trinh_khac_nap_them(tmp_path):
    """Server đang chạy, bạn chạy import_real_data ở cửa sổ khác -> không cần khởi động lại."""
    db = tmp_path / "shared.db"
    server = SQLiteRepository(db)
    try:
        server.add_medicines(_rows(NAMES[:5]))
        assert server.candidate_medicines("Lateadd la gi") == []
        other = SQLiteRepository(db)  # "tiến trình khác"
        other.add_medicines(_rows(["Lateadd"]))
        other.close()
        assert server.candidate_medicines("Lateadd la gi"), "phải thấy dữ liệu do tiến trình khác nạp"
    finally:
        server.close()
