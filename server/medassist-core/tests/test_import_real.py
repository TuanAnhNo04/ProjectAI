"""Công cụ nạp dữ liệu thật (`app/import_real.py`) — kiểm thử bằng dữ liệu HOÀN TOÀN GIẢ.

Điểm quan trọng nhất: báo cáo chỉ chứa số liệu, KHÔNG chứa nội dung dòng.
"""

from __future__ import annotations

import io

import pytest

from app.ingest import IngestError, load_csv, load_csv_mapped, to_rows
from app.import_real import run
from app.sqlite_repository import SQLiteRepository

#: Chuỗi "canh" — nếu xuất hiện trong output nghĩa là nội dung dòng bị lộ.
SECRET_NAME = "ThuocGiaBiMat-Zeta9"
SECRET_USE = "CongDungGiaBiMat-Omega7"

CSV_HEADER = "Medicine_Name,Composition,Uses,Side_effects,Manufacturer\n"


def _write(tmp_path, body: str, header: str = CSV_HEADER, name: str = "thuoc.csv", bom: bool = False):
    path = tmp_path / name
    path.write_text(header + body, encoding="utf-8-sig" if bom else "utf-8")
    return path


def _run(tmp_path, *extra, csv_path=None, db=None, source="Nguon gia 2026"):
    out = io.StringIO()
    args = ["--csv", str(csv_path), "--source", source, *extra]
    if db is not None:
        args += ["--db", str(db)]
    code = run(args, out=out)
    return code, out.getvalue()


@pytest.fixture()
def csv_ok(tmp_path):
    return _write(
        tmp_path,
        f"{SECRET_NAME},Hoat chat A 500mg,{SECRET_USE},Buon non,Hang X\n"
        "Thuoc Giam Dau,Hoat chat B,Giam dau,,Hang Y\n"
        ",Hoat chat C,Khong ten,,\n"
        f"{SECRET_NAME},Hoat chat A 500mg,Dong trung,Buon non,Hang X\n",
    )


# ----------------------------------------------------------- đọc & báo cáo


def test_bao_cao_dem_dung(csv_ok):
    drugs, report = load_csv_mapped(csv_ok, source="Nguon gia")
    assert report.rows_read == 4
    assert report.imported == 2
    assert report.skipped_no_name == 1
    assert report.duplicates == 1
    assert [d.source for d in drugs] == ["Nguon gia", "Nguon gia"]
    assert all(d.is_authoritative is False for d in drugs)


def test_o_rong_la_none_khong_tu_bia(csv_ok):
    drugs, _ = load_csv_mapped(csv_ok, source="x")
    giam_dau = next(d for d in drugs if d.name == "Thuoc Giam Dau")
    assert giam_dau.side_effects is None


def test_khong_loai_trung_khi_keep_duplicates(csv_ok):
    _, report = load_csv_mapped(csv_ok, source="x", dedupe=False)
    assert report.imported == 3 and report.duplicates == 0


def test_file_co_bom_excel_van_doc_duoc(tmp_path):
    path = _write(tmp_path, "Thuoc A,Hoat chat,Cong dung,,\n", bom=True)
    drugs, report = load_csv_mapped(path, source="x")
    assert report.imported == 1 and drugs[0].name == "Thuoc A"


def test_cot_khac_ten_dung_anh_xa(tmp_path):
    path = _write(
        tmp_path,
        "Thuoc A,Hoat chat A,Cong dung A\n",
        header="Ten thuoc,Hoat chat,Cong dung\n",
    )
    drugs, report = load_csv_mapped(
        path,
        {"name": "Ten thuoc", "composition": "Hoat chat", "uses": "Cong dung"},
        source="x",
    )
    assert drugs[0].uses == "Cong dung A"
    assert "side_effects=Side_effects" in report.missing_columns


def test_thieu_cot_ten_la_loi_ro_rang(tmp_path):
    path = _write(tmp_path, "a,b\n", header="Cot1,Cot2\n")
    with pytest.raises(IngestError) as exc:
        load_csv_mapped(path, source="x")
    assert "Medicine_Name" in str(exc.value)


def test_thieu_source_la_loi(csv_ok):
    with pytest.raises(IngestError):
        load_csv_mapped(csv_ok, source="  ")


def test_file_khong_ton_tai(tmp_path):
    with pytest.raises(IngestError):
        load_csv_mapped(tmp_path / "khong-co.csv", source="x")


def test_o_qua_dai_bi_cat_va_dem(tmp_path):
    path = _write(tmp_path, "Thuoc A,HC," + "x" * 5000 + ",,\n")
    drugs, report = load_csv_mapped(path, source="x", max_field_len=100)
    assert report.truncated_fields == 1 and len(drugs[0].uses) == 100


def test_load_csv_mau_van_khong_authoritative():
    assert all(d.is_authoritative is False for d in load_csv())


# ----------------------------------------------------------- lệnh (CLI)


def test_chay_thu_khong_ghi_gi(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    code, out = _run(tmp_path, csv_path=csv_ok, db=db)
    assert code == 0 and "CHẠY THỬ" in out
    assert not db.exists()


def test_output_khong_lo_noi_dung_dong(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    _, dry = _run(tmp_path, csv_path=csv_ok, db=db)
    _, applied = _run(tmp_path, "--apply", csv_path=csv_ok, db=db)
    for text in (dry, applied):
        assert SECRET_NAME not in text
        assert SECRET_USE not in text


def test_apply_ghi_dung_va_gan_nguon(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    code, out = _run(tmp_path, "--apply", csv_path=csv_ok, db=db)
    assert code == 0 and "ĐÃ NẠP 2" in out
    repo = SQLiteRepository(db)
    try:
        assert repo.medicine_counts_by_source() == {"Nguon gia 2026": (2, 0)}
    finally:
        repo.close()


def test_authoritative_can_nguoi_kiem_duyet(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    code, out = _run(tmp_path, "--apply", "--authoritative", csv_path=csv_ok, db=db)
    assert code == 2 and "--reviewed-by" in out
    assert not db.exists()


def test_authoritative_ghi_ten_nguoi_kiem_duyet_vao_nguon(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    code, _ = _run(
        tmp_path, "--apply", "--authoritative", "--reviewed-by", "DS Nguyen Van A",
        csv_path=csv_ok, db=db,
    )
    assert code == 0
    repo = SQLiteRepository(db)
    try:
        (src, (total, auth)), = repo.medicine_counts_by_source().items()
        assert "DS Nguyen Van A" in src and total == auth == 2
    finally:
        repo.close()


def test_nhan_nguon_khong_duoc_gia_dang_du_lieu_mau(tmp_path, csv_ok):
    code, out = _run(tmp_path, csv_path=csv_ok, source="dev-sample: gia mao")
    assert code == 2 and "dev-sample" in out


def test_nap_lai_cung_nguon_khong_nhan_doi(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    _run(tmp_path, "--apply", csv_path=csv_ok, db=db)
    _run(tmp_path, "--apply", "--replace-source", csv_path=csv_ok, db=db)
    repo = SQLiteRepository(db)
    try:
        assert repo.medicine_counts_by_source() == {"Nguon gia 2026": (2, 0)}
    finally:
        repo.close()


def test_khong_replace_thi_nhan_doi(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    _run(tmp_path, "--apply", csv_path=csv_ok, db=db)
    _run(tmp_path, "--apply", csv_path=csv_ok, db=db)
    repo = SQLiteRepository(db)
    try:
        assert repo.medicine_counts_by_source() == {"Nguon gia 2026": (4, 0)}
    finally:
        repo.close()


def test_go_du_lieu_mau_chi_xoa_dev_sample(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    repo = SQLiteRepository(db)
    repo.add_medicines(to_rows(load_csv()))  # 10 thuốc mẫu
    repo.close()
    code, out = _run(tmp_path, "--apply", "--drop-dev-sample", csv_path=csv_ok, db=db)
    assert code == 0 and "Đã gỡ thuốc mẫu   : 10" in out
    repo = SQLiteRepository(db)
    try:
        assert repo.medicine_counts_by_source() == {"Nguon gia 2026": (2, 0)}
    finally:
        repo.close()


def test_sao_luu_truoc_khi_ghi(tmp_path, csv_ok):
    db = tmp_path / "x.db"
    repo = SQLiteRepository(db)
    repo.add_medicines(to_rows(load_csv()))
    repo.close()
    _run(tmp_path, "--apply", csv_path=csv_ok, db=db)
    assert list(tmp_path.glob("x.db.bak-*")), "phải có bản sao lưu"


def test_khong_ghi_vao_memory(tmp_path, csv_ok, monkeypatch):
    monkeypatch.delenv("MEDASSIST_DB", raising=False)
    code, out = _run(tmp_path, "--apply", csv_path=csv_ok)
    assert code == 2 and "MEDASSIST_DB" in out


def test_col_sai_dinh_dang(tmp_path, csv_ok):
    code, out = _run(tmp_path, "--col", "tenthuoc", csv_path=csv_ok)
    assert code == 2 and "--col sai" in out


def test_xoa_theo_nguon_rong_bi_tu_choi(tmp_path):
    repo = SQLiteRepository(tmp_path / "x.db")
    try:
        with pytest.raises(ValueError):
            repo.delete_medicines_by_source("  ")
    finally:
        repo.close()
