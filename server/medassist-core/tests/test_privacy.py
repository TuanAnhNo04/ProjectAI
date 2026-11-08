"""Day 29 — quyền riêng tư: xoá tài khoản, retention, audit log.

Toàn bộ dữ liệu ở đây là GIẢ. Test hai người dùng A/B để chắc chắn xoá A không
đụng tới B (cùng tinh thần test IDOR ở `test_security.py`).
"""

from __future__ import annotations

import logging

from app.observability import get_usage_tracker, reset_usage_tracker
from app.schemas import MedicineResponse
from app.sessions import get_session_store, reset_session_store
from conftest import DEFAULT_PASSWORD, login, register_user


def _med(name: str = "Thuoc Gia") -> MedicineResponse:
    return MedicineResponse(name=name)


def _delete_account(client, token: str, password: str = DEFAULT_PASSWORD):
    return client.post(
        "/v1/auth/delete-account",
        json={"password": password},
        headers={"Authorization": f"Bearer {token}"},
    )


def _setup_two_users(client, repo):
    a = register_user(client, "annie")
    b = register_user(client, "binhbb")
    repo.add_history(a["user"]["id"], "a.jpg", [_med("A")])
    repo.add_history(b["user"]["id"], "b.jpg", [_med("B")])
    return a, b


# ----------------------------------------------------------- xoá tài khoản ---


def test_xoa_tai_khoan_xoa_sach_du_lieu_cua_chinh_nguoi_do(client, repo):
    a, b = _setup_two_users(client, repo)
    reset_session_store()
    get_session_store().append(a["user"]["id"], "s1", "user", "cau hoi gia")
    reset_usage_tracker()
    get_usage_tracker().record(a["user"]["id"], "/v1/agent/chat", 10, 10)

    resp = _delete_account(client, a["token"]["access_token"])

    assert resp.status_code == 204
    assert repo.get_user_by_username("annie") is None
    assert repo.count_history(a["user"]["id"]) == 0
    assert get_session_store().history(a["user"]["id"], "s1") == []
    assert get_usage_tracker().get(a["user"]["id"])["requests"] == 0


def test_xoa_tai_khoan_a_khong_dung_toi_b(client, repo):
    a, b = _setup_two_users(client, repo)

    assert _delete_account(client, a["token"]["access_token"]).status_code == 204

    assert repo.get_user_by_username("binhbb") is not None
    assert repo.count_history(b["user"]["id"]) == 1
    assert login(client, "binhbb").status_code == 200


def test_sau_khi_xoa_khong_dang_nhap_hay_refresh_duoc(client, repo):
    a, _ = _setup_two_users(client, repo)
    refresh = a["token"]["refresh_token"]

    assert _delete_account(client, a["token"]["access_token"]).status_code == 204

    assert login(client, "annie").status_code == 401
    r = client.post("/v1/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 401


def test_xoa_tai_khoan_sai_mat_khau_khong_xoa_gi(client, repo):
    a, _ = _setup_two_users(client, repo)

    resp = _delete_account(client, a["token"]["access_token"], "sai-mat-khau")

    assert resp.status_code == 401
    assert repo.get_user_by_username("annie") is not None
    assert repo.count_history(a["user"]["id"]) == 1


def test_xoa_tai_khoan_can_token(client, repo):
    _setup_two_users(client, repo)
    resp = client.post("/v1/auth/delete-account", json={"password": DEFAULT_PASSWORD})
    assert resp.status_code == 401


def test_xoa_tai_khoan_bi_gioi_han_toc_do_doan_mat_khau(client, repo):
    a, _ = _setup_two_users(client, repo)
    token = a["token"]["access_token"]

    codes = [_delete_account(client, token, f"sai{i}").status_code for i in range(8)]

    assert codes[:5] == [401] * 5
    assert 429 in codes[5:]
    assert repo.get_user_by_username("annie") is not None


def test_phan_hoi_xoa_khong_lo_thong_tin(client, repo):
    a, _ = _setup_two_users(client, repo)
    resp = _delete_account(client, a["token"]["access_token"], "sai-mat-khau")
    assert "sai-mat-khau" not in resp.text
    assert "$2" not in resp.text


# --------------------------------------------------------------- retention ---


def test_retention_chi_xoa_lich_su_cu_hon_han(repo):
    user = repo.create_user("annie", "hash", "an@e.co")
    old = repo.add_history(user.id, "old.jpg", [_med()])
    new = repo.add_history(user.id, "new.jpg", [_med()])
    repo._conn.execute(  # noqa: SLF001
        "UPDATE history SET created_at = ? WHERE id = ?",
        ("2020-01-01T00:00:00+00:00", old.id),
    )
    repo._conn.commit()  # noqa: SLF001

    removed = repo.purge_history_older_than(days=30)

    assert removed == 1
    remaining = {h.id for h in repo.list_history(user.id, 10, 0)}
    assert remaining == {new.id}


def test_retention_khong_hop_le_khong_xoa_gi(repo):
    user = repo.create_user("annie", "hash", "an@e.co")
    repo.add_history(user.id, "x.jpg", [_med()])

    # 0 hoặc âm = tắt retention, tuyệt đối không được hiểu thành "xoá hết".
    assert repo.purge_history_older_than(days=0) == 0
    assert repo.purge_history_older_than(days=-5) == 0
    assert repo.count_history(user.id) == 1


def test_cau_hinh_retention_doc_tu_moi_truong(monkeypatch):
    from app.privacy import retention_days_from_env

    monkeypatch.delenv("MEDASSIST_HISTORY_RETENTION_DAYS", raising=False)
    assert retention_days_from_env() == 0
    monkeypatch.setenv("MEDASSIST_HISTORY_RETENTION_DAYS", "90")
    assert retention_days_from_env() == 90
    # Giá trị rác -> tắt, không ném lỗi làm server không khởi động.
    monkeypatch.setenv("MEDASSIST_HISTORY_RETENTION_DAYS", "abc")
    assert retention_days_from_env() == 0


# --------------------------------------------------------------- audit log ---


def test_audit_chi_ghi_truong_an_toan(caplog):
    from app.observability import AUDIT_LOG_FIELDS, audit

    with caplog.at_level(logging.INFO, logger="medassist.audit"):
        audit("account_deleted", "ok")

    assert caplog.records
    record = caplog.records[-1]
    logged = {k for k in vars(record) if k in AUDIT_LOG_FIELDS}
    assert logged == {"trace_id", "action", "outcome"}
    assert record.action == "account_deleted"
    assert record.outcome == "ok"


def test_audit_tu_choi_hanh_dong_la():
    import pytest

    from app.observability import audit

    # Chuỗi tự do có thể chứa email/tên thuốc -> chỉ nhận danh sách hành động cố định.
    with pytest.raises(ValueError):
        audit("nguoi dung an@e.co da xoa panadol", "ok")


def test_xoa_tai_khoan_co_audit_nhung_khong_lo_ten_dang_nhap(client, repo, caplog):
    a, _ = _setup_two_users(client, repo)

    with caplog.at_level(logging.INFO, logger="medassist.audit"):
        _delete_account(client, a["token"]["access_token"])

    audit_records = [r for r in caplog.records if r.name == "medassist.audit"]
    assert [r.action for r in audit_records] == ["account_deleted"]
    dumped = " ".join(f"{k}={v}" for r in audit_records for k, v in vars(r).items())
    assert "an@e.co" not in dumped
    assert DEFAULT_PASSWORD not in dumped
    assert "antoan" not in dumped


# ------------------------------------------------------ log không lộ từ khoá ---


def test_access_log_cua_uvicorn_bi_tat_vi_chua_query_string():
    """`uvicorn.access` ghi `?name=panadol` (dữ liệu sức khoẻ) -> phải tắt (D4)."""
    import app.main  # noqa: F401  (nạp module là đủ để tắt logger)

    assert logging.getLogger("uvicorn.access").disabled is True
