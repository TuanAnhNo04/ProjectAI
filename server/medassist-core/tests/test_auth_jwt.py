"""Day 16–17 — JWT thật, refresh rotation, rate limit đăng nhập.

DoD của ngày: token hết hạn -> **refresh hoạt động**; truy cập dữ liệu người
khác -> 403/404 (đã test ở `test_security.py`).
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from app.jwt_tokens import (
    ACCESS_TTL_SECONDS,
    REFRESH_TTL_SECONDS,
    decode,
    encode,
    hash_token,
    issue_access_token,
    issue_refresh_token,
)
from app.rate_limit import InMemoryRateLimiter, RateLimitExceeded
from conftest import login, register_user

# ------------------------------------------------------------- JWT thuần ----


def test_jwt_co_3_phan():
    token, _ = issue_access_token(7)
    assert token.count(".") == 2


def test_decode_dung_token():
    token, _ = issue_access_token(7)
    payload = decode(token)

    assert payload is not None
    assert payload["sub"] == 7
    assert payload["type"] == "access"


def test_token_het_han_bi_tu_choi():
    token, _ = issue_access_token(7, ttl_seconds=-1)

    assert decode(token) is None
    # Bỏ qua kiểm tra hạn thì vẫn đọc được -> lỗi là do `exp`, không phải chữ ký.
    assert decode(token, verify_exp=False) is not None


def test_chu_ky_bi_xao_hoi_bi_tu_choi():
    token, _ = issue_access_token(7)
    header, payload, signature = token.split(".")

    assert decode(f"{header}.{payload}.{signature[:-2]}xx") is None


def test_thay_doi_payload_bi_phat_hien():
    """Đổi `sub` trong payload nhưng giữ chữ ký cũ -> phải bị từ chối."""
    token, _ = issue_access_token(7)
    header, _, signature = token.split(".")

    forged = encode({"sub": 999, "type": "access", "exp": int(time.time()) + 600})
    forged_payload = forged.split(".")[1]

    assert decode(f"{header}.{forged_payload}.{signature}") is None


def test_alg_none_bi_tu_choi():
    """Token tự khai `alg: none` không được chấp nhận."""

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"sub": 1, "type": "access", "exp": 9999999999}).encode())

    assert decode(f"{header}.{payload}.") is None


def test_token_rac_dinh_bi_tu_choi():
    assert decode("khong-phai-jwt") is None
    assert decode("a.b") is None
    assert decode("") is None


def test_tao_va_kiem_phan_giay_hash():
    """Chỉ lưu hash trong DB, không lưu token thô."""
    token, _ = issue_refresh_token(7)
    assert hash_token(token) != token
    assert len(hash_token(token)) == 64  # sha256 hex


def test_dang_ky_tra_ca_hai_loai_token(client):
    body = register_user(client, "antoan", "matkhau123")

    assert body["token"]["access_token"]
    assert body["token"]["refresh_token"]
    assert body["token"]["expires_in"] == ACCESS_TTL_SECONDS
    assert body["token"]["refresh_expires_in"] == REFRESH_TTL_SECONDS


def test_refresh_token_luu_dang_hash_trong_kho(client, repo):
    """Trong DB **không** được có token thô."""
    refresh = register_user(client, "antoan", "matkhau123")["token"]["refresh_token"]
    assert repo.get_refresh_token(hash_token(refresh)) is not None

    stored = repo._conn.execute(  # noqa: SLF001
        "SELECT token_hash FROM refresh_tokens"
    ).fetchall()
    assert all(r["token_hash"] != refresh for r in stored)


def test_access_token_het_han_roi_refresh_hoat_dong(client):
    """DoD Day 16–17: token hết hạn -> refresh lấy được token mới."""
    session = register_user(client, "antoan", "matkhau123")
    refresh_token = session["token"]["refresh_token"]
    auth_header = {"Authorization": f"Bearer {session['token']['access_token']}"}

    # Access token ban đầu vẫn dùng được.
    assert client.get("/v1/history", headers=auth_header).status_code == 200

    # Token hết hạn -> 401
    expired, _ = issue_access_token(session["user"]["id"], ttl_seconds=-1)
    blocked = client.get("/v1/history", headers={"Authorization": f"Bearer {expired}"})
    assert blocked.status_code == 401

    # ...nhưng refresh vẫn chạy được và cấp token mới.
    renewed = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert renewed.status_code == 200
    new_access = renewed.json()["token"]["access_token"]
    assert new_access != expired
    assert (
        client.get("/v1/history", headers={"Authorization": f"Bearer {new_access}"}).status_code
        == 200
    )


def test_refresh_token_dung_lan_2_bi_thu_hoi(client):
    """Rotation: refresh token cũ chết sau lần dùng đầu."""
    refresh_token = register_user(client, "antoan", "matkhau123")["token"]["refresh_token"]

    first = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert first.status_code == 200
    assert client.post("/v1/auth/refresh", json={"refresh_token": refresh_token}).status_code == 401


def test_access_token_khong_dung_duoc_cho_refresh(client):
    """Chặn nhầm loại token: access token không được đổi ở /refresh."""
    access = register_user(client, "antoan", "matkhau123")["token"]["access_token"]

    assert client.post("/v1/auth/refresh", json={"refresh_token": access}).status_code == 401


def test_logout_thu_hoi_phien(client):
    refresh_token = register_user(client, "antoan", "matkhau123")["token"]["refresh_token"]

    logout = client.post("/v1/auth/logout", json={"refresh_token": refresh_token})
    assert logout.status_code == 204
    after = client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert after.status_code == 401


def test_refresh_token_rac_bi_tu_choi(client):
    register_user(client, "antoan", "matkhau123")
    response = client.post("/v1/auth/refresh", json={"refresh_token": "a.b.c"})

    assert response.status_code == 401
    assert "không hợp lệ" in response.json()["message"]


# ---------------------------------------------------------- Rate limit -----


def test_dang_nhap_qua_nhieu_lan_bi_chan(client):
    """Sai mật khẩu 5 lần -> lần 6 bị 429 (chặn dò mật khẩu)."""
    register_user(client, "antoan", "matkhau123")

    for _ in range(5):
        login(client, "antoan", "sai-mat-khau")

    blocked = login(client, "antoan", "sai-mat-khau")
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "rate_limited"
    assert blocked.json()["detail"]["retry_after_seconds"] >= 1


def test_rate_limit_khong_an_huong_nguoi_dung_dung_mat_khau(client):
    """Đăng nhập đúng phải xoá bộ đếm — không phạt nhầm người dùng thật."""
    register_user(client, "antoan", "matkhau123")

    for _ in range(4):
        login(client, "antoan", "sai-mat-khau")

    # Lần 5 nhập đúng -> thành công và reset bộ đếm.
    assert login(client, "antoan", "matkhau123").status_code == 200
    # Sau đó sai 5 lần nữa vẫn chưa bị chặn.
    for _ in range(5):
        assert login(client, "antoan", "sai").status_code == 401


def test_rate_limit_tach_theo_ten_dang_nhap():
    """Tài khoản khác nhau không dùng chung bộ đếm."""
    limiter = InMemoryRateLimiter(max_attempts=2, window_seconds=60)

    limiter.check("login:1.1.1:an")
    limiter.check("login:1.1.1:binh")

    # `an` đã dùng 1 lượt; `binh` dùng lượt thứ 2 vẫn được.
    limiter.check("login:1.1.1:an")
    limiter.check("login:1.1.1:binh")

    # Lượt thứ 3 của mỗi tài khoản -> cả hai đều vượt hạn.
    with pytest.raises(RateLimitExceeded):
        limiter.check("login:1.1.1:an")
    with pytest.raises(RateLimitExceeded):
        limiter.check("login:1.1.1:binh")


def test_rate_limit_reset():
    limiter = InMemoryRateLimiter(max_attempts=1, window_seconds=60)
    limiter.check("k")

    with pytest.raises(RateLimitExceeded):
        limiter.check("k")

    limiter.reset("k")
    limiter.check("k")  # không ném nữa