"""Bước 1 bổ sung — fail-fast secret (N2) + giới hạn byte mật khẩu (N3).

Bổ sung cho Day 16–17, ghi ở `docs/DECISIONS.md`.
"""

from __future__ import annotations

import os

import pytest

from app import jwt_tokens
from app.jwt_tokens import (
    MIN_SECRET_BYTES,
    SECRET_ENV_VAR,
    ConfigError,
    decode,
    encode,
    load_secret,
)
from app.security import (
    BCRYPT_ROUNDS,
    MAX_PASSWORD_BYTES,
    PasswordTooLongError,
    hash_password,
    verify_password,
)
from conftest import register_user


@pytest.fixture(autouse=True)
def _restore_key_cache():
    """Mỗi test có khoá sạch -> không sót cache từ test trước."""
    jwt_tokens.reset_key_cache()
    yield
    jwt_tokens.reset_key_cache()


# ============================== (b) N2: secret phải fail-fast =================


def test_thieu_secret_thi_server_tu_choi_khoi_dong(monkeypatch):
    """Không có `MEDASSIST_SECRET_KEY` -> ném `ConfigError`, KHÔNG fallback."""
    monkeypatch.delenv(SECRET_ENV_VAR, raising=False)
    with pytest.raises(ConfigError, match=SECRET_ENV_VAR):
        load_secret()


def test_secret_qua_ngan_thi_tu_choi(monkeypatch):
    """Secret ngắn hơn 32 byte -> từ chối (yêu cầu của bạn)."""
    monkeypatch.setenv(SECRET_ENV_VAR, "short-key")
    with pytest.raises(ConfigError, match="32"):
        load_secret()


def test_secret_dai_vua_ban_thi_duoc_nhan(monkeypatch):
    monkeypatch.setenv(SECRET_ENV_VAR, "x" * MIN_SECRET_BYTES)
    assert load_secret() == b"x" * MIN_SECRET_BYTES


def test_secret_31_byte_thi_bi_tu_choi(monkeypatch):
    monkeypatch.setenv(SECRET_ENV_VAR, "x" * (MIN_SECRET_BYTES - 1))
    with pytest.raises(ConfigError):
        load_secret()


def test_validate_config_la_fail_fast(monkeypatch):
    """`validate_config()` (gọi lúc khởi động) cũng phải ném."""
    monkeypatch.delenv(SECRET_ENV_VAR, raising=False)
    with pytest.raises(ConfigError):
        jwt_tokens.validate_config()


def test_server_that_khoi_dong_duoc_khi_co_secret():
    """Sanity: cấu hình đúng thì `validate_config()` không ném."""
    assert os.environ.get(SECRET_ENV_VAR)  # conftest đã đặt
    jwt_tokens.validate_config()  # không ném


def test_token_doi_khoa_thi_khong_duoc_chap_nhan(monkeypatch):
    """Token ký bằng khoá khác phải bị từ chối -> chứng minh khoá thật sự dùng."""
    token = jwt_tokens.encode(
        {"sub": 1, "type": "access", "exp": 9999999999, "iat": 0}
    )
    monkeypatch.setenv(SECRET_ENV_VAR, "y" * MIN_SECRET_BYTES)
    jwt_tokens.reset_key_cache()

    assert jwt_tokens.decode(token) is None


def test_error_khong_lo_le_mat_khau():
    """Thông báo lỗi không được chứa mật khẩu (quy tắc 3 của AGENTS.md)."""
    with pytest.raises(PasswordTooLongError) as exc:
        hash_password("A" * 100)
    assert "A" * 100 not in str(exc.value)


# ========================== (c) N3: giới hạn byte mật khẩu ==================


def test_mat_khau_qua_72_byte_bi_tu_choi():
    """bcrypt cắt cụt 72 byte -> ta **từ chối** thay vì cắt (quyết định N3)."""
    with pytest.raises(PasswordTooLongError):
        hash_password("A" * (MAX_PASSWORD_BYTES + 1))


def test_hai_mat_khau_khac_nhau_khong_cung_duoc_chap_nhan():
    """Ca lỗ hổng cắt cụt: 72 byte đầu giống nhau phải cho KẾT QUẢ KHÁC NHAU."""
    prefix = "A" * MAX_PASSWORD_BYTES
    with pytest.raises(PasswordTooLongError):
        hash_password(prefix + "X")
    with pytest.raises(PasswordTooLongError):
        hash_password(prefix + "Y")


def test_mat_khau_dung_72_byte_van_dung():
    digest = hash_password("A" * MAX_PASSWORD_BYTES)
    assert verify_password("A" * MAX_PASSWORD_BYTES, digest) is True


def test_verify_mat_khau_qua_dai_tra_false_khong_nem():
    """Đăng nhập với mật khẩu quá dài -> 401 chứ không 500."""
    digest = hash_password("matkhau123")
    assert verify_password("A" * 200, digest) is False


def test_mat_khau_rong_bi_tu_choi():
    with pytest.raises(ValueError):
        hash_password("")


def test_dung_bcrypt_rounds_it_nhat_12():
    assert BCRYPT_ROUNDS >= 12


def test_do_dai_tinh_theo_byte_chua_khong_theo_ky_tu():
    """Unicode nhiều byte vẫn được đo theo **byte**."""
    assert verify_password("Nguyễn Văn A", hash_password("Nguyễn Văn A"))
    # 50 ký tự "Ặ" = 100 byte UTF-8 -> phải bị từ chối.
    with pytest.raises(PasswordTooLongError):
        hash_password("Ặ" * 50)


def test_dang_ky_mat_khau_qua_dai_tra_422(client):
    """API trả 422 chứ không 500 (không để lộ lỗi nội bộ)."""
    body = {
        "username": "userlongpw",
        "password": "A" * 100,
        "email": "long@example.com",
    }
    assert client.post("/v1/auth/register", json=body).status_code == 422


def test_dang_nhap_mat_khau_qua_dai_tra_401(client):
    register_user(client, "userloginlong", "matkhau123", "long@example.com")
    resp = client.post(
        "/v1/auth/login",
        json={"username": "userloginlong", "password": "A" * 100},
    )
    assert resp.status_code == 401
# =================== (a) N1: thu hồi cả chuỗi token khi tái sử dụng =======


def _login(client, username="userfam", password="matkhau123"):
    reg = client.post(
        "/v1/auth/register",
        json={"username": username, "password": password, "email": f"{username}@e.com"},
    )
    return reg.json()["token"]["refresh_token"]


def _refresh(client, token):
    return client.post("/v1/auth/refresh", json={"refresh_token": token})


def test_tai_su_dung_token_da_xoay_bi_phat_hien(client):
    """Dùng lại token đã xoay -> 401."""
    first = _login(client)
    assert _refresh(client, first).status_code == 200
    assert _refresh(client, first).status_code == 401


def test_tai_su_dung_thu_hoi_ca_CHUOI_token(client):
    """⭐ Yêu cầu (a): tái sử dụng token đã xoay => thu hồi **cả family**.

    Kịch bản: nạn nhân đăng nhập -> xoay (T1 -> T2). Sau đó nạn nhân dùng lại
    T1 (đã bị thu hồi) => phát hiện tái sử dụng => **T2 cũng phải chết**.
    Nếu không, kẻ tấn công đã chôm T1 vẫn dùng được T2.
    """
    token1 = _login(client, "victim")

    # Nạn nhân xoay: T1 -> T2.
    token2 = _refresh(client, token1).json()["token"]["refresh_token"]

    # Nạn nhân (hoặc kẻ tấn công) dùng lại T1 -> phát hiện tái sử dụng.
    assert _refresh(client, token1).status_code == 401

    # 🔴 Chỗ cũ bị hởng: nếu token2 (cùng chuỗi) còn sống thì kẻ tấn công
    # vẫn dùng được. Test này chống lại đúng điều đó.
    assert _refresh(client, token2).status_code == 401, (
        "token cùng chuỗi phải chết khi phát hiện tái sử dụng"
    )


def test_ket_qua_khong_lo_le_chi_tinh_mot_nguoi_dung(client):
    """Thông báo lỗi không được nói "phát hiện tái sử dụng" (lộ chiến thuật)."""
    t = _login(client, "sneaky")
    _refresh(client, t)
    resp = _refresh(client, t)
    assert resp.status_code == 401
    assert "tái sử dụng" not in resp.text.lower()
    assert "reuse" not in resp.text.lower()


def test_thu_hoi_chuoi_khong_dinh_dong_phi_en_da_khac(client):
    """Người dùng khác đăng nhập ở máy khác -> phiên họ **không** bị ảnh hưởng."""
    a = _login(client, "userA")
    _login(client, "userB")

    _refresh(client, a)  # xoay hợp lệ
    assert _refresh(client, a).status_code == 401  # tái sử dụng -> thu hồi chuỗi A

    # Phiên B phải vẫn đăng nhập được.
    resp_b = client.post(
        "/v1/auth/login",
        json={"username": "userB", "password": "matkhau123"},
    )
    assert resp_b.status_code == 200


def test_cha_token_van_dung_duoc_sau_khi_xoay(client):
    """Chuỗi chỉ chết khi **tái sử dụng**, không chết khi xoay bình thường."""
    t1 = _login(client, "userok")
    t2 = _refresh(client, t1).json()["token"]["refresh_token"]
    t3 = _refresh(client, t2).json()["token"]["refresh_token"]
    assert _refresh(client, t3).status_code == 200


def test_revoke_family_chi_dung_family_do(repo):
    """Unit test trực tiếp `revoke_token_family`.

    ⚠️ Phải tạo user thật vì `refresh_tokens.user_id` có khoá ngoại.
    """
    from app.security import hash_password

    u1 = repo.create_user("famA", hash_password("matkhau123"), "a@e.com")
    u2 = repo.create_user("famB", hash_password("matkhau123"), "b@e.com")

    repo.save_refresh_token("h1", u1.id, 9_999_999_999, "fam-A")
    repo.save_refresh_token("h2", u1.id, 9_999_999_999, "fam-A")
    repo.save_refresh_token("h3", u2.id, 9_999_999_999, "fam-B")

    assert repo.revoke_token_family("fam-A") == 2
    assert repo.count_active_family_tokens("fam-A") == 0
    assert repo.count_active_family_tokens("fam-B") == 1


def test_family_rong_khong_lam_gi(repo):
    """Token cũ (family_id rỗng) không gây lỗi."""
    assert repo.revoke_token_family("") == 0
    assert repo.count_active_family_tokens("") == 0


def test_logout_va_tai_su_dung_van_the_quy_tac(client):
    """Sau logout, dùng lại token cũng phải 401."""
    t = _login(client, "userlogout")
    client.post("/v1/auth/logout", json={"refresh_token": t})
    assert _refresh(client, t).status_code == 401

# ========================================== 8. N4: dùng PyJWT, tương thích chuẩn


def test_dung_thu_vien_pyjwt_that():
    """⚠️ Phải là `PyJWT`, không phải bản tự cài tay (nợ N4 đã đóng 05/10/2026)."""
    import jwt

    from app import jwt_tokens

    # Token do hệ thống phát phải **PyJWT đọc được**.
    # ⚠️ `verify_sub=False` vì `sub` của ta là **int** theo thiết kế (xem docstring
    # `app/jwt_tokens.py`) — đây là lệch RFC 7519 đã biết, không phải lỗi.
    token = jwt_tokens.encode({"sub": 1, "type": "access", "exp": 9999999999, "iat": 0})
    decoded = jwt.decode(
        token, jwt_tokens.get_key(), algorithms=["HS256"], options={"verify_sub": False}
    )
    assert decoded["sub"] == 1
    assert decoded["type"] == "access"


def test_khong_con_ham_ma_thu_con_thu():
    """Không được tự dựng lại HS256 bằng `hmac` — đó chính là nợ N4."""
    import inspect

    from app import jwt_tokens

    source = inspect.getsource(jwt_tokens)
    assert "import hmac" not in source
    assert "_b64url_encode" not in source


def test_token_co_du_ba_phan_chuan_rfc7519():
    """Định dạng JWT chuẩn: header.payload.signature."""
    token = encode({"sub": 1, "type": "access", "exp": 9999999999, "iat": 0})
    assert token.count(".") == 2
    assert not token.endswith("="), "base64url không có padding"


def test_alg_none_bi_tu_choi():
    """⭐ Tấn công kinh điển: header `alg: none`, không chữ ký."""
    import base64
    import json

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"sub": 1, "type": "access", "exp": 9999999999}).encode())
    assert decode(f"{header}.{payload}.") is None
    assert decode(f"{header}.{payload}") is None


def test_alg_khong_trong_allowlist_bi_tu_choi():
    """Token ký bằng thuật toán khác không được chấp nhận."""
    import base64
    import hashlib
    import hmac
    import json

    from app.jwt_tokens import ALGORITHM, get_key

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    header = b64(json.dumps({"alg": "HS512", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"sub": 1, "type": "access", "exp": 9999999999}).encode())
    sig = hmac.new(get_key(), f"{header}.{payload}".encode(), hashlib.sha512).digest()

    assert decode(f"{header}.{payload}.{b64(sig)}") is None
    assert ALGORITHM == "HS256"


def test_payload_vo_bi_tu_choi():
    assert decode("") is None
    assert decode("abc") is None
    assert decode("a.b") is None
    assert decode("a.b.c.d") is None
    assert decode("...") is None


def test_token_het_han_bi_tu_choi():
    token = encode({"sub": 1, "type": "access", "exp": 1, "iat": 0})
    assert decode(token) is None
    # `verify_exp=False` là đường thoát duy nhất — dùng khi kiểm tra token hết hạn.
    assert decode(token, verify_exp=False) is not None


def test_sub_phai_la_int():
    """`sub` là `user_id` — sai kiểu thì bị từ chối (chặn lọt số qua token)."""
    assert decode(encode({"sub": "1", "exp": 9999999999})) is None
    assert decode(encode({"sub": 1.5, "exp": 9999999999})) is None
    assert decode(encode({"sub": True, "exp": 9999999999})) is None
    assert decode(encode({"exp": 9999999999})) is None


def test_token_khong_co_exp_bi_tu_choi():
    """Thiếu `exp` -> không xác định hạn -> từ chối, không cho token vô hạn."""
    assert decode(encode({"sub": 1})) is None


def test_chu_ky_ba_cat_bi_tu_choi():
    """Chữ ký bị sửa 1 ký tự -> từ chối."""
    token = encode({"sub": 1, "type": "access", "exp": 9999999999, "iat": 0})
    head, body, sig = token.split(".")
    flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
    assert decode(f"{head}.{body}.{flipped}") is None


def test_doi_khoa_thi_token_cu_khong_dung_duoc():
    """Đổi `MEDASSIST_SECRET_KEY` -> mọi token cũ mất hiệu lực (đúng như thiết kế)."""
    token = encode({"sub": 1, "type": "access", "exp": 9999999999, "iat": 0})
    jwt_tokens.reset_key_cache()
    os.environ[SECRET_ENV_VAR] = "mot-khoa-khac-dai-tam-32-byte-0123456789"
    assert decode(token) is None
