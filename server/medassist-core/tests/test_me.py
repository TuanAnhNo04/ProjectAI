"""`GET /v1/auth/me` — app khôi phục phiên và hiện tên người dùng.

Dữ liệu GIẢ. Endpoint chỉ trả dữ liệu của CHÍNH người gọi (id lấy từ token).
"""

from __future__ import annotations

from conftest import DEFAULT_PASSWORD, register_user


def _headers(body):
    return {"Authorization": f"Bearer {body['token']['access_token']}"}


def test_me_tra_ve_chinh_nguoi_goi(client):
    a = register_user(client, "annie")
    b = register_user(client, "binhbb")

    resp_a = client.get("/v1/auth/me", headers=_headers(a))
    resp_b = client.get("/v1/auth/me", headers=_headers(b))

    assert resp_a.status_code == 200 and resp_b.status_code == 200
    assert resp_a.json()["username"] == "annie"
    assert resp_b.json()["username"] == "binhbb"


def test_me_khong_lo_mat_khau_hay_hash(client):
    a = register_user(client, "annie")
    resp = client.get("/v1/auth/me", headers=_headers(a))
    assert "password" not in resp.json()
    assert "$2" not in resp.text
    assert DEFAULT_PASSWORD not in resp.text


def test_me_can_token(client):
    assert client.get("/v1/auth/me").status_code == 401
    bad = client.get("/v1/auth/me", headers={"Authorization": "Bearer khong-hop-le"})
    assert bad.status_code == 401


def test_me_sau_khi_xoa_tai_khoan_la_401(client):
    a = register_user(client, "annie")
    headers = _headers(a)
    deleted = client.post(
        "/v1/auth/delete-account", json={"password": DEFAULT_PASSWORD}, headers=headers
    )
    assert deleted.status_code == 204

    # Access token còn hạn nhưng tài khoản đã mất -> không được coi là còn phiên.
    assert client.get("/v1/auth/me", headers=headers).status_code == 401
