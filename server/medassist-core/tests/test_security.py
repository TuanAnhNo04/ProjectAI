"""Test BẢO MẬT — phần quan trọng nhất của server.

Theo `AGENTS.md` mục 3: server thật phải hash mật khẩu, dùng truy vấn có tham số,
và **phân quyền theo `user_id` trên mọi endpoint lịch sử**.

Các test ở đây cố ý "đóng vai kẻ tấn công": biết trước id của người khác rồi thử
truy cập. Nếu chỉ test luồng hợp lệ thì lỗi IDOR sẽ lọt qua.
"""

from __future__ import annotations

from app.security import hash_password, verify_password
from conftest import login, register_user

# ---------------------------------------------------------------- Mật khẩu ---


def test_mat_khau_luu_dang_hash(repo, client):
    """Trong DB không được có plaintext."""
    register_user(client, "antoan", "matkhau123")

    stored = repo.get_password_hash("antoan")
    assert stored is not None
    assert "matkhau123" not in stored
    assert stored.startswith("$2")  # bcrypt


def test_response_khong_bao_mat_khau(client):
    """Không response nào được chứa mật khẩu hay hash."""
    body = register_user(client, "antoan", "matkhau123")
    dumped = str(body)

    assert "matkhau123" not in dumped
    assert "password" not in body.get("user", {})
    assert "$2" not in dumped  # không lộ bcrypt hash


def test_hash_khong_giong_nhau_cho_cung_mat_khau():
    """bcrypt tự sinh salt -> 2 lần băm cùng mật khẩu phải khác nhau."""
    assert hash_password("matkhau123") != hash_password("matkhau123")


def test_verify_password_dung_sai():
    hashed = hash_password("matkhau123")

    assert verify_password("matkhau123", hashed) is True
    assert verify_password("matkhau124", hashed) is False
    # Hash hỏng -> coi như sai, KHÔNG ném lỗi 500.
    assert verify_password("matkhau123", "khong-phai-bcrypt") is False


# ------------------------------------------------------------- Xác thực -----


def test_dang_ky_thanh_cong(client):
    body = register_user(client, "antoan", "matkhau123")

    assert body["user"]["username"] == "antoan"
    assert body["token"]["access_token"]
    assert body["token"]["token_type"] == "bearer"
    # `register_user` đã assert 201 rồi.


def test_dang_ky_trung_ten(client):
    """Tên trùng phải trả 409 (không phải 500 do vi phạm UNIQUE)."""
    register_user(client, "antoan", "matkhau123")

    response = client.post(
        "/v1/auth/register",
        json={"username": "antoan", "password": "matkhau456", "email": "an@e.co"},
    )

    assert response.status_code == 409
    assert response.json()["code"] == "conflict"


def test_dang_nhap_sai_mat_khau_bi_tu_choi(client):
    register_user(client, "antoan", "matkhau123")

    response = client.post(
        "/v1/auth/login", json={"username": "antoan", "password": "sai-mat-khau"}
    )

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_dang_nhap_sai_ten_va_sai_mat_khau_bao_cao_giong_nhau(client):
    """Không để kẻ tấn công dò ra tài khoản nào tồn tại."""
    register_user(client, "antoan", "matkhau123")

    sai_ten = client.post(
        "/v1/auth/login", json={"username": "khong-ton-tai", "password": "matkhau123"}
    )
    sai_mk = client.post(
        "/v1/auth/login", json={"username": "antoan", "password": "sai-mat-khau"}
    )

    assert sai_ten.json()["message"] == sai_mk.json()["message"]


def test_mat_khau_ngang_den(client):
    """Server phải chặn mật khẩu quá ngắn (không trả 500)."""
    response = client.post(
        "/v1/auth/register",
        json={"username": "antoan", "password": "abc", "email": "a@b.co"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_loi_validation_khong_leak_gia_tri_da_gui(client):
    """Lỗi 422 không được phản chiếu lại mật khẩu của người dùng."""
    response = client.post(
        "/v1/auth/register",
        json={"username": "ab", "password": "matkhau-bi-lo", "email": "x"},
    )

    assert "matkhau-bi-lo" not in str(response.json())


# ------------------------------------------------- Phân quyền / IDOR ------


def _token(client, username: str) -> str:
    return register_user(client, username, "matkhau123")["token"]["access_token"]


def test_lich_su_can_token(client):
    assert client.get("/v1/history").status_code == 401


def test_token_garbage_bi_tu_choi(client, auth):
    response = client.get("/v1/history", headers=auth.header("khong-phai-token-that"))

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_user_a_khong_thay_lich_su_cua_user_b(client, repo, auth):
    """**IDOR**: A không thấy dữ liệu của B — kể cả khi biết đúng id bản ghi."""
    token_a = _token(client, "user_a")
    token_b = _token(client, "user_b")

    for path in ("b1.jpg", "b2.jpg"):
        assert (
            client.post(
                "/v1/history", json={"filepath": path}, headers=auth.header(token_b)
            ).status_code
            == 201
        )

    # A thấy danh sách **rỗng**.
    a_view = client.get("/v1/history", headers=auth.header(token_a)).json()
    assert a_view["history"] == []
    assert a_view["total"] == 0

    # A **không xoá được** bản ghi của B (dù đã biết id chính xác).
    victim_id = repo.list_history(repo.get_user_by_username("user_b").id, 10, 0)[0].id
    denied = client.delete(f"/v1/history/{victim_id}", headers=auth.header(token_a))
    assert denied.status_code == 404

    # Dữ liệu của B vẫn còn nguyên.
    assert client.get("/v1/history", headers=auth.header(token_b)).json()["total"] == 2


def test_user_a_va_b_co_lich_su_rieng(client, auth):
    """Mỗi người chỉ thấy đúng phần của mình."""
    token_a = _token(client, "user_a")
    token_b = _token(client, "user_b")

    client.post("/v1/history", json={"filepath": "a.jpg"}, headers=auth.header(token_a))
    client.post("/v1/history", json={"filepath": "b1.jpg"}, headers=auth.header(token_b))
    client.post("/v1/history", json={"filepath": "b2.jpg"}, headers=auth.header(token_b))

    a_files = [
        h["filepath"]
        for h in client.get("/v1/history", headers=auth.header(token_a)).json()["history"]
    ]
    b_files = [
        h["filepath"]
        for h in client.get("/v1/history", headers=auth.header(token_b)).json()["history"]
    ]

    assert a_files == ["a.jpg"]
    assert sorted(b_files) == ["b1.jpg", "b2.jpg"]


def test_user_xoa_duoc_ban_ghi_cua_ruot(client, auth):
    """Chủ sở hữu thì xoá được."""
    token = _token(client, "antoan")
    client.post("/v1/history", json={"filepath": "a.jpg"}, headers=auth.header(token))

    item_id = client.get("/v1/history", headers=auth.header(token)).json()["history"][0]["id"]
    assert client.delete(f"/v1/history/{item_id}", headers=auth.header(token)).status_code == 204
    assert client.get("/v1/history", headers=auth.header(token)).json()["total"] == 0


# ------------------------------------------------- SQL injection -----------


def test_sql_injection_khong_thoat_ra_kho(client, repo):
    """Tên đăng nhập chứa SQL injection không được phá vỡ câu truy vấn."""
    response = client.post(
        "/v1/auth/login",
        json={"username": "an'; DROP TABLE users; --", "password": "matkhau123"},
    )

    assert response.status_code == 401
    # Bảng `users` phải còn nguyên -> kho vẫn truy vấn được.
    assert repo.get_user_by_username("antoan") is None


def test_sql_injection_trong_search(client, repo):
    """`%` và `'` trong từ khoá tìm kiếm không được nội suy vào SQL."""
    repo.add_medicines([{"name": "Paracetamol 500mg"}])

    response = client.get("/v1/search", params={"name": "'; DROP TABLE medicines; --"})

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert client.get("/v1/search", params={"name": "para"}).json()["total"] == 1