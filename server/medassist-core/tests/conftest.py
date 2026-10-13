"""Fixture và helper dùng chung cho test.

Cố ý **không** import `conftest` từ file test khác — pytest khuyến nghị dùng
fixture. `register_user`/`auth_header` được gọi trực tiếp trong test, và `auth`
fixture gói sẵn thứ test thường cần.
"""

from __future__ import annotations

import os
from typing import Any

# ⚠️ Đặt khoá ký **TRƯỚC** khi import bất kỳ module `app.*` — `app.main` gọi
# `validate_config()` lúc nạp nên nếu thiếu biến này thì import sẽ ném
# `ConfigError`. Test cần khoá riêng, không dùng khoá thật của máy dev.
os.environ.setdefault(
    "MEDASSIST_SECRET_KEY", "test-only-secret-key-do-not-use-in-production-0123456789"
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.deps import get_repository, reset_repository  # noqa: E402
from app.main import app  # noqa: E402
from app.rate_limit import reset_limiter  # noqa: E402
from app.sqlite_repository import SQLiteRepository  # noqa: E402

DEFAULT_PASSWORD = "matkhau123"


@pytest.fixture()
def repo() -> SQLiteRepository:
    """Kho in-memory mới cho mỗi test (không đụng đĩa thật)."""
    r = SQLiteRepository(":memory:")
    reset_repository()
    # Bộ đếm rate limit là **singleton toàn tiến trình**; không xoá thì số lượt
    # thử của test trước làm test sau đỡ 429 một cách ngẫu nhiên.
    reset_limiter()
    app.dependency_overrides[get_repository] = lambda: r
    yield r
    app.dependency_overrides.clear()
    reset_repository()
    reset_limiter()


@pytest.fixture()
def client(repo) -> TestClient:
    """Client FastAPI.

    ⚠️ Cố ý **phụ thuộc** fixture `repo`: nếu không, các test không dùng kho sẽ
    dùng kho singleton của tiến trình và **dữ liệu rò giữa các test** (ví dụ đăng
    ký trùng tên ở test sau vì test trước đã tạo tài khoản đó).
    """
    return TestClient(app)


@pytest.fixture()
def seeded_client(repo) -> TestClient:
    """Client đã **có sẵn danh mục thuốc dev**.

    Chỉ dùng cho test endpoint cần dữ liệu (ví dụ `/v1/agent/ask`). Fixture
    `client` cố ý để kho rỗng — test không nên phụ thuộc dữ liệu nạp sẵn.
    """
    from app.ingest import load_csv, to_rows

    repo.add_medicines(to_rows(load_csv()))
    return TestClient(app)


def register_user(
    client: TestClient,
    username: str = "antoan",
    password: str = DEFAULT_PASSWORD,
    email: str | None = None,
) -> dict[str, Any]:
    """Đăng ký và trả về body JSON của response."""
    response = client.post(
        "/v1/auth/register",
        json={"username": username, "password": password, "email": email or f"{username}@e.co"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def login(client: TestClient, username: str, password: str = DEFAULT_PASSWORD):
    return client.post("/v1/auth/login", json={"username": username, "password": password})


def token_for(client: TestClient, username: str) -> str:
    """Đăng ký rồi lấy token — gọn cho test chỉ cần token."""
    return register_user(client, username)["token"]["access_token"]


@pytest.fixture()
def auth(client: TestClient):
    """Công cụ xác thực cho test: `auth.token('an')` rồi `auth.header(tok)`."""

    class _Auth:
        @staticmethod
        def token(username: str, password: str = DEFAULT_PASSWORD) -> str:
            return token_for(client, username)

        @staticmethod
        def header(token: str) -> dict[str, str]:
            return {"Authorization": f"Bearer {token}"}

    return _Auth()