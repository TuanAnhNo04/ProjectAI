"""Test `/v1/health` và hợp đồng OpenAPI."""

from __future__ import annotations


def test_health_ok(client):
    response = client.get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "medassist-core"
    assert "time" in body


def test_health_khong_can_token(client):
    """Health phải công khai, nếu không app không ping được lúc khởi động."""
    assert client.get("/v1/health").status_code == 200


def test_openapi_co_dung_endpoint(client):
    """Spec phải liệt kê đủ endpoint đã hứa trong ROADMAP."""
    spec = client.get("/openapi.json").json()

    paths = spec["paths"]
    for expected in (
        "/v1/health",
        "/v1/auth/register",
        "/v1/auth/login",
        "/v1/search",
        "/v1/analyze",
        "/v1/history",
        "/v1/history/{item_id}",
    ):
        assert expected in paths, f"thiếu endpoint {expected}"


def test_export_openapi_tao_file(tmp_path, client):
    """`export_openapi` ghi được file — dùng để review hợp đồng API."""
    from app.main import export_openapi

    target = tmp_path / "openapi.json"
    written = export_openapi(target)

    assert written.exists()
    assert '"openapi"' in written.read_text(encoding="utf-8")


def test_loi_404_dung_format_thong_nhat(client):
    """Kể cả lỗi của FastAPI cũng phải ra `{code, message, detail}`."""
    response = client.get("/v1/khong-ton-tai")

    assert response.status_code == 404
    body = response.json()
    assert set(body) == {"code", "message", "detail"}
    assert body["code"] == "not_found"
