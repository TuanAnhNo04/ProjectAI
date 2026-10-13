# medassist-core

Backend FastAPI cho app Flutter MedAssist. Xem [`../../AGENTS.md`](../../AGENTS.md)
trước khi đóng góp — đó là quy tắc bảo mật bắt buộc.

⚠️ **Thông tin thuốc chỉ mang tính tham khảo, không thay thế chẩn đoán hoặc đơn
thuốc của bác sĩ.**

## Chạy

```powershell
cd server/medassist-core
uvicorn app.main:app --reload --port 8000
```

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/v1/health

> ⚠️ Server này chứa dữ liệu sức khoẻ và tài khoản — **chỉ phục vụ `localhost`**.

## Test

```powershell
cd server/medassist-core
python -m pytest
```

Chạy offline hoàn toàn: kho SQLite **in-memory**, `TestClient` không mở socket ra
ngoài.

## Endpoint

| Method | Đường dẫn | Auth | Mô tả |
|---|---|---|---|
| GET | `/v1/health` | công khai | Kiểm tra server sống |
| POST | `/v1/auth/register` | công khai | Tạo tài khoản → token |
| POST | `/v1/auth/login` | công khai | Đăng nhập → token |
| POST | `/v1/auth/validate-password` | công khai | Kiểm tra độ mạnh mật khẩu |
| GET | `/v1/search` | công khai | Tìm thuốc (không dấu/hoa thường đều được) |
| GET | `/v1/history` | **bắt buộc** | Lịch sử **của người đang đăng nhập** |
| POST | `/v1/history` | **bắt buộc** | Ghi bản ghi lịch sử |
| DELETE | `/v1/history/{id}` | **bắt buộc** | Xoá bản ghi **của mình** |
| POST | `/v1/analyze` | **bắt buộc** | Placeholder — chưa gắt mô hình (trả 501) |

### Format lỗi thống nhất

Mọi lỗi — kể cả lỗi do FastAPI/Pydantic sinh ra — đều có cùng hình dạng:

```json
{ "code": "validation_error", "message": "Dữ liệu gửi lên không hợp lệ.", "detail": [] }
```

`code` là mã ổn định để client xử lý logic (`unauthorized`, `forbidden`,
`not_found`, `validation_error`, `conflict`, `rate_limited`, `server_error`).

### Xuất đặc tả OpenAPI

```powershell
python -m app.main          # ghi openapi.json
```

## Bảo mật đã hiện thực hoá

| Quy tắc | Cách làm |
|---|---|
| Hash mật khẩu | **bcrypt**, không bao giờ lưu plaintext (`app/security.py`) |
| Không rò rỉ bí mật | Response không có `password`/`password_hash`; lỗi 422 **không** phản chiếu giá trị đã gửi |
| Chống SQL injection | **Mọi** truy vấn dùng tham số `?`/`:name`, không ghép chuỗi |
| Phân quyền theo `user_id` | `user_id` lấy từ token, **không** nhận từ query/body → chặn IDOR |
| Không dò tài khoản | Sai tên và sai mật khẩu trả **cùng** một thông báo |
| So sánh chữ ký chống timing | `hmac.compare_digest` |
| Dữ liệu thuốc có nguồn | Mỗi kết quả có `source` + `updated_at` |

## ⚠️ Giới hạn đã biết (đọc trước khi dùng)

1. **Máy dev KHÔNG có Docker.** Do đó:
   - `docker-compose.yml` và `Dockerfile` **được viết nhưng CHƯA BAO GIỜ CHẠY**;
   - DoD "`docker compose up` chạy" của ROADMAP **chưa đạt được** trên máy này.
   - Toàn bộ Day 15 được kiểm chứng bằng SQLite in-memory + pytest.
2. **Kho là SQLite (stdlib), chưa phải Postgres/pgvector.** Interface
   `Repository` trong `app/repository.py` đã sẵn sàng: khi có Postgres chỉ cần
   viết `PostgresRepository` mà **không sửa route nào**.
3. ~~**Token KHÔNG phải JWT chuẩn** (HMAC ký `base64(payload).base64(sig)`)~~
   → **ĐÃ SỬA (nợ N4, 05/10/2026)**: dùng **`PyJWT`**, token đúng chuẩn
   RFC 7519, mọi thư viện JWT khác đều đọc được. Cài `pip install -r requirements.txt`.
4. ~~**Khoá ký sinh ngẫu nhiên theo tiến trình** nếu không đặt
   `MEDASSIST_SECRET_KEY`~~ → **ĐÃ SỬA (quyết định N2)**: `MEDASSIST_SECRET_KEY`
   giờ là **bắt buộc**, phải dài **≥ 32 byte**. Thiếu hoặc ngắn thì server
   **từ chối khởi động** (`ConfigError`), không còn fallback.
   → Tạo khoá: `python -c "import secrets; print(secrets.token_urlsafe(48))"`
5. **Chưa có rate limit** → Day 16 thêm (cần Redis).
6. **`/v1/analyze` trả 501** — chưa gắt mô hình nhận diện. Endpoint này
   **không** gửi ảnh ra ngoài. Nếu sau này gọi LLM/OCR bên ngoài thì bắt buộc
   phải có **sự đồng ý người dùng + khử định danh + `docs/PRIVACY.md`** theo
   `AGENTS.md` mục 2.

## Cấu trúc

```
app/
├── main.py           dựng app, gắn router, xuất openapi.json
├── errors.py         format lỗi {code, message, detail}
├── schemas.py        Pydantic — hợp đồng với app Flutter
├── repository.py     interface Repository (port)
├── sqlite_repository.py  adapter SQLite (chỉ stdlib)
├── security.py       bcrypt + token
├── deps.py           get_repository, current_user_id
└── routers/          health · auth · data
tests/                pytest (offline)
```