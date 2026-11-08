# Triển khai `medassist-core` lên Railway

> Chủ dự án tự làm các bước dưới đây (kể cả `git push`). Không dán khoá API, mật khẩu
> hay token vào chat, code, issue hay ảnh chụp màn hình.

## 0. Đã chuẩn bị sẵn trong repo

| File | Vai trò |
|---|---|
| `server/medassist-core/Dockerfile` | Build image; trong container nghe `0.0.0.0:$PORT` (Railway cấp `PORT`). Chạy local vẫn dùng `127.0.0.1`. |
| `server/medassist-core/railway.json` | Build bằng Dockerfile, healthcheck `GET /v1/health`, tự khởi động lại khi lỗi. |
| `server/medassist-core/.dockerignore` | Không đưa `.env*`, `data/real/`, `*.db`, test vào image. |
| `server/medassist-core/.env.example` | Danh sách biến môi trường, **giá trị giả**. |

Khi khởi động, server **từ chối chạy** nếu `MEDASSIST_SECRET_KEY` thiếu hoặc < 32 byte; với
`MEDASSIST_ENV=production` còn từ chối khoá mẫu/yếu. Nếu CSDL rỗng, server nạp danh mục
thuốc **mẫu** (`MEDASSIST_SEED_IF_EMPTY=1`, đặt sẵn trong Dockerfile).

## 1. Tạo project

1. Đăng nhập Railway → **New Project** → **Deploy from GitHub repo** → chọn repo của dự án.
2. Vào service vừa tạo → **Settings** → **Root Directory** = `server/medassist-core`.
   Railway sẽ đọc `railway.json` và build bằng `Dockerfile`.

## 2. Gắn Volume (lưu CSDL qua các lần deploy)

Service → **Volumes** → **New Volume** → Mount path: `/data`.

   ⚠️ Quyền ghi của ổ `/data` (chưa kiểm chứng, Dockerfile chưa từng được build thử): container chạy bằng
   user thường `app` (không phải root, theo bản sửa B-04) nhưng ổ đĩa Railway có thể được gắn với quyền
   root → server không tạo được `/data/medassist.db` (lỗi `unable to open database file` / `Permission denied`
   trong log lần deploy đầu). Nếu gặp, theo tài liệu hiện hành của Railway thêm biến `RAILWAY_RUN_UID=0`
   (chạy container bằng root) hoặc chỉnh quyền ổ đĩa; hãy tự kiểm tra lại tên biến trên trang tài liệu
   của Railway trước khi dùng. Sau khi deploy, gọi `/v1/health` và đăng ký thử một tài khoản giả để chắc
   chắn CSDL ghi được.

## 3. Biến môi trường (Service → Variables)

| Biến | Giá trị |
|---|---|
| `MEDASSIST_SECRET_KEY` | Khoá ngẫu nhiên MỚI, tạo trên máy bạn: `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `MEDASSIST_ENV` | `production` |
| `MEDASSIST_DB` | `/data/medassist.db` |
| `MEDASSIST_LLM` | `claude` (hoặc `mock` nếu chưa muốn bật AI đám mây) |
| `ANTHROPIC_API_KEY` | Một khoá **MỚI** tạo riêng cho bản deploy này trong Anthropic Console. Không dùng lại khoá đã từng dán vào chat/code. |
| `MEDASSIST_CLAUDE_MODEL` | Tuỳ chọn; mặc định `claude-sonnet-5-5` |

Không cần đặt `PORT` — Railway tự cấp.

## 4. Tên miền HTTPS

Service → **Settings** → **Networking** → **Generate Domain**. Railway cấp
`https://<ten>.up.railway.app` với chứng chỉ TLS sẵn — không cần cấu hình HTTPS trong app.

Kiểm tra: mở `https://<domain>/v1/health` → `{"status": "ok", ...}`.

## 5. Build app Flutter trỏ tới server

```powershell
flutter build apk --release --dart-define=API_BASE_URL=https://<domain>
```

Bản release chỉ dùng HTTPS; không bật cleartext, không tắt kiểm tra chứng chỉ.

## 6. Ghi chú bảo mật

- **Dữ liệu sức khoẻ nằm trên điện thoại.** Server chỉ lưu lâu dài: tài khoản (mật khẩu
  băm bcrypt) + danh mục thuốc. Thuốc đang dùng, bệnh nền, dị ứng, chủ đề gần đây chỉ được
  dùng trong một request, không lưu CSDL, không ghi log.
- **Phiên chat chỉ ở RAM** — mất khi restart/redeploy; người dùng xoá được bất kỳ lúc nào.
- **Claude chỉ được gọi khi đủ hai cổng:** server bật `MEDASSIST_LLM=claude` + có khoá, VÀ
  người dùng đã đồng ý trong app (`cloud_ai_consent=true` cho từng request). Câu hỏi được
  khử định danh trước khi gửi. Chi tiết: `docs/PRIVACY.md` mục 6.1.
- **Xoay vòng khoá:** đổi `MEDASSIST_SECRET_KEY` (mọi phiên đăng nhập sẽ hết hiệu lực) và
  `ANTHROPIC_API_KEY` định kỳ, và **ngay lập tức** nếu nghi lộ. Thu hồi khoá cũ trong
  Anthropic Console.
- **Không** dùng giá trị trong `.env.example` hay `docker-compose.yml` cho production.
- **Chuyển dữ liệu ra nước ngoài:** máy chủ Railway và Anthropic đặt ngoài Việt Nam. Dữ
  liệu tài khoản (email) và câu hỏi đã khử định danh vì thế được xử lý ở nước ngoài — cần
  đánh giá nghĩa vụ chuyển dữ liệu cá nhân xuyên biên giới (hồ sơ đánh giá tác động, thông
  báo/đồng ý của chủ thể dữ liệu) theo **Nghị định 356/2025/NĐ-CP** trước khi phát hành cho
  người dùng thật. Với đồ án/demo: chỉ dùng tài khoản và dữ liệu giả.
- Danh mục thuốc seed là **dữ liệu mẫu** (`is_authoritative=false`) — không dùng cho tư vấn thật.
