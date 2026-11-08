# MedAssist

> Ứng dụng tra cứu thông tin thuốc: chụp ảnh hộp thuốc để nhận diện, tìm theo tên,
> và xem lại lịch sử các lần đã quét — **có hoạt động offline** và **có disclaimer
> y khoa ở mọi màn kết quả**.

⚠️ **Thông tin trong app chỉ mang tính tham khảo, không thay thế chẩn đoán hoặc đơn
thuốc của bác sĩ/dược sĩ.**

---

## Mục tiêu

| Hiện tại (v0.1.0) | Sắp tới |
|---|---|
| Nhận diện thuốc từ ảnh (gọi server Flask) | Nhận diện **on-device** bằng ML Kit (Day 26) |
| Tìm thuốc theo tên | Agent hỏi đáp có trích dẫn nguồn (P2, Day 15→24) |
| Lịch sử quét ảnh | Lịch sử theo tài khoản, đồng bộ nhiều thiết bị |
| Đăng nhập + token bảo mật | Backend FastAPI + JWT thật, phân quyền theo `user_id` |

## Kiến trúc

**Edge (Flutter) – Core (backend + agent) – Provider (LLM thay thế được)**, theo
ports & adapters, contract-first, offline-first.

```
lib/
├── domain/ports/     HỢP ĐỒNG thuần Dart — không import Flutter (có test canh)
├── data/
│   ├── models/       Model kiểu hoá, JSON key nằm trong hằng số
│   ├── adapters/     Cài adapter HTTP thật (Flask) + kho token bảo mật
│   └── local/        Cache offline (shared_preferences — KHÔNG chứa bí mật)
├── core/
│   ├── config/       Cấu hình qua --dart-define, validate lúc khởi động
│   ├── network/      ApiClient (Dio) + AppException + Result<T>
│   ├── providers.dart  ĐIỂM LẮP RÁP DUY NHẤT (Riverpod)
│   └── theme/        Material 3 light/dark
├── screens/          Giao diện — chỉ biết port, không biết Flask tồn tại
└── widgets/          Disclaimer y khoa, banner ngoại tuyến
```

**Nguyên tắc bất di bất dịch:** màn hình **không** tự gọi mạng. Đổi backend =
sửa đúng một chỗ trong `core/providers.dart`. Điều này có test canh hồi quy.

## Chạy dự án

```powershell
flutter pub get --offline          # không gọi mạng
flutter gen-l10n                   # sinh code dịch (vi/en) — chạy lại khi sửa .arb
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000
```

> `10.0.2.2` là alias của `localhost` **trong Android Emulator**. Khi chạy trên máy
> thật hoặc iOS Simulator, dùng `http://127.0.0.1:5000` hoặc IP LAN của máy.

### Biến môi trường (`--dart-define`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `API_BASE_URL` | `http://10.0.2.2:5000` | Địa chỉ server. **Bắt buộc khi chạy thật.** |
| `API_PREFIX` | *(rỗng)* | Tiền tố path, ví dụ `/api/v1`. |
| `BACKEND` | `core` | `core` = `medassist-core` (FastAPI, `/v1/...`); `flask` = server Flask cũ / `tools/mock_server`. |
| `ANALYZER` | `ondevice` (core) / `remote` (flask) | `remote` \| `ondevice` \| `hybrid`. |

**Chạy với backend `medassist-core` trên máy ảo Android** (khoá/dữ liệu GIẢ, chỉ localhost):

```powershell
# cửa sổ 1 — server
$env:MEDASSIST_SECRET_KEY = "<chuỗi-ngẫu-nhiên-≥32-ký-tự>"
$env:MEDASSIST_DB = "$env:TEMP\medassist-dev.db"
cd server\medassist-core ; python -m app.seed
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# cửa sổ 2 — app (10.0.2.2 = máy host nhìn từ máy ảo)
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000 --dart-define=BACKEND=core
```


> ⚠️ **Bản build phát hành phải dùng HTTPS.** `AppConfig.validate()` báo lỗi nếu
> cấu hình sai; app không tự động nới lỏng kiểm tra này.

### Server giả lập (không cần backend thật)

```powershell
python tools/mock_server/app.py    # chạy ở localhost:5000
flutter test tools/mock_server/live_check_test.dart --dart-define=API_BASE_URL=http://127.0.0.1:5000
```

> Mock server **chỉ** phục vụ `localhost`, **không** bao giờ mở ra LAN/Internet,
> và không được dùng dữ liệu thật.

## Kiểm thử

```powershell
flutter analyze          # kỳ vọng: No issues found!
flutter test             # kỳ vọng: All tests passed (268 test)
flutter test --coverage  # + coverage/lcov.info
```

Chi tiết độ phủ theo từng tầng và các bài học khi viết test:
**[`docs/TESTING.md`](docs/TESTING.md)**.

## Bảo mật

Nếu bạn đọc hoặc đóng góp cho dự án, hãy đọc **[`AGENTS.md`](AGENTS.md)** trước — đó là
bộ quy tắc bắt buộc (không rò rỉ dữ liệu sức khoẻ, không gọi mạng ngoài
localhost, không thêm package, không commit bằng `git add -A`…).

Các nguyên tắc đã hiện thực hoá:

| Nguyên tắc | Cách làm |
|---|---|
| Token bảo mật | `flutter_secure_storage` (Keychain / Keystore), **không** dùng `shared_preferences` |
| Không log bí mật | `AuthInterceptor.redactHeaders/redactBody` che `Authorization`, `password`, `token` |
| 401 hết phiên | Xoá token + đưa về màn Login, xoá sạch stack điều hướng |
| Dữ liệu offline | Chỉ cache dữ liệu công khai; token **không** nằm trong cache |
| Disclaimer y khoa | Luôn hiện ở cả 3 màn ra kết quả, không có tuỳ chọn tắt |
| Không bịa nguồn y khoa | Mọi nội dung thuốc đến từ server có nguồn; app không tự sinh kết luận |

## Tài liệu

| File | Nội dung |
|---|---|
| [`AGENTS.md`](AGENTS.md) | **Quy tắc bảo mật bắt buộc** — đọc trước khi làm việc |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Kế hoạch theo ngày + nhật ký thực thi từng ngày |
| [`docs/TESTING.md`](docs/TESTING.md) | Độ phủ test và bài học |
| [`docs/SETUP_WINDOWS.md`](docs/SETUP_WINDOWS.md) | Cài JDK, Android SDK, checklist môi trường |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Xử lý các lỗi môi trường đã gặp thật |

## Công nghệ

Flutter 3.19 · Dart 3.3 · Riverpod 2.4 · Dio 5.4 · flutter_secure_storage 9.2 ·
shared_preferences 2.2 · gen-l10n (vi/en)
