# HƯỚNG DẪN RELEASE (Day 30) — việc cần CHỦ DỰ ÁN làm

> Những bước dưới đây **không thể** làm hộ: cần máy có Flutter/JDK, khoá ký của riêng bạn, hoặc quyết định
> của bạn. Việc phía code đã chuẩn bị xong được đánh dấu ✅.

## 0. Điều kiện máy

Máy dev hiện **không có** Flutter SDK, JDK, Docker (đã kiểm tra 24/10/2026). Cài và kiểm tra:

```powershell
flutter doctor -v          # Android toolchain phải ✓
java -version              # JDK 17
```

## 1. Kiểm tra chất lượng trước khi build

```powershell
flutter pub get
dart format --output=none --set-exit-if-changed .
flutter analyze
flutter test
cd server\medassist-core ; python -m pytest -q ; python eval.py
```

⚠️ Từ Day 25 đến nay (chat streaming, OCR, hồ sơ sức khoẻ, nhắc thuốc) và thay đổi Day 29
(xoá cache khi đăng xuất) **chưa từng được chạy `flutter analyze/test`** vì thiếu SDK. Hãy chạy
bước này trước tiên và sửa lỗi nếu có.

## 2. Tạo khoá ký (làm trên máy của bạn, KHÔNG gửi cho ai)

```powershell
keytool -genkey -v -keystore $env:USERPROFILE\secrets\medassist-release.jks `
  -keyalg RSA -keysize 2048 -validity 10000 -alias medassist-release
copy android\key.properties.example android\key.properties   # rồi điền giá trị THẬT
```

- ✅ `android/app/build.gradle` đã đọc `android/key.properties` và ký bản release bằng khoá đó.
- ✅ `android/.gitignore` đã chặn `key.properties`, `*.jks`, `*.keystore`.
- Thiếu `key.properties` thì Gradle cảnh báo và rơi về khoá debug (không nộp được Google Play).
- **Sao lưu keystore ở nơi an toàn.** Mất khoá = không cập nhật được app trên Google Play.

## 3. Build bản release

Từ 2026-10-28 có 3 flavor (`android/app/build.gradle`) cài song song được; **mọi lệnh run/build cần `--flavor`**:

| Flavor | applicationId | Nhãn app |
|---|---|---|
| `dev` | `com.medassist.app.dev` | MedAssist Dev |
| `staging` | `com.medassist.app.staging` | MedAssist Staging |
| `prod` | `com.medassist.app` | MedAssist |

```powershell
flutter run --flavor dev --dart-define=APP_FLAVOR=dev --dart-define=API_BASE_URL=http://10.0.2.2:8000
flutter build appbundle --release --flavor prod --dart-define=APP_FLAVOR=prod --dart-define=API_BASE_URL=https://<server-https-cua-ban>
```

✅ Đã kiểm tra trên máy này: `.aab` prod release build được (R8 bật, cần `android/app/proguard-rules.pro` —
đừng xoá: thiếu thì R8 lỗi vì ML Kit, và nhắc thuốc crash vì Gson), APK release cài + chạy được trên AVD. Chưa ký
bằng khoá thật (chưa có `key.properties`).

- Bản release **bắt buộc HTTPS** (không có `network_security_config` cleartext ngoài thư mục `debug/`).
- Server thật cần: `MEDASSIST_SECRET_KEY` ≥ 32 byte (đặt qua biến môi trường), HTTPS (reverse proxy), CSDL
  mã hoá, và nên đặt `MEDASSIST_HISTORY_RETENTION_DAYS` (xem `docs/PRIVACY.md`).
- Chạy server thật qua môi trường riêng; **không** dùng `tools/mock_server` (chỉ localhost, dữ liệu giả).

## 4. Cần bạn QUYẾT ĐỊNH (AGENTS.md yêu cầu hỏi trước)

| Việc | Vì sao chưa làm | Tác động nếu không làm |
|---|---|---|
| ✅ Nhắc thuốc đúng giờ (duyệt 2026-10-28) | `timezone` + quyền `POST_NOTIFICATIONS`, `SCHEDULE_EXACT_ALARM`, `RECEIVE_BOOT_COMPLETED` đã thêm; lịch dựng lại mỗi lần mở app | ✅ Đã thử thật trên máy ảo (28/10/2026): thông báo nổ đúng 08:00; 7 báo thức/tuần. Chưa thử trên điện thoại thật |
| ✅ Icon/splash | Dùng adaptive icon vector thuần Android (không thêm `flutter_launcher_icons`/`flutter_native_splash`); icon tạm (dấu thập trắng nền teal) — thay bằng thiết kế của bạn khi có | Android 12+ dùng splash hệ thống với icon |
| ✅ Flavor | Tên/ID mặc định ở bảng mục 3 — đổi nếu bạn muốn | — |
| ⛔ Crash reporting | **Chưa làm**: cần bạn chọn SDK (Sentry/Crashlytics) + cung cấp DSN/cấu hình (là bí mật, tôi không được đọc), bắt buộc có consent trong UI, che dữ liệu trước khi gửi, và cập nhật `PRIVACY.md` (`AGENTS.md` mục 2) | Không có báo lỗi từ máy người dùng |

## 5. Tự rà các package đã thêm từ Day 26–28

Các package sau được thêm trong giai đoạn P3. `AGENTS.md` yêu cầu chủ dự án duyệt trước; log thực thi không ghi
rõ đã duyệt hay chưa — **hãy xác nhận lại**:

- Dart: `google_mlkit_text_recognition`, `path_provider`, `sqflite`, `flutter_local_notifications`
- Python: `mcp[cli]`, `httpx-sse` (server MCP; chỉ chạy localhost)

## 6. Bước cuối

```powershell
git add <từng-file-cụ-thể>      # không dùng `git add -A` / `git add .`
git diff --staged               # rà trước khi commit
git commit -m "..."
git tag v1.0.0                  # chỉ khi bạn đã duyệt
```

Ghi chú: toàn bộ thay đổi từ Day 9 đến nay **vẫn chưa commit** (xem nhật ký trong `ROADMAP.md`).
