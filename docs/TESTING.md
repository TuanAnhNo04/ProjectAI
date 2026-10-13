# KIỂM THỬ & ĐỘ PHỦ — MedAssist (Milestone M1)

> **Ngày chốt số liệu:** 03/10/2026 · **Trạng thái:** ✅ `flutter analyze` sạch · ✅ `flutter test` **268/268 xanh**
> Số liệu dưới đây **đo thật** bằng `flutter test --coverage` và đọc từ `coverage/lcov.info` — không ước lượng.

---

## 1. Chạy test

```powershell
flutter pub get --offline     # không gọi mạng (mọi dep đã nẵm trong cache)
flutter test                  # chạy toàn bộ
flutter test --coverage       # + sinh coverage/lcov.info
flutter analyze               # phải ra "No issues found!"
```

> **Vì sao luôn `--offline`?** Quy tắc bảo mật của dự án (`AGENTS.md`) cấm gọi
> pub.dev. Mọi dependency đã được tải sẵn nên `pub get` thường không cần mạng —
> thêm `--offline` để **bảo đảm** điều đó.

## 2. Tổng quan

| Chỉ số | Giá trị |
|---|---|
| Số file test | **24** |
| Số test case | **268** |
| Thời gian chạy | ~8 giây |
| Phủ mã `lib/` | **80.4%** (1225/1523 dòng) |
| Phủ mã `lib/data/` | **96.2%** (333/346) ← DoD yêu cầu ≥ 50% |

### Phủ theo tầng

| Tầng | Đã phủ | Tỉ lệ | Nhận xét |
|---|---|---|---|
| `lib/data` | 333/346 | **96.2%** | Adapter + model gần như kín |
| `lib/core` | 394/448 | **87.9%** | Còn `main.dart` ở 56.5% (khó test) |
| `lib/domain` | 12/16 | **75.0%** | `agent_client` — chưa có adapter thật (Day 22) |
| `lib/screens` | 368/503 | **73.2%** | `process_image_screen` 40% (cần chọn ảnh thật) |
| `lib/widgets` | 42/42 | **100%** | Kín |

### Phủ chi tiết từng file

| File | Phủ | % |
|---|---|---|
| `lib/core/network/result.dart` | 29/29 | 100% |
| `lib/core/network/api_exception.dart` | 80/81 | 98.8% |
| `lib/core/theme/app_theme.dart` | 10/11 | 90.9% |
| `lib/core/config/app_config.dart` | 29/33 | 87.9% |
| `lib/core/providers.dart` | 152/173 | 87.9% |
| `lib/core/network/api_client.dart` | 94/121 | 77.7% |
| `lib/data/adapters/remote_analyzer.dart` | 8/8 | 100% |
| `lib/data/adapters/remote_auth_repository.dart` | 33/33 | 100% |
| `lib/data/adapters/remote_medicine_repository.dart` | 12/12 | 100% |
| `lib/data/adapters/secure_token_store.dart` | 10/11 | 90.9% |
| `lib/data/local/cache_store.dart` | 45/46 | 97.8% |
| `lib/data/models/json_utils.dart` | 40/40 | 100% |
| `lib/data/models/user.dart` | 47/48 | 97.9% |
| `lib/data/models/medicine.dart` | 67/71 | 94.4% |
| `lib/data/models/analysis_result.dart` | 29/31 | 93.5% |
| `lib/data/models/history_item.dart` | 42/46 | 91.3% |
| `lib/domain/ports/agent_client.dart` | 12/16 | 75.0% |
| `lib/widgets/medical_disclaimer.dart` | 22/22 | 100% |
| `lib/widgets/offline_banner.dart` | 20/20 | 100% |
| `lib/screens/medicine_search_screen.dart` | 76/78 | 97.4% |
| `lib/screens/login_screen.dart` | 91/96 | 94.8% |
| `lib/screens/history_screen.dart` | 90/120 | 75.0% |
| `lib/screens/register_screen.dart` | 63/89 | 70.8% |
| `lib/screens/process_image_screen.dart` | 48/120 | 40.0% |
| `lib/main.dart` | 13/23 | 56.5% |
| `lib/l10n/generated/*` | 63/145 | 43.4% |

> File trong `lib/l10n/generated/` do `flutter gen-l10n` sinh ra — không cần test
> thủ công; phủ thấp ở `app_localizations_en.dart` là bình thường vì app chạy
> mặc định tiếng Việt.

## 3. Cách tổ chức test

```
test/
├── core/          config, network, provider (DI + controller)
├── data/          adapters (mock Dio), models, kho token, cache
├── domain/        port thuần Dart + TEST CHẶN KIẾN TRÚC
├── l10n/          đổi ngôn ngữ vi -> en
├── screens/       login, register, search, history
└── widget_test.dart
```

### Ba kỹ thuật quan trọng

**a) Mock Dio — không bao giờ chạm socket.**
`test/data/adapters/fake_http.dart` cài `HttpClientAdapter` giả. Nhờ vậy test
adapter chạy được **offline hoàn toàn**, trong `flutter test` (vốn chặn mạng).

**b) Override provider thay vì mock static.**
Mọi dependency đều qua Riverpod nên test chỉ cần:
```dart
ProviderScope(
  overrides: [authRepositoryProvider.overrideWithValue(_FakeAuthRepository())],
  child: const LoginScreen(),
)
```
Thay adapter = 1 dòng, không cần sửa code app.

**c) Test canh kiến trúc (`test/domain/ports/domain_is_pure_dart_test.dart`).**
Không chỉ test hành vi mà còn **chặn hồi quy**:
- `lib/domain/**` không được import `package:flutter/`, `dart:ui`, `dart:html`;
- không được import `data/adapters/`, `screens/`, `core/providers`;
- có test phụ "có ít nhất 1 file" để chắc chắn test **không pass giả** do sai
  đường dẫn (đã cứu dự án 1 lần khi `flutter analyze` không chạy được).

## 4. Những gì cố ý CHƯA test (và vì sao)

| Phần | Lý do |
|---|---|
| `process_image_screen.dart` (40%) | Cần **chọn ảnh thật** qua `image_picker`; không giả lập được trong test. Phần logic đã test qua `analysisControllerProvider`. |
| `main.dart` (56.5%) | Cần cả pipeline thật + `flutter run`; phần quan trọng (ProviderScope, locale, theme) đã kiểm qua `widget_test.dart`. |
| `agent_client.dart` (75%) | **Chưa có adapter thật** — sẽ viết ở Day 22. Giờ chỉ có port + test "khuôn". |
| Luồng `/logout`, `/me` thật | Server Flask chưa có endpoint; mock đã phủ qua `RemoteAuthRepository`. |
| Bảo mật thật (keystore, TLS) | **Không test được trong unit test** — cần kiểm thử trên thiết bị thật trước khi phát hành (xem `docs/ROADMAP.md` Day 11). |

## 5. Test bảo mật riêng

`test/core/network/auth_security_test.dart` kiểm chứng bằng test, không tin bằng mắt:
- `/history` và `/process-image` **có** header `Authorization: Bearer <token>`;
- `/login` (công khai) **không** gửi token;
- 401 ở endpoint có bảo vệ → `onUnauthorized` gọi **đúng 1 lần**;
- **401 ở `/login` → KHÔNG gọi** (sai mật khẩu không phải hết phiên);
- `redactHeaders`/`redactBody` không lộ password/token ở cả cấp lồng.

`test/data/adapters/secure_token_store_test.dart` kiểm tra token không bao giờ
lọt ra `null`/`''` rác, và **kho hỏng thì nuốt lỗi chứ không crash**.

## 6. Bài học khi viết test (dành cho người tiếp theo)

> **1. Test mà chỉ "chạy qua" thì vô nghĩa.** Với 3 chỗ Day 14 phát hiện lỗi thật,
> nguyên tắc chung: **assert hành vi người dùng thấy**, không assert biến nội bộ.
>
> **2. `pumpWidget` với cây widget y hệt sẽ chỉ UPDATE tại chỗ** — `initState`
> không chạy lại. Muốn mô phỏng "mở lại app" thì phải **đổi `key`**.
> (Đã vấp đúng lỗi này ở test ngoại tuyến: banner không hiện vì `load()` không được gọi.)
>
> **3. `SharedPreferences` và `flutter_secure_storage` đều cần mock trong test:**
> `SharedPreferences.setMockInitialValues({...})` và
> `FlutterSecureStorage.setMockInitialValues({...})`. Thiếu → `MissingPluginException`
> **che mất** lỗi thật đang muốn kiểm tra.
>
> **4. `unawaited(...)` trong code sẽ không xong khi `pumpAndSettle` kết thúc** —
> thêm `await tester.pump()` nếu test phụ thuộc việc ghi đó hoàn tất.
>
> **5. Widget test phải ghim `locale` + `localizationsDelegates`** (từ Day 13 mọi
> chuỗi đến từ `AppLocalizations.of(context)`). Thiếu → *"Null check operator
> used on a null value"* lúc build.

## 7. Định mức

| Mức | Quy tắc |
|---|---|
| Tối thiểu khi sửa code | `flutter analyze` sạch + `flutter test` xanh |
| Thêm/sửa logic tầng `data`/`core` | kèm test cho nhánh thành công **và** nhánh lỗi |
| Trước khi tag phiên bản | `flutter test --coverage` + cập nhật số liệu ở file này |

> `coverage/` đã được thêm vào `.gitignore` — chỉ là số liệu tạm để đo, **không**
> track. Số liệu chốt nằm trong file này.