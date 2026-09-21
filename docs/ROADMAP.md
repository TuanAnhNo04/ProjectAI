# KẾ HOẠCH THEO NGÀY — image_processing_app → Trợ lý thuốc AI

> **Phiên bản:** 1.0 · **Ngày tạo:** 21/09/2026
> **Phạm vi:** từ "project không build được" → "app có kiến trúc nền tảng + agent phía server"
> **Trạng thái:** 📋 KẾ HOẠCH (chỉ mục 1 là việc ĐÃ hoàn thành)
> **Kiến trúc đã chốt:** Edge (Flutter) – Core (backend + agent) – Provider (LLM thay thế được), ports & adapters, contract-first, offline-first.

---

## 0. Cách dùng tài liệu này

- Mỗi ngày là **1 khối việc 2–4 giờ**, có **DoD (Definition of Done)** = lệnh chạy ra kết quả mong đợi; chỉ tick xong khi DoD đạt.
- Nhãn công việc: ✋ tự làm (không cần AI) · 🤖 nên giao AI viết · 🤖✋ AI viết, bạn tự verify.
- **Vì Cursor trial sắp hết quota, tuân thủ 6 quy tắc:**
  1. **1 ngày = 1 phiên AI.** Không mở phiên mới giữa ngày; không hỏi những câu chỉ để "giải thích".
  2. **Prompt theo mẫu ở Phụ lục A** (luôn kèm: file cần đọc → việc cần làm → lệnh cần chạy → điều kiện dừng).
  3. **Tự làm hết việc ✋** (git, chạy lệnh, sửa config, test tay, đọc log) — chiếm ~40% khối lượng, tốn 0 quota.
  4. **Commit sau mỗi task**: `git add -A; git commit -m "..."`. AI làm hỏng → `git restore .` và làm lại, không mất gì.
  5. Việc cần AI sửa nhiều file → **gom thành 1 task duy nhất**, đừng tách nhỏ.
  6. Hết quota giữa đường → xem **Phương án B (Phụ lục D)**: các việc ✋ vẫn làm tiếp được.

---

## 1. ĐÃ LÀM — kết quả phiên audit (không cần làm lại)

| # | Việc đã làm | Bằng chứng | Trạng thái |
|---|---|---|---|
| 1 | Đọc & lập bản đồ toàn bộ project | 9 file Dart; `models/user_model.dart`, `services/api_service.dart`, `widgets/custom_button.dart` = **0 byte** | ✅ |
| 2 | Trích ra API contract client đang gọi | grep `http://` → 6 chỗ trỏ `10.0.2.2:5000` | ✅ |
| 3 | Kiểm chứng blocker build | `flutter pub upgrade --dry-run` → `The current Dart SDK version is 3.3.2 ... requires SDK version ^3.5.3, version solving failed` | ✅ (phát hiện, **chưa sửa**) |
| 4 | Phát hiện lockfile lệch + chưa từng build | `pubspec.lock` chỉ có 5 direct deps; không có thư mục `.dart_tool` | ✅ |
| 5 | Xác định dependency chết | `tflite_flutter_helper`: discontinued, `sdk >=2.12 <3.0.0`, kéo `tflite_flutter ^0.9`, `camera ^0.9.4`, `image ^3` → xung đột với pubspec hiện tại | ✅ |
| 6 | Xác định lỗi cấu hình nền tảng | Không có `res/xml/network_security_config.xml`; Manifest thiếu `CAMERA`; `Info.plist` thiếu `NSPhotoLibraryUsageDescription`/`NSCameraUsageDescription`; macOS thiếu `network.client`; `enableJetifier` deprecated | ✅ |
| 7 | Xác định dead code & bug | `_searchQuery` không bao giờ gán; `_buildHistoryButton()` không dùng; `MedicineSearchScreen.dart:69-73` gọi `requestFocus(FocusNode())` 2 lần → **không mở được bàn phím**; URL không encode (`?name=$ten`); parse bằng `split(': ')` | ✅ |
| 8 | Xác định lỗi test/CI/quy trình | `test/widget_test.dart` là test counter mặc định → **sẽ fail**; chưa có git repo; chưa có CI; README template | ✅ |
| 9 | Chốt kiến trúc dài hạn | Edge–Core–Provider + MCP cho tools; on-device chỉ làm tầng Edge (ML Kit GenAI Prompt API còn **beta, không SLA**; AI Edge RAG SDK **deprecated**) | ✅ |

**Chưa làm (bắt đầu từ Day 1):** sửa pubspec, cấu hình nền tảng, ports & adapters, auth, cache, backend Core, agent, on-device, release.

### Tổng quan lộ trình

| Giai đoạn | Ngày | Kết quả cuối |
|---|---|---|
| **P0 – Cứu project** | Day 0 → Day 4 | `pub get` + `analyze` + `test` xanh; app chạy trên máy; git + CI |
| **P1 – Nền tảng client** | Day 5 → Day 14 | Ports & adapters, model typed, Riverpod, auth token, cache/offline, l10n, test → **tag v0.1.0** |
| **P2 – Core + Agent server** | Day 15 → Day 24 | FastAPI + Postgres/pgvector + JWT + RAG có citation + agent tool-calling + SSE + eval |
| **P3 – Nâng cao & Release** | Day 25 → Day 30 | Chat streaming, on-device OCR, memory & nhắc thuốc, MCP servers, privacy → **tag v1.0.0** |

---

## 2. GIAI ĐOẠN P0 — Cứu project (Day 0 → Day 4)

### Day 0 — Khởi động & chốt "trạng thái số 0" (1 giờ) ✋ ✅ **21/09/2026**
- [x] ✋ `flutter --version` · `flutter doctor`
      - Ghi nhận: Flutter **3.19.4** stable / Dart **3.3.2** · git **2.55.0.windows.3** · identity `Tuan Anh <davidpham01104@gmail.com>` · Windows 10.0.26200 · locale en-US
      - ⚠️ `flutter doctor`: **Android toolchain ✗ (thiếu `cmdline-tools`)** · Android Studio chưa cài · Visual Studio chưa cài · 3 device khả dụng nhưng **không có emulator/máy Android**
- [x] ✋ `git init -b main`; `.gitignore` của project đã có sẵn (không cần thêm)
- [x] ✋ Commit baseline → `8a74b8f` *chore: baseline before refactor (Flutter app + audit findings)*
- [x] ✋ Commit tài liệu này → `4f30d26` *docs: add day-by-day roadmap (P0-P3)*
- [x] **DoD:** `git log --oneline` có 2 commit; working tree sạch

### Day 1 — Nâng SDK + dọn `pubspec.yaml` (3 giờ) 🤖✋ ✅ **21/09/2026**
- [x] ✋ **Quyết định: chọn phương án (B)** — giữ Flutter 3.19.4, hạ `environment.sdk: ^3.5.3` → `^3.3.0` (khớp Dart 3.3.2 đang cài; việc `flutter upgrade` lên ≥ 3.24 dời sang Day 26 vì ML Kit cần iOS 15.5/compileSdk 35)
- [x] 🤖 Sửa `pubspec.yaml`:
      - **XOÁ 11 dep không dùng:** `tflite_flutter`, `tflite_flutter_helper` (discontinued), `google_mlkit_text_recognition`, `camera`, `cached_network_image`, `lottie`, `image`, `path_provider`, `uuid`, `build_runner`
      - **GIỮ:** `http ^0.13.6` (code hiện tại còn dùng — thay bằng `dio` ở Day 8–9), `cupertino_icons`, `image_picker`
      - **THÊM sẵn cho P1:** `dio`, `flutter_riverpod`, `flutter_secure_storage`, `shared_preferences`, `intl`
      - Xoá 3 file rỗng: `lib/models/user_model.dart`, `lib/services/api_service.dart`, `lib/widgets/custom_button.dart`
- [x] ✋ Resolve lại từ đầu:
      ```powershell
      Remove-Item pubspec.lock -Force
      flutter pub get     # → Changed 79 dependencies!
      ```
- [x] **DoD (đạt):** `pubspec.lock` mới có **9 direct deps** (đủ 100% khai báo), `sdks: dart >=3.3.0, flutter >=3.19.0`
- 📌 Ghi chú: `flutter analyze` = **38 issues (0 warning, 38 info)** — toàn bộ là lint của code cũ, xử lý ở Day 10. `flutter test` fail 1 test mặc định → viết lại ở Day 3.

### Day 2 — Cấu hình nền tảng để app chạy thật (3 giờ) ✋🤖 ✅ **21/09/2026** (code xong, `flutter run` ⏳ chờ JDK)
- [x] ✋ Tạo `android/app/src/debug/res/xml/network_security_config.xml` — cho phép cleartext tới `10.0.2.2`, `localhost`, `127.0.0.1` (**chỉ debug**; bản release không có file này → bắt buộc HTTPS)
- [x] ✋ `android/app/src/debug/AndroidManifest.xml`: thêm `<application android:networkSecurityConfig="@xml/network_security_config" />`
- [x] ✋ `android/app/src/main/AndroidManifest.xml`: thêm `CAMERA` + `<uses-feature camera required="false">`; **giữ `INTERNET`**
      - 📌 **Chủ động KHÔNG thêm `READ_MEDIA_IMAGES`/`READ_EXTERNAL_STORAGE`**: `image_picker 1.x` dùng Android Photo Picker (API 33+)/`ACTION_GET_CONTENT` nên không cần quyền đọc bộ nhớ; thêm quyền thừa sẽ bị Google Play soi khi review.
- [x] 🤖 `ios/Runner/Info.plist`: thêm `NSPhotoLibraryUsageDescription`, `NSCameraUsageDescription` (mô tả tiếng Việt) và `NSAppTransportSecurity` → `NSExceptionDomains` cho `localhost`/`127.0.0.1` (không dùng `NSAllowsArbitraryLoads` nên HTTPS ở release vẫn nguyên vẹn)
- [x] ✋ `macos/Runner/DebugProfile.entitlements` + `Release.entitlements`: thêm `com.apple.security.network.client`
- [x] ✋ `android/gradle.properties`: **xoá** `android.enableJetifier=true`
- [x] ✋ `android/app/build.gradle`: `minSdk = 26`; `namespace` + `applicationId` = **`com.medassist.app`**; di chuyển `MainActivity.kt` sang `kotlin/com/medassist/app/` (đổi cả `package` — namespace mới resolve `.MainActivity` theo đường dẫn này)
- [x] ✋ **Phát sinh:** pin `ndkVersion = "26.3.11579264"` (bản đã cài trên máy) thay cho `flutter.ndkVersion` (25.1.8937393 **không có** trên máy → Gradle sẽ cố tải NDK và fail vì thiếu `cmdline-tools`)
- [x] **Kiểm chứng đã chạy:** parse XML/plist hợp lệ **6/6 file** (`[xml]` PowerShell) · commit `ec6346d` (10 file)
- [ ] ⏳ **DoD `flutter run`:** **CHƯA verify được** — máy thiếu **JDK** (`java` không có trong PATH, `JAVA_HOME` chưa set → `flutter build apk --debug` trả `ERROR: JAVA_HOME is not set`), chưa có **AVD** (`flutter emulators` → *No emulators available* dù đã có system-image android-33/34/35). Cách gỡ: xem mục 11.

### Day 3 — Test & CI (2 giờ) 🤖✋ ✅ **21/09/2026**
- [x] ✋ Xoá nội dung test counter trong `test/widget_test.dart` → thay bằng smoke test khởi động app
- [x] 🤖 Viết **5 test thật** (nhiều hơn 3 test dự kiến — thêm test điều hướng và test validate search):
      1. `test/widget_test.dart` — app khởi động vào màn Đăng nhập (2 `TextField` + nút `Đăng nhập` + link đăng ký)
      2. `test/screens/login_screen_test.dart` — render đủ ô nhập/nút/link
      3. `test/screens/login_screen_test.dart` — server lỗi (flutter_test chặn HTTP và trả 400) → hiện `SnackBar "Đăng nhập thất bại!"`, **không crash, không điều hướng sai**
      4. `test/screens/login_screen_test.dart` — bấm "Đăng ký tài khoản mới" → mở `RegisterScreen`
      5. `test/screens/medicine_search_screen_test.dart` — bỏ trống tên thuốc → hiện `"Vui lòng nhập tên thuốc!"`, **không gọi API**
- [x] ✋ `flutter analyze` (38 info, 0 warning/error) + `flutter test`
- [x] ✋ Tạo `.github/workflows/ci.yml`: `flutter pub get` → `dart format --output=none --set-exit-if-changed .` → `flutter analyze --no-fatal-infos` → `flutter test` (ghim Flutter 3.19.4 cho khớp máy dev)
- [x] ✋ Chạy `dart format .` (5 file được format lại) để **CI không đỏ ngay lần push đầu**
- [x] **DoD (đạt):** `flutter test` → **`All tests passed!` (5/5)**; cả 3 bước CI chạy local đều **exit code 0** (format: 0 changed · analyze: 38 info non-fatal · test: 5 passed). Commit `5f68469`.


### Day 4 — Buffer / trả nợ kỹ thuật (2 giờ) ✋
- [ ] Xử lý phần trượt từ Day 1–3 (nâng Flutter thường phát sinh lỗi Gradle/CocoaPods)
- [ ] Ghi mọi lỗi gặp phải + cách sửa vào `docs/TROUBLESHOOTING.md` (tài liệu này cứu bạn ở Day 15+)
- [ ] **DoD:** checkbox Day 1–3 tick hết. 🎉 **Milestone M0**

---

## 3. GIAI ĐOẠN P1 — Nền tảng client (Day 5 → Day 14)

### Day 5 — Config & HTTP client (3 giờ) 🤖✋
- [ ] 🤖 Tạo 4 file:
      - `lib/core/config/app_config.dart` — `baseUrl` đọc từ `String.fromEnvironment('API_BASE_URL', defaultValue: 'http://10.0.2.2:5000')`
      - `lib/core/network/api_client.dart` — Dio (timeout, interceptor log, chỗ gắn token)
      - `lib/core/network/api_exception.dart` — map lỗi mạng/401/404/500 → thông báo tiếng Việt
      - `lib/core/network/result.dart` — `Result<T>` (Ok/Err) tự viết, không thêm dependency
- [ ] ✋ Chạy `flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000`, in `AppConfig.baseUrl` để verify
- [ ] **DoD:** tìm `10.0.2.2` trong `lib/` chỉ còn **1 kết quả** (trong `app_config.dart`).

### Day 6 — Model typed (3 giờ) 🤖✋
- [ ] 🤖 Tạo `lib/data/models/`: `medicine.dart` (Medicine_Name, Composition, Uses, Side_effects, Manufacturer, 3 tỉ lệ review), `history_item.dart` (id, filepath, searchResults), `user.dart`, `analysis_result.dart`
- [ ] 🤖 Mỗi model: `fromJson` / `toJson` / `copyWith`, **không dùng `freezed`** (giữ ít dependency)
- [ ] 🤖 Test `test/data/models/*_test.dart` bằng **JSON thật lấy từ server** (cách này phát hiện sai tên trường ngay lập tức)
- [ ] **DoD:** `flutter test test/data` xanh; không còn chỗ nào parse JSON thủ công.

### Day 7 — Ports (tầng domain, không import Flutter) (2 giờ) 🤖✋
- [ ] 🤖 Tạo `lib/domain/ports/`:
      - `medicine_analyzer.dart` → `abstract interface class MedicineAnalyzer { Future<AnalysisResult> analyze(String imagePath); String get capability; }`
      - `medicine_repository.dart` → `search(name)`, `history()`, `deleteHistory(id)`
      - `auth_repository.dart` → `login(u,p)`, `register(u,p,e)`, `logout()`, `currentToken()`
      - `agent_client.dart` → `Stream<AgentEvent> chat({sessionId, message})` + `AgentEvent` (TokenDelta, ToolCallStarted, Citation, Done, Failure)
- [ ] **DoD:** trong `lib/domain/**` **không** có `import 'package:flutter/...'` (kiểm bằng `Select-String`).

### Day 8 — Adapter remote (nối Flask hiện tại) (4 giờ) 🤖✋
- [ ] 🤖 `lib/data/adapters/remote_analyzer.dart` — multipart field `image` → `POST /process-image` → map `medicine_info` vào `AnalysisResult`
- [ ] 🤖 `lib/data/adapters/remote_auth_repository.dart` — `/login`, `/register` (lưu token nếu server trả), `/logout`
- [ ] 🤖 `lib/data/adapters/remote_medicine_repository.dart` — `/search-medicine` (dùng `Uri(queryParameters:)` để encode), `/history`, `DELETE /history/{id}`
- [ ] ✋ Chạy server Flask → test tay 5 luồng: login sai, login đúng, upload ảnh, search "panadol", xoá 1 history
- [ ] **DoD:** app chạy được toàn bộ luồng cũ nhưng qua adapter mới (chưa cần sửa UI vội — Day 9 mới nối).

### Day 9 — Nối UI vào port + Riverpod (4 giờ) 🤖✋
- [ ] 🤖 `lib/core/providers.dart`: `ProviderScope` + provider cho ApiClient, 3 repository, analyzer
- [ ] 🤖 Sửa 5 màn hình (`login_screen`, `register_screen`, `process_image_screen`, `HistoryScreen`, `MedicineSearchScreen`) dùng controller/repository thay `http` trực tiếp; thêm trạng thái loading/error/empty
- [ ] ✋ `flutter analyze` → xử lý hết `use_build_context_synchronously` (thêm `if (!mounted) return;` sau mỗi `await`)
- [ ] **DoD:** `Select-String -Path lib/screens/*.dart -Pattern 'http'` = **0 kết quả**; app chạy full luồng.

### Day 10 — Sửa bug + dọn dead code (3 giờ) 🤖✋
- [ ] 🤖 `MedicineSearchScreen`: xoá 2 dòng `requestFocus(FocusNode())` trong `onTap` (bug **không mở được bàn phím**); thêm `textInputAction: TextInputAction.search` + `onSubmitted`
- [ ] 🤖 `process_image_screen`: xoá `_searchQuery` + `_getFilteredResults()` + `_buildHistoryButton()` (dead code); render kết quả từ model thay cho `split('\n')` + `split(': ')`
- [ ] 🤖 Layout: thay `SizedBox(height: 400)` bằng `Flexible`/`ListView`, bọc `SafeArea`, kiểm tra với cỡ chữ hệ thống lớn
- [ ] ✋ `dart format .; flutter analyze`
- [ ] **DoD:** `flutter analyze` → `No issues found!`

### Day 11 — Phiên đăng nhập & bảo mật client (3 giờ) 🤖✋
- [ ] 🤖 Lưu token bằng `flutter_secure_storage`; Dio interceptor tự gắn `Authorization: Bearer <token>`
- [ ] 🤖 Gặp 401 → xoá token + điều hướng về Login; thêm nút **Đăng xuất** ở Home
- [ ] 🤖 Form: `autofillHints`, nút hiện/ẩn mật khẩu, validate rỗng + mật khẩu ≥ 6 ký tự + email hợp lệ
- [ ] ✋ Verify: xem header request có token; xoá token → app về Login
- [ ] **DoD:** mọi request `/history`, `/process-image` đều có header Authorization; không log mật khẩu/token ra console.

### Day 12 — Cache & offline tối thiểu (3 giờ) 🤖✋
- [ ] 🤖 `lib/data/local/cache_store.dart` — cache JSON `history` + `search` bằng `shared_preferences`, kèm `cachedAt`
- [ ] 🤖 UI: banner "Dữ liệu ngoại tuyến · cập nhật <thời gian>" + nút Thử lại
- [ ] 🤖 Dùng `cached_network_image` cho thumbnail ảnh (nếu server trả URL)
- [ ] ✋ Test: mở app có mạng → tắt mạng → vẫn xem được History/Search gần nhất
- [ ] **DoD:** tắt wifi, app không trắng màn hình và không crash.

### Day 13 — Theme, i18n, disclaimer (3 giờ) 🤖✋
- [ ] 🤖 `MaterialApp`: `theme` Material 3 (light/dark), `title` đúng tên app (bỏ "Flutter Demo"), `localizationsDelegates` vi/en
- [ ] 🤖 Tạo `l10n/app_vi.arb` + `app_en.arb`, chuyển toàn bộ chuỗi hardcode (đang lẫn Việt/Anh) sang `AppLocalizations`
- [ ] 🤖 Thêm **disclaimer** cố định ở màn kết quả và màn chat: "Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán/đơn của bác sĩ."
- [ ] **DoD:** `flutter gen-l10n` chạy sạch; đổi ngôn ngữ trong app không còn chuỗi hardcode.

### Day 14 — Test + chốt M1 (3 giờ) 🤖✋
- [ ] 🤖 Test cho repository (mock Dio), model, validate form, trạng thái empty/error
- [ ] ✋ `flutter test --coverage`; ghi số liệu vào `docs/TESTING.md`
- [ ] ✋ Cập nhật `README.md` thật: mục tiêu, kiến trúc, cách chạy, biến `--dart-define`
- [ ] ✋ `git tag v0.1.0`
- [ ] **DoD:** `flutter analyze` sạch + `flutter test` xanh + coverage tầng `data/` ≥ 50%. 🎉 **Milestone M1**

---

## 4. GIAI ĐOẠN P2 — Core & Agent phía server (Day 15 → Day 24)

> Nguyên tắc: **backend là "bộ não"**. Giai đoạn này chủ yếu làm backend; app chỉ thêm gọi `/v1` + SSE.
> Nếu chưa có repo backend: tạo mới ở Day 15. Nếu đã có Flask: chạy song song và chuyển dần (không big-bang).

### Day 15 — Dựng Core (4 giờ) ✋🤖
- [ ] ✋ Tạo repo `medassist-core` với `docker compose` gồm **FastAPI + Postgres + Redis**, chạy được local
- [ ] 🤖 Skeleton FastAPI: `/v1/health`, `/v1/auth/*`, `/v1/analyze`, `/v1/history`, `/v1/search`; Pydantic schema; format lỗi thống nhất `{code, message, detail}`
- [ ] 🤖 Xuất `openapi.json`
- [ ] **DoD:** `docker compose up` chạy; `/docs` hiện đủ endpoint; app gọi được `/v1/health` sau khi đổi `API_BASE_URL`.

### Day 16–17 — Auth thật + phân quyền dữ liệu (2 ngày) 🤖✋
- [ ] 🤖 Đăng ký/đăng nhập: hash **argon2/bcrypt**, JWT access (15 phút) + refresh (7 ngày), rate limit login qua Redis
- [ ] 🤖 Mọi bảng có `user_id`; **mọi** endpoint lịch sử/ảnh lọc theo user đang đăng nhập → sửa lỗi IDOR hiện tại của client
- [ ] ✋ Test: hai tài khoản A/B, A **không** thấy và không xoá được history của B
- [ ] **DoD:** token hết hạn → refresh hoạt động; truy cập dữ liệu người khác trả 403/404.

### Day 18–19 — Dữ liệu thuốc & tìm kiếm (2 ngày) 🤖✋
- [ ] 🤖 Schema Postgres: `users, drugs, ingredients, interactions, history, assets`
- [ ] 🤖 Pipeline ingest danh mục thuốc (nguồn được phép dùng) → chuẩn hoá tên + hoạt chất + hàm lượng
- [ ] 🤖 `/v1/search`: tìm theo tên **và** hoạt chất, fuzzy bằng `pg_trgm`, phân trang, trả kèm `source` + `updated_at`
- [ ] ✋ Kiểm tra tay: gõ không dấu "panadol", "para 500" đều ra đúng thuốc
- [ ] **DoD:** ≥ 90% ca trong danh sách test tay ra kết quả đúng và có nguồn.

### Day 20–21 — RAG có citation (2 ngày) 🤖✋
- [ ] 🤖 Bật `pgvector`; ingest tờ hướng dẫn/leaflet → chunk → embedding → lưu vector
- [ ] 🤖 Endpoint hỏi đáp: retrieval top-k + trả lời **bắt buộc kèm citation** (id tài liệu + đoạn trích)
- [ ] 🤖 Guardrail: độ tương đồng thấp → trả lời *"Không đủ dữ liệu để trả lời, hãy hỏi dược sĩ/bác sĩ"*, không được bịa
- [ ] **DoD:** 20 câu hỏi thử đều có citation đúng đoạn; câu hỏi ngoài dữ liệu bị từ chối.

### Day 22–23 — Agent core v1 (2 ngày) 🤖✋
- [ ] 🤖 Orchestrator: planner + tool-calling với 5 tool: `search_drug`, `check_interaction`, `normalize_prescription`, `explain_for_patient`, `schedule_reminder`
- [ ] 🤖 Bọc tool sau interface/factory để Day 28 thay bằng MCP server mà **không sửa orchestrator**
- [ ] 🤖 SSE streaming: `POST /v1/agent/chat` trả `text/event-stream` (token delta + sự kiện gọi tool + citation)
- [ ] 🤖 Memory: session ngắn hạn ở Redis, hồ sơ người dùng ở Postgres
- [ ] ✋ Test: "Tôi đang uống warfarin, thêm panadol được không?" → agent phải gọi tool tương tác và trả lời có căn cứ
- [ ] **DoD:** hội thoại 5 lượt giữ được ngữ cảnh; câu trả lời y khoa đều đi qua tool/nguồn, không tự bịa.

### Day 24 — Eval + quan sát + hợp đồng API (2 ngày gộp) 🤖✋
- [ ] 🤖 Bộ **eval 100–200 ca** (`evals/cases.yaml`): đơn thuốc mẫu, ảnh mờ, tương tác nguy hiểm, câu hỏi bẫy, câu hỏi ngoài phạm vi
- [ ] 🤖 `eval.py` chạy tự động → báo cáo pass/fail theo nhóm; chạy lại mỗi khi đổi prompt/model
- [ ] 🤖 Observability: trace id mỗi request, log chi phí token theo user, dashboard cơ bản
- [ ] 🤖 Sinh **Dart client** từ `openapi.json` (retrofit/openapi_generator) → app dùng type-safe, bỏ parse tay
- [ ] **DoD:** `python eval.py` in bảng điểm; đổi prompt → điểm thay đổi rõ ràng. 🎉 **Milestone M2**

---

## 5. GIAI ĐOẠN P3 — Nâng cao & Release (Day 25 → Day 30+)

### Day 25 — Chat UI streaming (3 giờ) 🤖✋
- [ ] 🤖 Màn `AgentChatScreen`: stream token (Dio `ResponseType.stream` / SSE), bong bóng hội thoại, **citation bấm được**
- [ ] 🤖 Trạng thái "đang tra cứu…" khi agent gọi tool; nút Dừng; lịch sử hội thoại lưu offline
- [ ] **DoD:** chữ hiện dần theo token; bấm citation mở đúng đoạn tài liệu.

### Day 26 — On-device OCR + Hybrid (4 giờ) 🤖✋
- [ ] ✋ Thêm lại dependency đúng: `google_mlkit_text_recognition`, `image`, `path_provider` (**không** dùng `tflite_flutter_helper` — đã discontinued)
- [ ] ✋ iOS: sửa `Podfile` thành `platform :ios, '15.5'` + block `post_install` set `IPHONEOS_DEPLOYMENT_TARGET`; đặt `compileSdk/targetSdk = 35`
- [ ] 🤖 `ondevice_analyzer.dart` (ML Kit OCR + chuẩn hoá chuỗi + tra DB local) và `hybrid_analyzer.dart` (local trước → fallback remote → ghi cache)
- [ ] 🤖 Feature flag chọn adapter: `--dart-define=ANALYZER=hybrid` (không sửa UI)
- [ ] ✋ Test trên **thiết bị thật** (emulator OCR kém chính xác) ở chế độ máy bay
- [ ] **DoD:** chế độ máy bay + ảnh có trong DB local → vẫn trả kết quả < 2 giây.

### Day 27 — Memory & nhắc uống thuốc (3 giờ) 🤖✋
- [ ] 🤖 Hồ sơ sức khoẻ: dị ứng, bệnh nền, thuốc đang dùng (có consent rõ ràng, sửa/xoá được)
- [ ] 🤖 Nhắc uống thuốc bằng local notification; ghi nhận adherence; agent dùng dữ liệu này khi trả lời
- [ ] **DoD:** tạo lịch nhắc → nhận được thông báo; agent biết "thuốc đang dùng" của tôi.

### Day 28 — Đóng gói tool thành MCP servers (3 giờ) 🤖✋
- [ ] 🤖 Expose `search_drug`, `check_interaction` ra **MCP server** (chuẩn mở, dùng lại được cho client AI khác)
- [ ] 🤖 Orchestrator gọi tool **qua MCP** thay vì import trực tiếp → chứng minh khả năng tái sử dụng
- [ ] ✋ Kiểm thử bằng MCP Inspector
- [ ] **DoD:** gọi được tool từ ngoài app (một MCP client bất kỳ) — đây là điểm "bước trước thời đại".

### Day 29 — An toàn, riêng tư, ổn định (3 giờ) 🤖✋
- [ ] 🤖 Consent + chính sách dữ liệu trong app; retention (tự xoá ảnh sau N ngày); audit log hành động nhạy cảm
- [ ] 🤖 Crash reporting (Crashlytics/Sentry), timeout + retry, offline queue
- [ ] ✋ Test tay 8 tình huống xấu: mất mạng giữa upload, token hết hạn, server 500, ảnh cực lớn, màn hình nhỏ, cỡ chữ lớn, xoay màn hình, đăng nhập 2 tài khoản
- [ ] **DoD:** không có màn trắng/crash trong 8 tình huống trên.

### Day 30 — Release (3 giờ) ✋🤖
- [ ] ✋ Tạo keystore release thật (bỏ debug signing); `flutter build appbundle --release --dart-define=API_BASE_URL=https://...`
- [ ] ✋ Icon/splash (`flutter_launcher_icons`, `flutter_native_splash`), tên app, ảnh chụp màn hình
- [ ] ✋ Flavor `dev/staging/prod` + `--dart-define` cho từng môi trường
- [ ] ✋ Beta nội bộ (Internal testing / TestFlight), thu phản hồi
- [ ] ✋ `git tag v1.0.0`, cập nhật tiến độ vào Phụ lục C
- [ ] **DoD:** `.aab` release build thành công, cài được từ Internal testing. 🎉 **Milestone M3**

### Sau Day 30 — backlog dài hạn
- Multi-agent tách vai (interaction checker, adherence coach, report generator) · Voice input (STT) cho người lớn tuổi · OCR đơn thuốc viết tay · PWA/web cho dược sĩ · Multi-tenant cho nhà thuốc · Model routing (việc đơn giản chạy on-device) · Wearable (huyết áp/đường huyết) · Fine-tune model y khoa tiếng Việt · Bản on-premise cho bệnh viện.

---

## 6. Phụ lục A — Mẫu prompt tiết kiệm quota

**Mẫu chuẩn (dán 1 lần cho 1 ngày):**
```
Đọc trước: <danh sách file>.
Nhiệm vụ: <mô tả 1 việc duy nhất>.
Ràng buộc: không sửa file ngoài danh sách; tuân theo kiến trúc ports & adapters trong docs/ROADMAP.md.
Sau khi sửa, chạy: <lệnh> và in kết quả.
Nếu build/lint lỗi thì dừng, in log lỗi và đề xuất 1 cách sửa. Không giải thích dài dòng.
```

**Mẫu sửa lỗi:**
```
Lỗi khi chạy: <dán log nguyên văn>.
File liên quan: <...>.
Hãy tìm nguyên nhân gốc, sửa tối thiểu, rồi chạy lại <lệnh> để chứng minh đã hết lỗi.
```

**Mẫu sinh test:**
```
Đọc <file>. Viết test cho <hàm/class>, dùng đúng framework đang có (flutter_test + mocktail nếu cần).
Chỉ tạo file trong test/. Chạy flutter test và in kết quả.
```

**Quy tắc vàng:** mỗi prompt phải có đủ 4 phần: (1) file cần đọc · (2) việc cần làm · (3) lệnh cần chạy · (4) điều kiện dừng.

---

## 7. Phụ lục B — Lệnh dùng hằng ngày (✋ tự làm)

```bash
flutter --version
flutter doctor -v
flutter pub get
flutter analyze
dart format .
flutter test --coverage
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000
flutter build appbundle --release --dart-define=API_BASE_URL=https://api.ten-ban.vn

git add -A; git commit -m "feat(...): ..."
git restore .                     # huỷ hết thay đổi nếu AI sửa hỏng
git switch -c day-08-adapter-remote
git tag v0.1.0
```

Kiểm tra nhanh bằng PowerShell:
```powershell
Select-String -Path lib/**/*.dart -Pattern '10.0.2.2'      # phải chỉ còn app_config.dart
Select-String -Path lib/screens/*.dart -Pattern 'http'      # phải = 0 ở Day 9
Get-ChildItem -Recurse -File lib | ForEach-Object { "$($_.Length) $($_.FullName)" }  # phát hiện file rỗng
```

---

## 8. Phụ lục C — Bảng theo dõi tiến độ (tick khi DoD đạt)

| Ngày | Việc | AI? | DoD | Xong |
|---|---|---|---|---|
| 0 | Baseline + `git init` | ✋ | `git log` có commit | ✅ 21/09 |
| 1 | Nâng SDK + dọn pubspec | 🤖✋ | `flutter pub get` xanh | ✅ 21/09 |
| 2 | Cấu hình nền tảng | ✋🤖 | `flutter run` không crash | ✅ 21/09 (run ⏳ chờ JDK) |
| 3 | Test + CI | 🤖✋ | `flutter test` xanh, CI xanh | ✅ 21/09 |
| 4 | Buffer → **M0** | ✋ | Day 1–3 tick hết | ☐ |
| 5 | AppConfig + ApiClient | 🤖✋ | chỉ còn 1 chỗ có `10.0.2.2` | ☐ |
| 6 | Model typed | 🤖✋ | test models xanh | ☐ |
| 7 | Ports | 🤖✋ | domain không import Flutter | ☐ |
| 8 | Adapter remote | 🤖✋ | 5 luồng cũ chạy | ☐ |
| 9 | Nối UI + Riverpod | 🤖✋ | screens không còn `http` | ☐ |
| 10 | Bug + dead code | 🤖✋ | `analyze` sạch | ☐ |
| 11 | Token + bảo mật | 🤖✋ | mọi request có token | ☐ |
| 12 | Cache/offline | 🤖✋ | tắt mạng vẫn xem được | ☐ |
| 13 | Theme/i18n/disclaimer | 🤖✋ | không còn chuỗi hardcode | ☐ |
| 14 | Test → **v0.1.0** | 🤖✋ | coverage `data/` ≥ 50% | ☐ |
| 15 | Dựng Core (docker) | ✋🤖 | `/docs` hiện endpoint | ☐ |
| 16–17 | Auth + phân quyền | 🤖✋ | A không thấy dữ liệu của B | ☐ |
| 18–19 | Dữ liệu thuốc + search | 🤖✋ | ≥ 90% ca tay đúng | ☐ |
| 20–21 | RAG + citation | 🤖✋ | 20 câu hỏi có citation | ☐ |
| 22–23 | Agent core v1 | 🤖✋ | hội thoại 5 lượt giữ ngữ cảnh | ☐ |
| 24 | Eval + observability → **M2** | 🤖 | `eval.py` in bảng điểm | ☐ |
| 25 | Chat streaming | 🤖✋ | token hiện dần + citation | ☐ |
| 26 | On-device OCR + Hybrid | 🤖✋ | máy bay vẫn trả kết quả | ☐ |
| 27 | Memory + nhắc thuốc | 🤖✋ | nhận được thông báo nhắc | ☐ |
| 28 | MCP servers | 🤖✋ | gọi tool từ MCP Inspector | ☐ |
| 29 | An toàn/riêng tư/ổn định | 🤖✋ | 8 tình huống xấu không crash | ☐ |
| 30 | Release → **v1.0.0** | ✋🤖 | `.aab` release build OK | ☐ |

---

## 9. Phụ lục D — Phương án B khi HẾT quota AI

- **Việc không cần AI, làm tiếp ngay được:** Day 0, 2, 3 (phần chạy lệnh + CI), 4, 12 (test tay), 14 (chạy test + viết README), 15 (docker), 16–17 (test phân quyền), 29, 30.
- **Việc nên gom vào 1 phiên AI duy nhất:** Day 5→9 (refactor lớn) và Day 13 (i18n hàng loạt).
- **Nguồn AI thay thế:** Copilot/Codeium free tier cho việc nhỏ; Gemini/ChatGPT free cho câu hỏi ngắn. Luôn **dán nội dung file + log lỗi nguyên văn** thay vì mô tả bằng lời.
- **Nếu bị chặn hoàn toàn:** đi theo thứ tự số trong mỗi ngày; chỉ cần xong phần ✋ thì project vẫn tiến (P0 và phần lớn P1 là ✋).
- **Nguyên tắc chống mất việc:** commit sau mỗi task, không để quá 1 ngày không commit.

---

## 10. Phụ lục E — Rủi ro & cách xử lý

| Rủi ro | Xác suất | Cách xử lý |
|---|---|---|
| `flutter upgrade` vỡ Gradle/CocoaPods | Cao | Dành Day 4 làm buffer; ghi vào `docs/TROUBLESHOOTING.md` |
| Server Flask thiếu endpoint | Trung bình | Day 15 tạo Core mới đúng contract ở mục 1; client không đổi |
| Không có nguồn dữ liệu thuốc hợp pháp | Trung bình | Chỉ dùng nguồn được phép; luôn hiển thị nguồn + ngày cập nhật |
| Chi phí LLM vượt kiểm soát | Trung bình | Quota/user, cache câu trả lời, model nhỏ cho việc đơn giản, dashboard chi phí từ Day 24 |
| Agent trả lời sai về y tế | Cao nếu không kiểm soát | Grounded-only + citation + từ chối + eval chạy mỗi lần đổi prompt |
| Hết quota AI giữa giai đoạn | Cao | Áp dụng Phụ lục D |

---

---

## 11. NHẬT KÝ THỰC THI

### 21/09/2026 — Day 0 ✅ + Day 1 ✅

**Môi trường ghi nhận:** Flutter **3.19.4** stable / Dart **3.3.2** · git **2.55.0.windows.3** · identity `Tuan Anh <davidpham01104@gmail.com>` · Windows 10.0.26200 · locale en-US.

**Day 0 — baseline & version control**
- `git init -b main` → 2 commit: `8a74b8f` *chore: baseline before refactor* và `4f30d26` *docs: add day-by-day roadmap (P0-P3)*; working tree sạch.
- `flutter doctor`: Flutter ✓ · Windows ✓ · Chrome ✓ · Network ✓ · **Android toolchain ✗ (thiếu `cmdline-tools`)** · Android Studio **chưa cài** · Visual Studio **chưa cài** → không build được Windows desktop · 3 device khả dụng (Windows/Chrome/Edge) nhưng **không có emulator hay máy Android**.

**Day 1 — gỡ blocker build**
- Chọn **phương án (B)**: giữ Flutter 3.19.4, hạ `environment.sdk` `^3.5.3` → `^3.3.0` (trước đây SDK constraint cao hơn Dart đang cài nên `pub get` fail 100%).
- Xoá 11 dependency không dùng + 3 file rỗng; giữ `http` để code hiện tại vẫn compile; thêm sẵn `dio`, `flutter_riverpod`, `flutter_secure_storage`, `shared_preferences`, `intl` cho P1.
- `Remove-Item pubspec.lock` → `flutter pub get` = **`Changed 79 dependencies!`** → `pubspec.lock` mới: **9 direct deps** (đủ khai báo), `sdks: dart >=3.3.0, flutter >=3.19.0`. ✅ **DoD đạt** — project đã resolve được dependency lần đầu tiên.
- `flutter analyze` = **38 issues** (0 warning + 38 info: `prefer_const_constructors`, `use_build_context_synchronously`, `unused_element`, `avoid_print`, `library_private_types_in_public_api`) → dọn ở Day 10.
- `flutter test` = **fail 1 test** (`Counter increments smoke test`, `test/widget_test.dart:19`) đúng như dự đoán → viết lại ở Day 3.
- **Phát sinh ngoài kế hoạch:** `pubspec.yaml` được thêm block `assets: - assets/models/` (không phải do phiên này sửa) trong khi thư mục chưa tồn tại → analyzer warning `asset_directory_does_not_exist`. Đã xử lý bằng cách tạo `assets/models/README.md` (ghi rõ model `.tflite` dùng từ Day 26) → warning biến mất (39 → 38 issues).

**Chênh lệch có chủ đích so với kế hoạch:** giữ `http` tới Day 8–9 (thay vì xoá ở Day 1) để P0 vẫn build được; `cached_network_image` thêm ở Day 12 khi thực sự cần.

**Chặn cần bạn gỡ (ảnh hưởng DoD Day 2):** máy chưa có Android emulator/thiết bị và thiếu `cmdline-tools` → không ai verify được `flutter run`. Cần: cài `cmdline-tools;latest` (hoặc cài Android Studio) + tạo emulator, hoặc cắm máy thật bật USB debugging.

**Day 2 — cấu hình nền tảng (✅ code xong, verify `flutter run` còn treo vì thiếu JDK/AVD)**
- Tạo `android/app/src/debug/res/xml/network_security_config.xml` (cleartext chỉ cho `10.0.2.2`/`localhost`/`127.0.0.1`) + trỏ `android:networkSecurityConfig` trong **debug manifest** → bản release không có file này nên vẫn bắt buộc HTTPS.
- `AndroidManifest.xml`: thêm `CAMERA` + `<uses-feature camera required="false">`; **không** thêm `READ_MEDIA_IMAGES` (image_picker 1.x dùng Photo Picker → không cần; tránh quyền thừa).
- `Info.plist` (iOS): `NSPhotoLibraryUsageDescription` + `NSCameraUsageDescription` (thiếu là **iOS kill app** khi chọn ảnh) + ATS `NSExceptionDomains` cho `localhost`/`127.0.0.1` (không dùng `NSAllowsArbitraryLoads`).
- macOS: thêm `com.apple.security.network.client` vào cả Debug và Release entitlements.
- `gradle.properties`: xoá `android.enableJetifier=true`; `build.gradle`: `minSdk = 26`, `namespace`/`applicationId` = `com.medassist.app`, **pin `ndkVersion`** về bản đã cài (`26.3.11579264`) vì NDK mặc định 25.1.8937393 không có trên máy.
- `MainActivity.kt` chuyển sang `kotlin/com/medassist/app/` (đổi `package` cho khớp `namespace`).
- **Kiểm chứng:** parse XML/plist **OK 6/6 file**; commit `ec6346d` (10 file).
- **BLOCKER chưa verify được:** `flutter build apk --debug` → `ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH` → **máy chưa cài JDK nào** (đã quét: `Program Files\Android\Android Studio\jbr`, `Program Files\Java`, `Eclipse Adoptium`, `Microsoft`, `Zulu`, `.jdks`, VS Code extensions, registry `JavaSoft` → **không có**). Ngoài ra `flutter emulators` → *No emulators available* (đã có system-image android-33/34/35 nhưng chưa tạo AVD; mà tạo AVD cần `avdmanager` thuộc `cmdline-tools` đang thiếu).

**Day 3 — test & CI (✅ hoàn tất và đã verify)**
- Thay test counter mặc định → smoke test khởi động app; thêm `test/screens/login_screen_test.dart` (3 test) và `test/screens/medicine_search_screen_test.dart` (1 test) → **tổng 5 test**.
- `dart format .` → format lại 5 file (để CI không đỏ lần push đầu); `flutter analyze` → **38 info, 0 warning, 0 error**.
- `flutter test` → **`All tests passed!` (5/5)**.
- Chạy đúng 3 lệnh của CI ở local: format check **exit 0** (0 changed) · `flutter analyze --no-fatal-infos` **exit 0** · `flutter test` **exit 0** → **CI sẽ xanh ngay lần push đầu**.
- Tạo `.github/workflows/ci.yml` (ghim Flutter 3.19.4); commit `5f68469` (9 file).

**Việc cần bạn làm để qua được M0 (2 việc, ~15 phút, không cần AI):**
```powershell
# 1) Cài JDK 17 (Temurin) rồi set JAVA_HOME — bắt buộc để build Android
winget install --id EclipseAdoptium.Temurin.17.JDK -e
# sau khi cài, mở PowerShell MỚI và kiểm tra: java -version
# nếu chưa tự có JAVA_HOME: setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot"

# 2) Tạo & chạy AVD (cần cmdline-tools để có avdmanager)
#    Cách A (khuyến nghị): cài Android Studio -> Device Manager -> Create Device (API 34/35)
#    Cách B (dòng lệnh): cài cmdline-tools rồi:
flutter emulators --create --name pixel_api34
flutter emulators --launch pixel_api34
flutter devices      # phải thấy emulator-5554
```
Sau đó `flutter run -d emulator-5554` → app phải vào màn Đăng nhập (đây là DoD còn lại của Day 2 + M0).

**Kế tiếp (Day 4 — buffer/M0):** dọn nốt 38 lint info nếu muốn (đã dự kiến ở Day 10), ghi `docs/TROUBLESHOOTING.md` các lỗi môi trường vừa gặp, rồi chốt **Milestone M0** sau khi bạn chạy được `flutter run`.

---

*Đây là kế hoạch sống: cập nhật lại mỗi cuối ngày (tick Phụ lục C). Khi bạn ra lệnh thực thi, mặc định tôi sẽ làm **Day 0 → Day 4 (P0)** trước, rồi dừng ở M0 để bạn tự kiểm tra trước khi đi tiếp P1.*

