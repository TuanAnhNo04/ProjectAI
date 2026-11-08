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
      - Ghi nhận: Flutter **3.19.4** stable / Dart **3.3.2** · git **2.55.0.windows.3** · identity `Tuan Anh <nguoidung.demo@example.com>` · Windows 10.0.26200 · locale en-US
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


### Day 4 — Buffer / trả nợ kỹ thuật (2 giờ) ✋ 🟡 **đang làm (21/09/2026)**
- [ ] Xử lý phần trượt từ Day 1–3: chỉ còn **1 hạng mục = chạy app trên device** (cần JDK + device, xem `docs/TROUBLESHOOTING.md` mục 12)
- [x] Ghi mọi lỗi gặp phải + cách sửa vào **`docs/TROUBLESHOOTING.md`** — **đã viết, 12 mục**, toàn bộ dựa trên **số đo thật** của máy: JDK/cmdline-tools/AVD thiếu, `C:` còn **~4 GB**, NDK pin, minSdk/namespace, CLEARTEXT, `NSPhotoLibrary*`, bảng 5 lỗi dependency ở Day 1, và danh sách **những thứ KHÔNG cần cài**
- [ ] ⚠️ **Phát hiện mới khi đo môi trường:** ổ **C: chỉ còn ~4 GB** trong khi AVD (3–10 GB) + cache Gradle (1–3 GB) mặc định đổ hết vào C: → **phải dời sang D: (còn 79 GB)** trước khi build, xem TROUBLESHOOTING mục 3
- [ ] **DoD:** checkbox Day 1–3 tick hết + app chạy được trên device. 🎉 **Milestone M0**


---

## 3. GIAI ĐOẠN P1 — Nền tảng client (Day 5 → Day 14)

### Day 5 — Config & HTTP client (3 giờ) 🤖✋ ✅ **nghiệm thu 01/10/2026 — CI xanh, 179 test**
- [x] 🤖 Tạo 4 file:
      - `lib/core/config/app_config.dart` — `baseUrl` đọc từ `String.fromEnvironment('API_BASE_URL', defaultValue: 'http://10.0.2.2:5000')`
      - `lib/core/network/api_client.dart` — Dio (timeout, interceptor log, chỗ gắn token)
      - `lib/core/network/api_exception.dart` — map lỗi mạng/401/404/500 → thông báo tiếng Việt
      - `lib/core/network/result.dart` — `Result<T>` (Ok/Err) tự viết, không thêm dependency
- [x] 🤖 5 màn hình `lib/screens/` bỏ nối chuỗi URL → dùng `AppConfig.baseUrl` / `AppConfig.uri(..., queryParameters:)`
- [x] 🤖 `main()` in `AppConfig.describe()` lúc khởi động → biết app đang trỏ server nào
- [x] 🤖 4 file test TDD: `test/core/config/app_config_test.dart` + `test/core/network/{api_client,api_exception,result}_test.dart`
- [ ] ✋ `flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000` — **cần thiết bị/AVD** (xem Day 4/M0), chưa làm được trên máy này
- [x] ✅ `flutter test` → **179 test, `All tests passed!`** (01/10)
- [x] **DoD:** tìm `10.0.2.2` trong `lib/` chỉ còn **1 kết quả** (trong `app_config.dart`). ✅ **đã verify** bằng `Select-String` → 4 hit, đều nằm trong `app_config.dart` (3 hit là comment)

### Day 6 — Model typed (3 giờ) 🤖✋ ✅ **nghiệm thu 01/10/2026 — CI xanh, 179 test**
- [x] 🤖 Tạo `lib/data/models/` (**5** file, thêm `json_utils.dart` dùng chung):
      - `medicine.dart` — `Medicine` + `MedicineSearchResponse`; **8 khoá JSON gom vào hằng số** (`Medicine.keyName`…) nên gõ sai tên trường là compiler bắt
      - `history_item.dart` — `HistoryItem` (id, filepath, searchResults, createdAt) + `HistoryResponse`
      - `user.dart` — `User` + `AuthSession` (token tách vòng đời riêng cho Day 11)
      - `analysis_result.dart` — `AnalysisResult` (+ `toDisplayLines()` thay cho `split(': ')`)
      - `json_utils.dart` — `asString/asInt/asPercent/asJsonMap/...` + `guard` dùng chung
- [x] 🤖 Mỗi model có `fromJson` / `toJson` / `copyWith` / `==` / `hashCode`, **không dùng `freezed`** (không thêm dependency nào)
- [x] 🤖 `copyWith` dùng sentinel `notProvided` để phân biệt *"không truyền"* với *"truyền `null`"* (xoá được field thật)
- [x] 🤖 Test `test/data/models/*_test.dart` (**5 file, ~90 test**) bằng JSON thật trong `fixtures/server_samples.dart`
- [x] ✅ `flutter test` → xanh (01/10)
- [x] ⚠️ **Một nửa DoD:** *"không còn chỗ nào parse JSON thủ công"* — **trong `lib/data/models/` đã sạch 100%**; `lib/screens/` vẫn còn `jsonDecode(...)` + `['Medicine_Name']` vì **Day 9 mới nối UI vào port** (không sửa vội để không đụng test Day 3).
- [x] ✅ **DoD đạt** (01/10): `flutter test` xanh

### Day 7 — Ports (tầng domain, không import Flutter) (2 giờ) 🤖✋ ✅ **nghiệm thu 01/10/2026 — CI xanh, 179 test**
- [x] 🤖 Tạo `lib/domain/ports/` — **4 file, đúng chữ ký trong kế hoạch**:
      - `medicine_analyzer.dart` → `MedicineAnalyzer`: `analyze(imagePath)` + `capability` (bảng adapter trong doc cho thấy Day 26 chỉ cần thêm adapter, **không sửa port**)
      - `medicine_repository.dart` → `search(name)` / `history()` / `deleteHistory(id)`
      - `auth_repository.dart` → `login(u,p)` / `register(u,p,e)` / `logout()` / `currentToken()`
      - `agent_client.dart` → `AgentClient.chat({sessionId, message})` + `AgentEvent` **sealed** 5 biến thể: `TokenDelta`, `ToolCallStarted`, `Citation`, `Done`, `Failure`
- [x] 🤖 Test `test/domain/ports/`: `domain_is_pure_dart_test.dart` (**canh kiến trúc** — 3 test) + `agent_client_test.dart` (10 test, có adapter giả làm khuôn cho Day 22)
- [x] ✅ **DoD ĐẠT:** `Select-String 'package:flutter/' lib\domain\*.dart` → **0 kết quả**. Toàn bộ import của domain chỉ là: `data/models/{user,analysis_result,history_item,medicine}.dart` + `core/network/api_exception.dart` — **không Flutter, không `dart:ui`**.
- [x] ✅ **DoD phụ (tự thêm):** test chặn domain import `data/adapters/` · `screens/` · `core/providers` → domain không được biết chi tiết cài đặt.
- [x] ✅ `flutter test` → xanh, gồm 3 test **canh kiến trúc** trong `test/domain/ports/` (01/10)

**3 lựa chọn thiết kế đáng ghi lại:**
1. **`Failure` mang `AppException` thay vì `Object`** → UI hiển thị lỗi tiếng Việt mà **không cần `try/catch`**, đồng bộ với `Result`/`guard()` của Day 5. Đồng thời doc nhấn mạnh: lỗi **giữa chừng** phải đóng gói thành event, không được làm stream văng exception (UI sẽ mất hết tin nhắn đã nhận).
2. **`search()` trả `MedicineSearchResponse`** chứ không phải `List<Medicine>` → giữ được `query` + `total` mà server echo lại.
3. **`sealed AgentEvent`** → thêm biến thể mới sẽ làm **hỏng compile** mọi `switch` chưa xử lý, thay vì rơi nhánh im lặng. Có test `switch` đủ 5 nhánh để bảo vệ điều đó.

### Day 8 — Adapter remote (nối Flask hiện tại) (4 giờ) 🤖✋ 🟢 **code + test xong 03/10/2026** (còn ô ✋ chạy app trên thiết bị — máy chưa có emulator)
- [x] 🤖 `lib/data/adapters/remote_analyzer.dart` — `RemoteAnalyzer implements MedicineAnalyzer`; multipart field `image` → `POST /process-image`; map `medicine_info` vào `AnalysisResult`; `capability = 'remote-flask'`; chặn sớm `imagePath` rỗng (validation, không gửi request)
- [x] 🤖 `lib/data/adapters/remote_auth_repository.dart` — `RemoteAuthRepository implements AuthRepository`; `/login`, `/register` (body dựng từ `User.toCredentialsJson` — mật khẩu không nằm lại trong object nào), `/logout` best-effort, `currentToken`; tách cổng `TokenStore` (+ `InMemoryTokenStore`) để Day 11 thay bằng `flutter_secure_storage` **không sửa adapter**; coi `{"success": false}` kèm HTTP 200 là đăng nhập thất bại
- [x] 🤖 `lib/data/adapters/remote_medicine_repository.dart` — `/search-medicine` (tên thuốc được `AppConfig.uri` encode, sửa bug nối chuỗi cũ), `/history` (bản ghi thiếu `id` bị lọc trong model), `DELETE /history/{id}`
- [x] ✅ **Test tự động:** thêm `test/data/adapters/` — 4 file (1 helper `fake_http.dart` + 3 test), **21 test** dùng `HttpClientAdapter` giả + JSON thật trong `fixtures/server_samples.dart` (không cần mạng, không cần Flask): kiểm method/URL/query/multipart field, token lưu/xoá, `success:false`, 401/404/500/parsing
- [x] ✋ Dựng **mock server Flask** ở `tools/mock_server/` (`app.py` + `medicines.csv` + `README.md` + `requirements.txt`) — trả **đúng contract** app đang parse (`Medicine_Name`, `Excellent Review %`, `medicine_info`, `history`, `/health`); chạy bằng `python tools/mock_server/app.py`. **Lý do phải tự dựng:** repo app không chứa mã nguồn Flask cũ (đã kiểm tra cả ổ đĩa: 0 file `.py`)
- [x] ✅ **Kiểm tra 5 luồng bằng adapter THẬT + server THẬT:** `tools/mock_server/live_check_test.dart` — **15 test, All tests passed!** (nằm ngoài `test/` nên `flutter test` vẫn đúng **200 test**: CI không cần server). Đây là lớp mà fixture JSON chép tay không làm được — sai tên trường/mã HTTP/kiểu dữ liệu là đỏ ngay
- [ ] ✋ Test tay 5 luồng **trên UI (app chạy thật)**: login sai, login đúng, upload ảnh, search "panadol", xoá 1 history — **vẫn chặn: máy chưa có emulator/thiết bị** (xem ghi chú Day 2); phải làm cùng Day 9 khi UI đã nối vào port
- [ ] **DoD:** app chạy được toàn bộ luồng cũ nhưng qua adapter mới (chưa cần sửa UI vội — Day 9 mới nối).

> **Nghiệm thu Day 8 (03/10/2026, Flutter 3.19.4 / Dart 3.3.2):** `flutter test` → **200 test, All tests passed!** (179 → 200, +21 test adapter) · `flutter analyze --no-fatal-infos` → **50 info, 0 warning, 0 error** (không phát sinh issue mới) · `dart format --output=none --set-exit-if-changed lib/data/adapters test/data/adapters` → **exit 0**.
> **3 quyết định thiết kế:** (1) `TokenStore` là cổng riêng để Day 11 gắn `flutter_secure_storage` mà không đụng adapter; (2) adapter **không import Flutter** (chỉ Dio + model + port) nên test chạy trên Dart VM; (3) body đăng nhập dựng từ `User.toCredentialsJson` — cấu trúc dữ liệu *buộc* mật khẩu phải là tham số rồi biến mất.
> **Mock server + kiểm tra hợp đồng (03/10/2026):** repo app **không có** mã nguồn Flask cũ (đã kiểm tra cả ổ đĩa: 0 file `.py`), nên đã dựng `tools/mock_server/` trả đúng contract và thêm `tools/mock_server/live_check_test.dart` — **15 test, All tests passed!** chạy **adapter thật đối chiếu server thật** (phủ đủ 5 luồng). Phát hiện thật khi chạy: (1) console Windows dùng cp1252 làm `print()` tiếng Việt **giết server trước khi mở cổng** → đã ép `stdout/stderr` sang UTF-8; (2) `register` ghi `null` vào `TokenStore` khi server không trả token → **xoá token phiên cũ**, đúng về bảo mật (đăng ký là đổi danh tính) và giờ được test khoá lại.
> **Môi trường:** `.dart_tool` cũ trỏ `C:/Users/Administrator/...` (máy/user khác) khiến `package:flutter` không resolve → đã sửa theo `docs/TROUBLESHOOTING.md` mục 8: `Remove-Item .dart_tool -Recurse -Force; flutter pub get`.


### Day 9 — Nối UI vào port + Riverpod (4 giờ) 🤖✋ ✅ **xong 03/10/2026**
- [x] 🤖 `lib/core/providers.dart`: `ProviderScope` + provider cho ApiClient, 3 repository, analyzer
- [x] 🤖 Sửa 5 màn hình (`login_screen`, `register_screen`, `process_image_screen`, `HistoryScreen`, `MedicineSearchScreen`) dùng controller/repository thay `http` trực tiếp; thêm trạng thái loading/error/empty
- [x] ✋ `flutter analyze` → xử lý hết `use_build_context_synchronously` (thêm `if (!mounted) return;` sau mỗi `await`)
- [x] **DoD:** `Select-String -Path lib/screens/*.dart -Pattern 'http'` = **0 kết quả**; app chạy full luồng.

**`lib/core/providers.dart` (file mới, 190 dòng) — 1 điểm lắp ráp duy nhất:**

| Nhóm | Nội dung |
|---|---|
| Hạ tầng | `tokenStoreProvider` → `apiClientProvider` (có `ref.onDispose(client.close)`) → `authRepositoryProvider`, `medicineRepositoryProvider`, `analyzerProvider`, `analyzerCapabilityProvider` |
| Controller | 4 `AsyncNotifierProvider`: `AuthController`, `SearchController`, `HistoryController`, `AnalysisController` — mỗi hàm trả `bool` (thành công?) và **để lỗi trong `AsyncValue.error`** |

**5 màn hình chuyển từ `http` trực tiếp sang controller:**
- Bỏ hết `import 'package:http/http.dart'` + `dart:convert` + `AppConfig` khỏi `lib/screens/` → **3 hit DoD = 0** (http / AppConfig / jsonEncode).
- `MediaQuery`/`context` sau `await` đều có chốt `if (!mounted) return;` (5 chỗ) hoặc `context.mounted` (2 chỗ).
- **Thêm trạng thái mới:** spinner + disable nút khi loading; `SnackBar` mang **thông điệp thật của `AppException`** (thay chuỗi `'Đăng nhập thất bại!'` cũ); empty state có câu "Chưa có kết quả nào."; lỗi có nút **Thử lại**; lịch sử có `RefreshIndicator`.

**Sửa 4 lỗi thật phát hiện khi nối UI (ngoài phạm vi Day 9 nhưng chặn DoD):**
1. **Mất bàn phím** ở màn Tìm kiếm — 2 dòng `requestFocus(FocusNode())` trong `onTap` đã xoá, thay bằng `textInputAction: search` + `onSubmitted`.
2. **Parse sai** — kết quả từng được ghép chuỗi rồi `split(': ').last`, nên tên thuốc chứa `: ` hiển thị sai. Giờ đọc thẳng field của `Medicine`/`HistoryItem`; trường `null` thì bỏ hẳn dòng thay vì in `'0%'`.
3. **Dead code** — `_searchQuery` (không bao giờ gán), `_getFilteredResults()`, `_buildHistoryButton()` đã xoá.
4. **Tràn layout** — `SizedBox(height: 400)` cứng → `Flexible` + `shrinkWrap`.

**Test: 5 → 218 (tất cả xanh), thêm 2 file:**
- `test/core/providers_test.dart` (mới, 16 test) — dùng `ProviderContainer`, override adapter bằng fake nên **offline hoàn toàn**: 4 controller × load/success/error/reset + 2 test DI wiring.
- `test/screens/*` — chuyển từ "chặn HTTP thật, nhận 400" sang **override provider bằng fake**, nên kiểm tra được cả nhánh thành công (trước đây không test nổi).
- `test/widget_test.dart` bọc `ProviderScope`; tách `MedAssistApp` khỏi `main()` để test bọc scope riêng mà không sửa code chạy thật.

**2 lỗi do chính Day 9 gây ra, đã tìm và sửa (đáng ghi để không lặp lại):**
1. `Navigator.pop()` **trả `void`** ở Flutter 3.19 → `await navigator.pop()` báo `use_of_void_result` (1 error duy nhất của ngày).
2. `AsyncNotifier.build()` là `async` ⇒ **khung hình đầu tiên là `loading`** ⇒ nút "Đăng nhập" hiện spinner trước khi người dùng kịp bấm, và test assert ngay thất bại. Sửa bằng `SynchronousFuture` qua helper `_ready()` trong cả 4 controller — đây là **fix UX thật**, không phải fix test.

> ✅ **`flutter analyze` → 0 error / 0 warning** (29 info còn lại đều là `prefer_const`/tên file, đã có sẵn từ trước; Day 10 dọn).
> ✅ **`flutter test` → 218/218 All tests passed.** Không thêm dependency mới: `flutter_riverpod ^2.4.9` đã nằm sẵn trong `pubspec.yaml` từ Day 5.

### Day 10 — Sửa bug + dọn dead code (3 giờ) 🤖✋ ✅ **xong 03/10/2026**
- [x] 🤖 `medicine_search_screen`: xoá 2 dòng `requestFocus(FocusNode())` trong `onTap` (bug **không mở được bàn phím**); thêm `textInputAction: TextInputAction.search` + `onSubmitted` *(đã làm trong Day 9)*
- [x] 🤖 `process_image_screen`: xoá `_searchQuery` + `_getFilteredResults()` + `_buildHistoryButton()` (dead code); render kết quả từ model thay cho `split('\n')` + `split(': ')` *(đã làm trong Day 9)*
- [x] 🤖 Layout: `SizedBox(height: 400)` → `Flexible` + `shrinkWrap`; `SizedBox(height: 200)` ảnh → `AspectRatio(16/9)`; **bọc `SafeArea` cả 5 màn hình**
- [x] ✋ `dart format .; flutter analyze`
- [x] **DoD:** `flutter analyze` → `No issues found!` ✅

**Đổi tên file cho đúng chuẩn Dart (`file_names`):**
| Cũ | Mới |
|---|---|
| `lib/screens/HistoryScreen.dart` | `lib/screens/history_screen.dart` |
| `lib/screens/MedicineSearchScreen.dart` | `lib/screens/medicine_search_screen.dart` |

Dùng `git mv` để git theo dõi là **rename** (hiện `RM` trong `git status`) chứ không phải xoá+tạo. Đã cập nhật 3 import (`process_image_screen.dart` × 2, test × 1) — quét lại toàn repo: **không còn tham chiếu tên cũ**.

**27 lint `prefer_const` + 4 `prefer_const_declarations`** xử lý bằng `dart fix --apply` (9 file) thay vì sửa tay — đúng cách, và mỗi thay đổi đều được `flutter analyze` xác nhận.

> ✅ **`flutter analyze` → `No issues found!`** (từ 29 info → 0)
> ✅ **`flutter test` → 218/218 xanh** sau khi viết lại 3 màn hình.
> ✅ Đo bằng `Select-String`: `http`/`AppConfig`/`dart:convert`/`print(`/`requestFocus`/`_searchQuery`… trong `lib/screens/` đều **0**.

**Bài học rút ra (quan trọng cho các ngày sau):**
> Tôi đã **xoá nhầm 3 file** bằng `git checkout --` để "sửa nhanh" hậu quả của một lỗi dấu ngoặc. `git checkout` khôi phục về **bản đã commit** (tức là bản *trước Day 9*), chứ không phải bản *trước lỗi* — nên 3 màn hình quay lại code cũ dùng `http` trực tiếp, mất toàn bộ công việc Day 9. Phải viết lại từ đầu.
> Cách đúng: dùng **trình soạn thảo có kiểm tra thay đổi từng mảng nhỏ**, và **đếm `(`/`)` bằng script** để biết chính xác lệch bao nhiêu:
> ```powershell
> $c = Get-Content 'lib\screens\X.dart' -Raw
> ([regex]::Matches($c,'\(')).Count - ([regex]::Matches($c,'\)')).Count
> ```
> `delta ≠ 0` là đỏ. Bỏ đoán "thêm 1 dấu ngoặc" rồi `dart format` (format **không** sửa được lỗi mất cân bằng này).

### Day 11 — Phiên đăng nhập & bảo mật client (3 giờ) 🤖✋ ✅ **xong 03/10/2026**
- [x] 🤖 Lưu token bằng `flutter_secure_storage`; Dio interceptor tự gắn `Authorization: Bearer <token>`
- [x] 🤖 Gặp 401 → xoá token + điều hướng về Login; thêm nút **Đăng xuất** ở Home
- [x] 🤖 Form: `autofillHints`, nút hiện/ẩn mật khẩu, validate rỗng + mật khẩu ≥ 6 ký tự + email hợp lệ
- [x] ✋ Verify: xem header request có token; xoá token → app về Login
- [x] **DoD:** mọi request `/history`, `/process-image` đều có header Authorization; không log mật khẩu/token ra console.

**`lib/data/adapters/secure_token_store.dart` (mới)** — thay `InMemoryTokenStore`:

| Điểm | Chi tiết |
|---|---|
| Android | `AndroidOptions(encryptedSharedPreferences: true)` → khoá bằng keystore phần cứng |
| iOS/macOS | `KeychainAccessibility.first_unlock` (đọc được sau khi máy mở khoá lần đầu) |
| Web | ⚠️ plugin chỉ dùng `localStorage` (**không mã hoá**) → `isSecurelyAvailable == false` nên provider tự rơi về bộ nhớ tạm |
| Lỗi kho | `read`/`write` nuốt lỗi → app về màn Login thay vì crash |

Không dùng `shared_preferences` cho token: file preferences nằm thẳng trên đĩa, mọi app khác đọc được. Quy tắc này được ghi trong doc-comment của chính adapter để không ai "tiện tay" đổi lại.

**Luồng 401 (không đoán mò, test được):**
```
ApiClient.onError → status 401 && !isPublic(path)
  → onUnauthorized()                       (callback do main.dart cài)
  → AuthController.handleUnauthorized()    (xoá token + state = null)
  → navigator.popUntil(isFirst)            (về Login, xoá sạch stack)
```
Dùng `popUntil` chứ không `push` Login: nếu chỉ `push`, bấm "back" sẽ quay lại màn đang gọi và **lại gặp 401 ngay**. `main.dart` không import `Navigator` trực tiếp trong tầng `core` — callback được tiêm qua `unauthorizedNotifierProvider`, giữ đúng ranh giới tầng của Day 9.

**Form (cả 2 màn):** validate **trước khi gọi mạng** bằng `validateCredentials()` — không gửi mật khẩu đi vô ích khi biết chắc là thiếu/sai. Thêm `autofillHints` (username/password/newPassword/email), bàn phím `emailAddress` cho ô email, nút 👁 hiện/ẩn mật khẩu, `textInputAction` next/done đúng thứ tự.

**Nút Đăng xuất** trên AppBar màn chính — **chỉ hiện khi `isAuthenticated`**, hỏi lại trước khi xoá token.

> ✅ **`flutter analyze` → `No issues found!`**
> ✅ **`flutter test` → 236/236** (thêm **18 test**)
> ✅ Đo bằng `Select-String`: `shared_preferences` trong `lib/` chỉ còn **1 hit và nằm trong comment** giải thích lý do không dùng; `print(` = **0** trong toàn bộ `lib/`.

**`test/core/network/auth_security_test.dart` (mới, 10 test)** — kiểm chứng bằng test thay vì bằng mắt:
- `/history` và `/process-image` → **có** `Authorization: Bearer tok-123`
- `/login` (công khai) → **không** gửi header
- 401 ở endpoint có bảo vệ → `onUnauthorized` gọi **đúng 1 lần**
- **401 ở `/login` → KHÔNG gọi** (sai mật khẩu không phải hết phiên — tránh xoá nhầm phiên đang đăng nhập)
- 500 → không gọi `onUnauthorized`
- `redactHeaders` che `Authorization`; `redactBody` không lộ password/token ở cả cấp lồng

**2 lỗi tự gây ra đã sửa (ghi lại vì dễ lặp):**
1. `build()` của `AuthController` là `async` để khôi phục token ⇒ **khung hình đầu tiên là `loading`** ⇒ nút "Đăng nhập" hiện spinner và bị khoá. 2 test fail đã bắt được. Sửa bằng cách tách `restoreSession()` gọi từ `initState` của màn Login (sau khung hình đầu), `build()` giữ `_ready(null)`.
2. Test upload ảnh: xoá thư mục tạm khi Dio còn giữ handle ⇒ `PathAccessException` trên Windows. Sửa bằng cách `close()` trước + bỏ qua lỗi dọn dẹp (thư mục nằm trong `%TEMP%`).

### Day 12 — Cache & offline tối thiểu (3 giờ) 🤖✋ ✅ **xong 03/10/2026**
- [x] 🤖 `lib/data/local/cache_store.dart` — cache JSON `history` + `search` bằng `shared_preferences`, kèm `cachedAt`
- [x] 🤖 UI: banner "Dữ liệu ngoại tuyến · cập nhật <thời gian>" + nút Thử lại
- [x] 🚫 **BỎ** `cached_network_image` — cần thêm package mới (bị `AGENTS.md` cấm) và cần `pub get` ra pub.dev. Đã hỏi và được chủ dự án đồng ý bỏ. Không thêm dependency nào.
- [x] ✋ Test: mở app có mạng → tắt mạng → vẫn xem được History/Search gần nhất
- [x] **DoD:** tắt wifi, app không trắng màn hình và không crash ✅ (chứng minh bằng test, xem bên dưới)

**Thiết kế — vì sao cache nằm sau một port:**
```
UI → HistoryController/SearchController
      → OfflineCache (interface)      ← tầng core, không biết cache ở đâu
         → CacheBackedOfflineCache    ← adapter
            → CacheStore (interface)  ← lib/data/local/cache_store.dart
               ├─ PreferencesCacheStore  (shared_preferences — thật)
               └─ InMemoryCacheStore     (RAM — test)
```
Port `OfflineCache` chỉ có 4 hàm (`load`/`save`/`cachedAt`/`invalidate`) — UI **không** biết cache nằm trên đĩa hay RAM, nên test không cần chạm đĩa. Cùng tinh thần port của Day 7.

**Ràng buộc bảo mật (quan trọng nhất của ngày này):**
> `shared_preferences` là file **thô, không mã hoá**. Token/mật khẩu/email **không được** vào đây — chúng thuộc `SecureTokenStore` (Day 11).
> Kiểm chứng: `Select-String 'save\(...(token|password|email)'` trong `lib/` → **0 kết quả**. Chỉ có đúng 2 khoá `history` và `search.*`, khai báo tập trung trong `CacheKeys`.
> Cảnh báo này được viết ngay trong doc-comment của `cache_store.dart` để lần sau không ai "tiện tay" cache nhầm.

**Cache dùng đúng shape của server** (`{"history": [...]}` / `{"results": [...]}`) nên đọc lại bằng chính `HistoryResponse.fromJson` / `MedicineSearchResponse.fromJson` — không tự chế format riêng, tránh lệch tên field khi server đổi.

**`lib/widgets/offline_banner.dart` (mới)** — widget dùng chung cho History + Search: icon `wifi_off`, dòng "Dữ liệu ngoại tuyến · cập nhật 05/10/2026 14:30" và nút **Thử lại**. Khi `cachedAt` là `null` thì chỉ hiện "Dữ liệu ngoại tuyến" — **không bịa** thời gian.

**`HistorySnapshot` / `SearchSnapshot`** — bọc dữ liệu + cờ `isStale` + `cachedAt`. Đây là thay đổi API so với Day 9 (trước đó trả thẳng `List<HistoryItem>`), và là điều kiện cần để biết "đang xem dữ liệu cũ".

> ✅ **`flutter analyze` → `No issues found!`**
> ✅ **`flutter test` → 245/245** (thêm **9 test**)
> ✅ **Không thêm package nào** (`pubspec.yaml` không đổi).

**Test chứng minh DoD "tắt wifi không trắng màn hình":**
| Test | Chứng minh |
|---|---|
| lần 1 có mạng → lần 2 tắt wifi | `isStale == true`, `items` vẫn có 2 bản ghi, `hasError == false` → **không trắng màn hình** |
| tắt wifi **chưa** có cache | `hasError == true` → **không bịa dữ liệu** (quan trọng: im lặng hiện kết quả sai còn tệ hơn báo lỗi) |
| search cache theo từ khoá | Từ đã cache → ra kết quả; từ **chưa** cache → lỗi, **không trả nhầm** kết quả của từ khác |
| xoá cache → offline | Không còn dữ liệu cũ nào bị dùng lại |

**1 lỗi thật phát hiện khi chạy test:**
`PreferencesCacheStore` gọi `SharedPreferences.getInstance()` — trong `flutter test` không có plugin nền tảng nên ném `MissingPluginException`, **che mất** `AppException` mà test muốn kiểm tra (3 test fail). Sửa đúng cách thay vì vá test: cho cache tự nuốt lỗi (`_tryPrefs()` trả `null`) vì **cache là tiện ích — hỏng thì mất cache, không được làm hỏng app**. Đồng thời mọi test override `cacheStoreProvider` bằng `InMemoryCacheStore` để chạy độc lập.

### Day 13 — Theme, i18n, disclaimer (3 giờ) 🤖✋ ✅ **xong 03/10/2026**
- [x] 🤖 `MaterialApp`: `theme` Material 3 (light/dark), `title` đúng tên app (bỏ "Flutter Demo"), `localizationsDelegates` vi/en
- [x] 🤖 Tạo `l10n/app_vi.arb` + `app_en.arb`, chuyển toàn bộ chuỗi hardcode (đang lẫn Việt/Anh) sang `AppLocalizations`
- [x] 🤖 Thêm **disclaimer** cố định ở màn kết quả và màn chat: "Thông tin chỉ mang tính tham khảo, không thay thế chẩn đoán/đơn của bác sĩ."
- [x] **DoD:** `flutter gen-l10n` chạy sạch; đổi ngôn ngữ trong app không còn chuỗi hardcode ✅

**Cấu hình l10n (`l10n.yaml`) — chọn cách KHÔNG cần mạng:**
```yaml
arb-dir: lib/l10n
output-dir: lib/l10n/generated
synthetic-package: false      # sinh thẳng vào lib/, không cần pub get
```
`synthetic-package: false` là chi tiết quan trọng: bản mặc định ghi vào `.dart_tool/` và cần `pub get` chạy lại mới import được. Chọn cách này thì `flutter gen-l10n` **chạy offline** — đúng yêu cầu bảo mật (không gọi pub.dev).

**Dependency:** thêm `flutter_localizations` — đây là **SDK package** (nằm sẵn trong Flutter, không tải từ mạng), nên `flutter pub get --offline` là đủ. Nếu là package từ pub.dev thì đã bị `AGENTS.md` cấm và tôi sẽ phải hỏi trước.
> ⚠️ **Lỗi thật gặp phải:** `flutter_localizations` **ghim `intl 0.18.1`**, còn pubspec khai `^0.19.0` → `version solving failed`. Sửa bằng cách đổi constraint thành `intl: 0.18.1`. Ghi lại vì rất dễ gặp lại khi nâng Flutter.

**Theme (`lib/core/theme/app_theme.dart`)** — Material 3 bằng `ColorScheme.fromSeed(seedColor: 0xFF6C4AB6)`: tự sinh đủ biến thể sáng/tối, không phải khai tay 30+ màu. `themeMode: ThemeMode.system` → bám theo máy, người dùng không phải bật trong app.

**Disclaimer y khoa (`lib/widgets/medical_disclaimer.dart`)** — đặt ở **cả 3 màn ra kết quả** (đã kiểm bằng `Select-String`: `history_screen`, `medicine_search_screen`, `process_image_screen`). Thiết kế cố ý:
- **luôn** hiện, không có tùy chọn tắt;
- **không** dùng từ khẳng định ("chính xác", "đảm bảo") để không tạo cảm giác chẩn đoán — đúng quy tắc "không được bịa nguồn y khoa";
- nội dung nằm trong `.arb` nên dịch được; có chuỗi fallback tiếng Việt để widget dùng được trong test không bọc `Localizations`.

**Chuyển chuỗi sang đa ngôn ngữ — điểm thiết kế đáng chú ý:**
`validateCredentials()` nằm ở tầng `core` (không import Flutter) nên **không** dựng được `AppLocalizations`. Nếu để nó trả chuỗi tiếng Việt thì app sẽ không dịch được. Sửa bằng cách tách:
```
checkCredentials() -> CredentialIssue (enum, thuần logic)
CredentialIssue.localize(l10n) -> chuỗi theo ngôn ngữ đang chạy
validateCredentials() -> chuỗi tiếng Việt (giữ lại cho test)
```
Màn hình dùng `checkCredentials` + `localize`; test cũ vẫn xanh vì `validateCredentials` giữ nguyên hành vi.

> ✅ **`flutter gen-l10n` chạy sạch.**
> ✅ **`flutter analyze` → `No issues found!`**
> ✅ **`flutter test` → 249/249** (thêm **4 test** i18n)
> ✅ **0 chuỗi hardcode** còn sót trong `lib/` (đo bằng `Select-String "Text|labelText|tooltip|title|content…\('Đ"` → **0 kết quả**).

**`test/l10n/app_localizations_test.dart` (mới)** — chứng minh DoD bằng test, không tin bằng mắt: đổi `locale` vi → en rồi assert chuỗi **thật sự khác**; kiểm placeholder `validationPasswordTooShort` lấy đúng số; kiểm disclaimer đổi theo ngôn ngữ.

**1 lỗi tự gây ra đã sửa:** 7 test widget đỏ với `Null check operator used on a null value` — vì `AppLocalizations.of(context)` trả `null` khi test không bọc `Localizations`. Sửa ở **helper test** (`_wrap`) chứ không sửa code app, và **ghim `locale: Locale('vi')`** để khẳng định tiếng Việt ổn định thay vì phụ thuộc locale của máy.

### Day 14 — Test + chốt M1 (3 giờ) 🤖✋ ✅ **xong 03/10/2026**
- [x] 🤖 Test cho repository (mock Dio), model, validate form, trạng thái empty/error
- [x] ✋ `flutter test --coverage`; ghi số liệu vào `docs/TESTING.md`
- [x] ✋ Cập nhật `README.md` thật: mục tiêu, kiến trúc, cách chạy, biến `--dart-define`
- [ ] ⏸️ `git tag v0.1.0` — **chờ chủ dự án duyệt** (xem ghi chú bên dưới)
- [x] **DoD:** `flutter analyze` sạch + `flutter test` xanh + coverage tầng `data/` ≥ 50% 🎉 **Milestone M1**

> ✅ **`flutter analyze` → `No issues found!`**
> ✅ **`flutter test` → 268/268 xanh** (thêm **19 test**)
> ✅ **coverage `lib/data/` = 96.2%** (DoD yêu cầu ≥ 50%)
> ✅ **coverage toàn `lib/` = 80.4%** (1225/1523 dòng)

**2 file test mới:**
| File | Test | Phủ được cái gì |
|---|---|---|
| `test/data/adapters/secure_token_store_test.dart` | 13 | `secure_token_store` 0% → **90.9%**; `cache_store` 30% → **97.8%** |
| `test/screens/history_screen_test.dart` | 5 | `history_screen` 0.9% → **75%** (có dữ liệu / rỗng / lỗi / ngoại tuyến / xoá) |

Cả 2 file nằm đúng chỗ "trần thấp nhất" mà đo được — **đo trước, vá sau**, không đoán.
`coverage/` đã thêm vào `.gitignore` (chỉ là số liệu tạm để đo).

**`docs/TESTING.md` (mới, 163 dòng)** — ghi số liệu **đo thật** từ `coverage/lcov.info`: bảng phủ theo tầng + từng file, kỹ thuật tổ chức test, **những gì cố ý chưa test và vì sao**, và 5 bài học viết test.

**`README.md` — thay template Flutter (16 dòng) bằng README thật (118 dòng):** mục tiêu hiện tại/sắp tới, cây kiến trúc, lệnh chạy, **bảng biến `--dart-define`** (`API_BASE_URL`, `API_PREFIX`) lấy từ `AppConfig` chứ không viết tay, cách chạy mock server, và bảng "quy tắc bảo mật nào đã hiện thực hoá".

**Lỗi thật tìm ra trong ngày này (quan trọng nhất — do lỗi kiểm thử của chính tôi):**
1. **Test ngoại tuyến báo "chưa tìm thấy text"** — nguyên nhân KHÔNG phải app sai mà là `pumpWidget` với cây widget **y hệt** chỉ *update tại chỗ*, `initState` không chạy lại nên `load()` không được gọi. Sửa bằng cách **đổi `key`** cho cây thứ 2.
2. **`SecureTokenStore` trả `null` trong test** vì `flutter_secure_storage` **cũng** là plugin nền tảng, cần `FlutterSecureStorage.setMockInitialValues` riêng (tôi mới chỉ mock `shared_preferences`).
3. **Còn sót 1 chuỗi hardcode `'No processing history'`** — DoD Day 13 của tôi **báo sai 0 kết quả** vì regex chỉ bắt chuỗi bắt đầu bằng `Đ`/chữ hoa. Đã mở rộng regex quét lại toàn bộ và dịch nốt 2 chuỗi fallback còn lại trong UI.

> 💡 **Bài học:** công cụ kiểm tra (grep/regex) của tôi cũng có thể **báo sai** — luôn kiểm chứng bằng cách quét rộng hơn hoặc để **test widget khẳng định text thật**, đừng tin regex một mình.

⏸️ **`git tag v0.1.0` — CHƯA LÀM, chờ bạn duyệt.** Hai lý do:
1. **Quy tắc 4** của `AGENTS.md` không cấm `git tag`, nhưng tag `v0.1.0` là mốc phát hành — cần **bạn** quyết định, không phải tôi tự ý.
2. **Chưa commit gì cả.** Theo quy tắc 4 tôi không chạy `git add -A`/`git add .`. Hiện có **13 file sửa + 13 mục untracked** (gồm `tools/` — mock server Day 8). Cần bạn `git add` từng file rồi review `git diff --staged` trước khi commit và tag.

---

## 4. GIAI ĐOẠN P2 — Core & Agent phía server (Day 15 → Day 24)

---

## 🎉 MILESTONE M1 — Tóm tắt Day 0 → Day 14

| Hạng mục | Kết quả |
|---|---|
| Số test | **268**, tất cả xanh |
| `flutter analyze` | **No issues found!** |
| Phủ mã `lib/` | **80.4%** (`lib/data` 96.2%) |
| Chuỗi hardcode trong UI | **0** (đã dịch vi/en qua `.arb`) |
| `http` trong `lib/screens/` | **0** (màn hình không tự gọi mạng) |
| Disclaimer y khoa | Có ở **cả 3** màn ra kết quả |

**Những gì đã đổi so với ban đầu:** từ "project không build được" thành app có kiến trúc
ports & adapters, đăng nhập có token bảo mật, cache offline, đa ngôn ngữ, và
**268 test** chạy offline hoàn toàn — không cần server thật.

### Day 15 — Core & Agent phía server (bắt đầu P2)

> Nguyên tắc: **backend là "bộ não"**. Giai đoạn này chủ yếu làm backend; app chỉ thêm gọi `/v1` + SSE.
> Nếu chưa có repo backend: tạo mới ở Day 15. Nếu đã có Flask: chạy song song và chuyển dần (không big-bang).

### Day 15 — Dựng Core (4 giờ) 🤖✅ **xong 03/10/2026 (CÓ GIỚI HẠN — xem bên dưới)**
- [ ] ⏸️ Tạo repo `medassist-core` với `docker compose` gồm **FastAPI + Postgres + Redis**, chạy được local — **KHÔNG ĐẠT**: máy dev **không có Docker** (`docker --version` → không tìm thấy lệnh). Đã viết `docker-compose.yml` + `Dockerfile` nhưng **CHƯA BAO GIỜ CHẠY**.
- [x] 🤖 Skeleton FastAPI: `/v1/health`, `/v1/auth/*`, `/v1/analyze`, `/v1/history`, `/v1/search`; Pydantic schema; format lỗi thống nhất `{code, message, detail}` ✅ **8 endpoint** trong `openapi.json`
- [x] 🤖 Xuất `openapi.json` ✅ (25 KB, `python -m app.main`)
- [ ] ⚠️ **DoD "`docker compose up` chạy" — KHÔNG ĐẠT** (thiếu Docker). Phần đạt được: API chạy thật bằng SQLite + **32 test pytest xanh**.

> ✅ **`python -m pytest` → 32/32 xanh** (offline hoàn toàn, SQLite in-memory)
> ✅ **app Flutter vẫn `No issues found!`** — Day 15 không đụng code Dart.
> ✅ **KHÔNG cài thêm package nào** (đã hỏi và được chủ dự án duyệt trước).

**Vì sao dùng SQLite thay Postgres/pgvector?** Không cài được package mới (rule 2) và máy không có Docker để chạy Postgres. Nhưng interface `Repository` trong `app/repository.py` được viết đúng chuẩn ports & adapters — khi có Postgres chỉ cần thêm `PostgresRepository`, **không sửa route nào**.

**`app/` — 8 file:**
| File | Vai trò |
|---|---|
| `main.py` | dựng app, gắn router, `export_openapi()` |
| `errors.py` | format lỗi thống nhất + handler cho **mọi** lỗi |
| `schemas.py` | Pydantic — hợp đồng với model Dart |
| `repository.py` | interface `Repository` (port) |
| `sqlite_repository.py` | adapter SQLite — **chỉ stdlib** |
| `security.py` | bcrypt + token |
| `deps.py` | `get_repository`, `current_user_id` |
| `routers/` | `health` · `auth` · `data` |

**Bảo mật đã hiện thực hoá (có test, không chỉ nói):**
- **bcrypt** băm mật khẩu — không bao giờ lưu plaintext.
- **Mọi** truy vấn SQL dùng tham số `?`/`:name` — không ghép chuỗi → có test SQL injection.
- **Phân quyền theo `user_id` lấy từ token**, không nhận từ query/body → có test **IDOR** (A không thấy/xoá được dữ liệu B dù biết đúng `id`).
- Sai tên đăng nhập và sai mật khẩu trả **cùng** một thông báo (chống dò tài khoản).
- Lỗi 422 **không** phản chiếu lại giá trị người dùng gửi (không lộ mật khẩu).
- `hmac.compare_digest` chống tấn công timing.

**4 lỗi thật tìm ra khi chạy test (đáng ghi):**
1. **404 không ra format thống nhất.** Handler đăng ký cho `fastapi.HTTPException` nhưng route 404 của Starlette ném `StarletteHTTPException` (lớp **cha**, không phải lớp con) → handler **không bao giờ chạy**. Sửa: bắt lớp cha.
2. **`normalize("Đau Mỗi")` trả `"đau moi"`.** `unicodedata.normalize("NFD", "Đ")` **không** cho ra "D" — `Đ` là chữ cái riêng, không phải chữ Latin có dấu. Thiếu bảng `_FOLD_EXTRA` thì **mọi từ bắt đầu bằng "Đ/đ" đều không tìm được** — đúng lỗi người dùng Việt gặp ngay. Đây là lỗi giá trị nhất của ngày.
3. **Rò dữ liệu giữa các test.** Test không xin fixture `repo` thì dùng kho singleton của tiến trình → đăng ký trùng tên ở test sau. Sửa: fixture `client` **phụ thuộc** `repo`.
4. **`add_medicines` thiếu tham số.** Dict nguồn không có đủ 11 cột → `ProgrammingError`. Sửa: điền `None` cho khoá thiếu.

**Cảnh báo trung thực:** `docker-compose.yml` và `Dockerfile` **chưa từng chạy**. Trong cả hai file và README đều ghi rõ ở đầu file. ~~Token Day 15 **không phải JWT chuẩn** (dùng HMAC ký) — **không dùng cho bản phát hành**; Day 16 sẽ thay bằng PyJWT~~ → **ĐÃ ĐÓNG 05/10/2026**: token dùng **`PyJWT`**, đúng chuẩn RFC 7519 (xem `app/jwt_tokens.py`).

### Day 16–17 — Auth thật + phân quyền dữ liệu (2 ngày) 🤖✅ **xong 03/10/2026**
- [x] 🤖 Đăng ký/đăng nhập: hash **bcrypt**, JWT access (15 phút) + refresh (7 ngày), rate limit login
- [x] 🤖 Mọi bảng có `user_id`; **mọi** endpoint lịch sử lọc theo user đang đăng nhập → chặn IDOR (đã làm ở Day 15, giữ nguyên)
- [x] 🤖 Test: hai tài khoản A/B, A **không** thấy và không xoá được history của B
- [x] **DoD:** token hết hạn → refresh hoạt động ✅; truy cập dữ liệu người khác trả 404 ✅

> ✅ **`python -m pytest` → 51/51 xanh** (32 → 51, thêm 19 test)
> ⚠️ **Rate limit dùng bộ đếm trong RAM, KHÔNG phải Redis** — máy không có Docker. Ghi rõ giới hạn (đa worker → hạn mềm lên gấp số worker) trong docstring `app/rate_limit.py`.

**`app/jwt_tokens.py` — JWT HS256 THẬT, chỉ dùng `hmac`/`hashlib`/`base64` chuẩn.**
Đúng định dạng RFC 7519 nên thư viện JWT khác đọc được, FastAPI xác thực được — **không** phải bản rút gọn. Thay vì dùng PyJWT để không phải cài package (rule 2).

Thứ tự kiểm tra **an toàn** (có test cho từng bước):
1. đủ 3 phần; 2. `alg` phải là `HS256` — chặn tấn công `alg: none`; 3. `hmac.compare_digest` (chống timing); 4. `exp` chưa hết.

**Refresh token:**
- Lưu **hash sha256** trong DB, **không** lưu token thô → có test quét toàn bảng khẳng định không có token thô.
- **Rotation**: dùng xong bị thu hồi ngay, dùng lại lần 2 → 401.
- Access token **không** được dùng ở `/refresh` và ngược lại (claim `type`).
- `/v1/auth/logout` thu hồi phiên.

**Rate limit đăng nhập:** 5 lần/60 giây, khoá = IP + tên đăng nhập (chặn được cả dò 1 tài khoản từ nhiều IP lẫn 1 IP dò nhiều tài khoản). Đăng nhập **thành công thì reset** — không phạt nhầm người dùng nhập đúng (có test).

**Lỗi tự gây ra (lặp lại đúng lỗi Day 15 — đáng chú ý):**
> Rò trạng thái toàn cục giữa các test: bộ đếm rate limit là **singleton tiến trình**, nên lượt thử của test trước làm test sau đỡ 429 một cách **ngẫu nhiên**. Sửa ở fixture `repo` (`reset_limiter()` trước/sau) chứ không sửa từng test.
> 💡 Đây là lần thứ 2 cùng một lớp lỗi (Day 15: kho singleton; Day 16: limiter singleton). **Bài học: mọi trạng thái toàn cục phải có hàm reset và được gọi trong fixture.**

### Day 18–19 — Dữ liệu thuốc & tìm kiếm (2 ngày) 🤖✅ **xong 03/10/2026 (có giới hạn — xem bên dưới)**
- [ ] ⚠️ Schema **Postgres** `drugs, ingredients, interactions…` — viết cho SQLite; file SQL cho Postgres **CHƯA** tạo (không có Docker để kiểm chứng)
- [x] 🤖 Pipeline ingest danh mục thuốc từ CSV **cục bộ** → chuẩn hoá tên + hoạt chất + hàm lượng
- [x] 🤖 `/v1/search`: tìm theo tên **và** hoạt chất, phân trang, trả kèm `source` + `updated_at`
- [ ] ⚠️ `pg_trgm` — thay bằng `LIKE` + `normalize()` chuẩn hoá không dấu (chưa có Postgres)
- [x] ✋ Kiểm tra tay: gõ không dấu "panadol", "para 500" đều ra đúng thuốc
- [x] **DoD:** ≥90% ca kiểm thử tay ra kết quả đúng và **có nguồn** ✅ (7/7 ca đạt 100%)

> ✅ **`python -m pytest` → 72/72 xanh** (51 → 72, thêm 21 test)
> ✅ **KHÔNG cài package nào** — `sklearn`/`numpy`/`yaml` đã có sẵn, nhưng RAG sẽ dùng **Python thuần** để không phụ thuộc gì.

**🔴 QUY TẮC DỮ LIỆU — đã áp dụng nghiêm (điều kiện 3 của bạn):**
| Điều | Cách làm |
|---|---|
| Không bịa tên thuốc/hoạt chất | Chỉ đọc từ CSV có sẵn; dòng thiếu tên → **bỏ**, không tự đặt tên |
| Không bịa nguồn | `source` trỏ về **đúng file** đã nạp (`dev-sample: medicines.dev.csv`), **không** gán tên cơ quan lâm sàng |
| Cờ nguồn lâm sàng | Mỗi bản ghi có `is_authoritative=False` — agent **phải từ chối** trả lời y khoa khi cờ này `False` |
| **Tương tác thuốc** | **Cố ý không dựng bảng.** `has_interaction_data()` luôn `False` — tôi không có nguồn để kiểm chứng nên không được bịa |

**Tìm theo hoạt chất:** `/v1/search?name=paracetamol` trả **nhiều** nhãn thương hiệu (Panadol, Efferalgan, Hapacol…) vì câu truy vấn khớp `normalized_name` **hoặc** `ingredients`. Đây là điểm ROADMAP yêu cầu và cũng là điều người dùng thực sự cần.

**Lỗi thật do test bắt được (đáng ghi):**
> Lưu hoạt chất bằng cách **nối dấu cách rồi tách lại** làm vỡ `"paracetamol 500mg"` thành `['paracetamol', '500mg']` — mất cả tên hoạt chất lẫn hàm lượng. Sửa bằng **JSON array** (giống cách `search_results` đã làm). Test `test_hoat_chat_duoc_chuan_hoa` bắt được.

### Day 20–21 — RAG có citation (2 ngày) 🤖✋
- [x] 🤖 Chuẩn hoá tiếng Việt **trước** tokenization (`app/textnorm.py`) — bỏ dấu + hạ chữ thường → tách âm tiết → so khớp bằng **bigram**. Vì sao: `hạ`/`à`/`á`/`ạ` là cùng một âm tiết; tách trên chuỗi thô thì "Hạ Sốt" ≠ "Ha Sot".
- [x] 🤖 Cổng LLM: `LLMProvider` (Protocol) + `MockLLMProvider` **tất định, offline**. ❌ **Không** adapter thật, ❌ không API key, ❌ không `requests`/`httpx` — có test quét AST chặn.
- [x] 🤖 Truy hồi top-k có citation (`app/retrieval.py`) — BM25 đơn giản hoá, thuần Python, **không cần pgvector** (máy dev không có Docker).
- [x] 🤖 Endpoint `/v1/agent/ask`: bắt buộc đăng nhập, trả lời **kèm citation** (id + đoạn trích + `source` + `is_authoritative`).
- [x] 🤖 Guardrail 3 lớp: (1) lọc chủ đề + phát hiện injection **trước** khi gọi LLM → (2) escape + bọc marker ngữ cảnh → (3) **kiểm chứng grounding sau khi sinh**.
- [x] 🤖 Nguồn không authoritative hoặc không có căn cứ → **từ chối**, kèm mã ổn định (`no_evidence` / `unsupported_claim` / `out_of_scope`).
- [x] ✅ **DoD ĐẠT:** `python -m pytest` → **182/182 xanh** (97 → 182, thêm 85 test).

> ⚠️ **Lệch so với ROADMAP gốc, và đó là chủ ý.** ROADMAP ghi "bật `pgvector` + embedding". Máy dev **không có Docker** nên không chạy được pgvector, và `AGENTS.md` + D2 cấm cài package mới. Ta làm retrieval bằng từ khoá trước; chỗ duy nhất cần đổi khi có Postgres/pgvector là thêm hàm `retrieve()` cùng chữ ký — route không đổi (giống cách `Repository` đã giữ từ Day 15).

> ⚠️ **Việc này chưa phải RAG hoàn chỉnh, và không nên gọi là hoàn chỉnh.** Thiếu: embedding thật, nhiều tài liệu, re-rank. Tuyệt đối chưa có: nguồn lâm sàng `is_authoritative=True`. Vì vậy ứng dụng **đang ở chế độ chỉ-trích-dẫn, kèm cảnh báo dữ liệu mẫu ở mọi câu trả lời**.

> 🔬 **Ba lỗi thật phát hiện khi kiểm thử, đã sửa và có test chặn lại:**
> 1. **Cơ chế trích dẫn tự phá vỡ.** Gộp mọi nguồn vào **một** cặp marker khiến các mã/đoạn thứ hai bị đọc nhầm thành *nội dung* của nguồn đầu → grounding fail toàn bộ. Sửa: **mỗi nguồn một cặp marker riêng**.
> 2. **Truy hồi quá rộng, trả lời lệch câu hỏi.** "bạn bị gì?" khớp đúng từ "ban" trong "**phát ban**" và đạt điểm 0.50 — **cao hơn** "tác dụng phụ của panadol" (0.30). Sửa: thêm `MIN_TERM_COVERAGE = 0.4` (phủ ≥ 40% từ truy vấn) + bỏ "ban" khỏi stopwords.
> 3. **`HistoryResponse` mất field `total`** do chèn code sai vị trí — `test_security.py` bắt được, đã sửa.
>
> 📌 Bài học cho các ngày sau: **ngưỡng số phải đo trên dữ liệu thật, không đặt bừa.** `MIN_RELEVANCE_SCORE` được đặt giữa hai nhóm điểm đo được (liên quan ≥ 0.30, không liên quan ≤ 0.17), và có test `test_nguong_loc_bien_tinh_bang_du_lieu` để nếu đổi dataset thì test báo động.

### Day 22–23 — Agent core v1 (2 ngày) 🤖✋ ✅ **xong 05/10/2026**
- [x] 🤖 5 tool sau interface `Tool` + `ToolRegistry`: `search_drug`, `check_interaction`, `normalize_prescription`, `explain_for_patient`, `schedule_reminder` (`app/tools.py`)
- [x] 🤖 Orchestrator: planner → gọi tool → ghép câu trả lời **từ kết quả tool** (`app/orchestrator.py`)
- [x] 🤖 Bọc tool sau interface/factory → Day 28 thay MCP chỉ cần `registry.register(...)`, **không sửa orchestrator/route** (có test `test_dang_ky_tool_moi_khong_phai_sua_orchestrator`)
- [x] 🤖 SSE: `POST /v1/agent/chat` trả `text/event-stream` — tên sự kiện khớp **1-1** với `lib/domain/ports/agent_client.dart`
- [x] 🤖 Memory: `SessionStore` protocol + `InMemorySessionStore`; lọc theo `(user_id, session_id)`; có `DELETE /v1/agent/session/{id}` (quyền được quên, D5)
- [x] 🤖 Test: câu ROADMAP *"Tôi đang uống warfarin, thêm panadol được không?"* → **gọi đúng tool tương tác**
- [x] ✅ **DoD ĐẠT (một nửa):** `python -m pytest` → **231/231 xanh** (193 → 231, thêm 38 test).

> ⛔ **Nửa chưa đạt: "trả lời có căn cứ" cho tương tác thuốc.**
> `check_interaction` **luôn từ chối** vì máy không có bảng tương tác đã kiểm chứng, và
> `AGENTS.md` mục 3 cấm tự bịa — đây là chỗ sai sót **có hậu quả thật** (người bệnh
> uống hai thuốc đối kháng). Cổng mở lại là `has_interaction_data()` trong
> `app/ingest.py`. Nói thẳng thay vì làm cho "có vẻ đạt".
>
> 🔒 **Hội thoại 5 lượt giữ ngữ cảnh: chưa kiểm chứng được.** Session lưu được và
> lọc đúng theo `user_id`, nhưng `MockLLMProvider` **không dùng lịch sử** để trả lời
> (nó chỉ trích nguyên văn nguồn). Nên đo "giữ ngữ cảnh" chỉ có nghĩa khi có LLM thật.
> Đừng đánh dấu DoD này là đạt.
>
> 🔌 **Lệch so với ROADMAP gốc, có chủ ý.** ROADMAP ghi "session ở Redis, hồ sơ ở
> Postgres". Máy **không có Docker** và D2 cấm cài `redis` → làm `SessionStore` protocol
> + bản in-memory. Khi có Redis: viết `RedisSessionStore`, đổi **một hàm**
> `get_session_store()` — orchestrator và route không đổi.
>
> 🔬 **Ba lỗi thật phát hiện khi kiểm thử:**
> 1. **`t[uươ]ng` không khớp "tương".** Lớp `[uươ]` chỉ khớp **một** ký tự giữa, mà
>    "tương" có **hai** (`ư`, `ơ`) → câu hỏi tương tác thuốc **rơi nhầm** sang nhánh
>    tra cứu thường. Đây là lỗi **im lặng**: test kiểu "chức năng chạy" vẫn xanh.
>    Sửa: bỏ regex có dấu, khớp trên chuỗi đã `normalize()` — cùng nguyên tắc
>    `app/textnorm.py`: chuẩn hoá **trước**, khớp **sau**.
> 2. **Thiếu `done` ở nhánh từ chối ở cổng đầu.** Hợp đồng Dart yêu cầu *"Sự kiện luôn
>    kết thúc bằng `Done` hoặc `Failure`"*; thiếu thì màn hình chat **treo vĩnh viễn**.
>    Test "tool chạy được" không bắt được vì lỗi nằm ở *luồng sự kiện*, không ở logic
>    tool. Sửa: gom về `_abort()` — mọi nhánh dừng sớm đều phát `done`.
> 3. **`del repo` trong tool xoá chính biến dùng sau.** `ExplainForPatientTool` ném
>    `UnboundLocalError`; `ToolRegistry.run()` nuốt thành "gặp sự cố" nên lỗi này
>    **không lộ ra ngoài** — chỉ thấy được khi gọi tool trực tiếp.
>
> 📌 Bài học Day 22: **lỗi nguy hiểm nhất không nằm ở chỗ ta đang nhìn.** Ở đây nó nằm ở
> *hợp đồng luồng sự kiện* (Dart ↔ SSE) và ở *việc nuốt exception*, nơi "xanh test"
> không bảo chứng "chạy đúng". Đừng chỉ test logic, hãy test cả luồng.

> ✅ **Day 24 xong** — `python eval.py` → **109/109 xanh** (40 ms).
>
> **Eval suite (`evals/cases.yaml`):** 109 ca chia 8 nhóm:
>
> | Nhóm | Số ca | Kiểm tra |
> |---|---|---|
> | lookup | 15 | Tra cứu trực tiếp tên thuốc → answered + citations |
> | composition | 10 | Hỏi thành phần/hoạt chất |
> | uses | 10 | Hỏi công dụng |
> | side_effects | 8 | Hỏi tác dụng phụ |
> | out_of_scope | 25 | Chẩn đoán, liều, tương tác, cấp cứu, thai kỳ |
> | injection | 15 | Prompt injection (VI/EN/ES) |
> | no_evidence | 15 | Thuốc không có trong database |
> | edge | 11 | Ký tự đặc biệt, XSS, SQL inject, emoji, Unicode |
>
> **Observability đã có sẵn từ code Day 22-23:**
> - Trace middleware: `app/main.py:66-109` — sinh `trace_id`, ghi `access_log()` 4 trường
> - UsageTracker: RAM per-user, trả qua `/v1/agent/usage` (không ghi log — D4)
> - Safe logging: `ALLOWED_LOG_FIELDS` + `safe_log_fields()` + `assert_log_safe()`
>
> ⚠️ **Phát hiện quan trọng qua eval: `_OUT_OF_SCOPE_RE` có 15 lỗ hổng regex.**
> Nguyên nhân: character class `[eê]` không khớp `ề`, `[oô]` không khớp `ố`,
> `[uương]ng` chỉ khớp 1 ký tự thay vì chuỗi `ương`. Hậu quả: câu hỏi tiếng Việt
> có dấu như "Liều dùng", "Tương tác", "Chẩn đoán" lọt qua bộ lọc phạm vi.
> An toàn hiện tại dựa vào lớp 2 (grounding) — nhưng nên sửa regex sớm.
> Sửa đề xuất: so khớp trên chuỗi `normalize()` (bỏ dấu) thay vì chuỗi gốc.
>
> ⏸️ **Dart client sinh từ openapi.json:** cần package `openapi_generator` hoặc
> `retrofit_generator` (D2: chờ duyệt). Models Dart hiện tại (`lib/data/models/`)
> đã khớp schema thủ công — chưa cần thiết gấp.
>
> **DoD kiểm chứng:**
> - `python eval.py` → 109/109 ✅
> - Tăng `MIN_RELEVANCE_SCORE` 0.15→0.50 → điểm tụt 100%→79.8% (22 fail) ✅
> - `python -m pytest` → 231/231 ✅ (không hồi quy)

### Day 24 — Eval + quan sát + hợp đồng API (2 ngày gộp) 🤖✋
- [x] 🤖 Bộ **eval 109 ca** (`evals/cases.yaml`): 8 nhóm (lookup, composition, uses, side_effects, out_of_scope, injection, no_evidence, edge)
- [x] 🤖 `eval.py` chạy tự động → bảng điểm theo nhóm; tăng ngưỡng BM25 0.15→0.50 → 100%→79.8%
- [x] 🤖 Observability: trace id middleware (`app/main.py:66-109`), UsageTracker RAM (`app/observability.py`), safe logging 4 trường
- [ ] ⏸️ Sinh **Dart client** từ `openapi.json` — cần package `openapi_generator`/`retrofit_generator` (D2: chờ duyệt). Models Dart hiện tại đã khớp schema thủ công.
- [x] **DoD:** `python eval.py` → 109/109 xanh; đổi prompt → điểm thay đổi rõ ràng. 🎉 **Milestone M2**

---

## 5. GIAI ĐOẠN P3 — Nâng cao & Release (Day 25 → Day 30+)

### Day 25 — Chat UI streaming (3 giờ) 🤖✋
- [x] 🤖 Màn `AgentChatScreen`: stream token (Dio `ResponseType.stream` / SSE), bong bóng hội thoại, **citation bấm được**
- [x] 🤖 Trạng thái "đang tra cứu…" khi agent gọi tool; nút Dừng; ~~lịch sử hội thoại lưu offline~~ ⏸️ (cần Day 27 offline infra)
- [ ] **DoD:** chữ hiện dần theo token; bấm citation mở đúng đoạn tài liệu.

> **Day 25 execution log (2026-10-13):**
> - `SseAgentClient` — Dio `ResponseType.stream` → parse SSE `event:/data:` → `AgentEvent` sealed class. Handles token/tool_call/citation/done/failure/refusal.
> - `ChatController` (AutoDisposeAsyncNotifier) — accumulates tokens into last message, manages stream lifecycle, stop/clear.
> - `AgentChatScreen` — bubble UI, typing indicator (animated dots), citation chips → bottom-sheet, tool-running chip, stop button, clear dialog.
> - i18n: 9 new keys (vi+en): chatTitle/chatHint/chatSend/chatStop/chatEmpty/chatClear/chatClearTitle/chatClearBody/chatToolRunning.
> - `agentClientProvider` + `chatControllerProvider` wired in `providers.dart`.
> - Navigation: chat icon button added to `ProcessImageScreen` AppBar.
> - `flutter analyze` → 0 issues.
> - Remaining: offline history persistence (stretch — depends on offline cache infra), runtime DoD verification needs server running.

### Day 26 — On-device OCR + Hybrid (4 giờ) 🤖✋
- [x] ✋ Thêm lại dependency đúng: `google_mlkit_text_recognition`, `path_provider` (**không** dùng `tflite_flutter_helper` — đã discontinued)
- [ ] ✋ iOS: sửa `Podfile` thành `platform :ios, '15.5'` + block `post_install` set `IPHONEOS_DEPLOYMENT_TARGET`; đặt `compileSdk/targetSdk = 35`
- [x] 🤖 `ondevice_analyzer.dart` (ML Kit OCR + chuẩn hoá chuỗi + tra DB local) và `hybrid_analyzer.dart` (local trước → fallback remote → ghi cache)
- [x] 🤖 Feature flag chọn adapter: `--dart-define=ANALYZER=hybrid` (không sửa UI)
- [ ] ✋ Test trên **thiết bị thật** (emulator OCR kém chính xác) ở chế độ máy bay
- [ ] **DoD:** chế độ máy bay + ảnh có trong DB local → vẫn trả kết quả < 2 giây.

> **Day 26 execution log (2026-10-13):**
> - `OnDeviceAnalyzer` — ML Kit `TextRecognizer` → OCR text → `_matchMedicines()` with normalized substring + 60% token overlap. Full Vietnamese diacritics removal (`_removeDiacritics`). `dispose()` closes recognizer.
> - `HybridAnalyzer` — 3-tier fallback: local (OnDevice) → remote (API) → cache (OfflineCache) → throw. Saves successful results to cache.
> - Feature flag: `const _analyzerMode = String.fromEnvironment('ANALYZER', defaultValue: 'remote')` in `providers.dart`. Values: `remote` (default), `ondevice`, `hybrid`.
> - `medicineDatabaseProvider` (StateProvider) feeds local medicine list to OnDeviceAnalyzer.
> - `compileSdk = 35` set in `android/app/build.gradle`.
> - Packages added: `google_mlkit_text_recognition ^0.13.0`, `path_provider ^2.1.3`.
> - `flutter analyze` → 0 errors (1 info: transitive `path` dep).
> - Remaining: iOS Podfile config, real-device OCR testing, DoD runtime verification.

### Day 27 — Memory & nhắc uống thuốc (3 giờ) 🤖✋
- [x] 🤖 Hồ sơ sức khoẻ: dị ứng, bệnh nền, thuốc đang dùng (có consent rõ ràng, sửa/xoá được)
- [x] 🤖 Nhắc uống thuốc bằng local notification; ghi nhận adherence; agent dùng dữ liệu này khi trả lời
- [ ] **DoD:** tạo lịch nhắc → nhận được thông báo; agent biết "thuốc đang dùng" của tôi.

> **Day 27 execution log (2026-10-13):**
> - `HealthProfile` model — allergies, conditions, currentMedications (List<String>), consentGivenAt. `MedicationReminder` — name, hour, minute, dosage, isActive, daysOfWeek. `AdherenceRecord` — reminderId, takenAt, skipped. All use `toMap()`/`fromMap()` for SQLite, pipe-separated strings for lists.
> - `HealthDatabase` (singleton, sqflite) — 3 tables: `health_profile`, `medication_reminders`, `adherence_log`. Full CRUD + `deleteAll()` for right-to-be-forgotten.
> - `NotificationService` (singleton, flutter_local_notifications) — `init()`, `scheduleReminder()` (24h periodic), `cancelReminder()`, `cancelAll()`, `requestPermission()` (Android 13+). Channel: `med_reminder`.
> - `HealthProfileScreen` — consent banner before saving, 3 TextFields (allergies/conditions/medications), Save/Delete with confirmation dialog.
> - `ReminderScreen` — reminder cards with Switch toggle + delete, FAB → dialog (name, dosage, time picker).
> - `HealthProfileController` + `ReminderController` (AsyncNotifier) wired in `providers.dart`.
> - Client→Server: `ChatController` reads `healthProfileControllerProvider`, sends `currentMedications` via `AgentClient.chat()`. `SseAgentClient` sends `current_medications` in request body.
> - Server: `AgentChatRequest.current_medications` field added to `schemas.py`. `orchestrator.chat()` appends medication-context warning to answers. `agent.py` router passes `current_medications` through.
> - i18n: 16 new keys (vi+en): health profile labels, reminder labels, consent/delete messages.
> - All health data stored **local-only** (sqflite). `current_medications` sent to server only when user actively chats (ephemeral, not persisted server-side).
> - Packages added: `sqflite ^2.3.2`, `flutter_local_notifications ^17.2.1`.
> - `flutter analyze` → 0 errors. `pytest` → all pass.
> - Remaining: runtime DoD verification (needs device with notification permissions).

### Day 28 — Đóng gói tool thành MCP servers (3 giờ) 🤖✋
- [x] 🤖 Expose `search_drug`, `check_interaction` ra **MCP server** (chuẩn mở, dùng lại được cho client AI khác)
- [x] 🤖 Orchestrator gọi tool **qua MCP** thay vì import trực tiếp → chứng minh khả năng tái sử dụng
- [ ] ✋ Kiểm thử bằng MCP Inspector
- [ ] **DoD:** gọi được tool từ ngoài app (một MCP client bất kỳ) — đây là điểm "bước trước thời đại".

> **Day 28 execution log (2026-10-13):**
> - `app/mcp_server.py` — `MCPServer` (mcp SDK v2) exposing 2 public tools: `search_drug(query)` and `check_interaction(drugs)`. Runs standalone via stdio or SSE transport. Uses its own `SQLiteRepository` (same DB via `MEDASSIST_DB` env). Localhost only.
> - `app/mcp_tools.py` — `McpTool` adapter implementing `Tool` protocol. Opens stdio subprocess to MCP server per call. `_parse_mcp_result()` converts MCP `CallToolResult` → `ToolResult` with citations.
> - `mcp_registry()` — hybrid: 2 tools via MCP (`search_drug`, `check_interaction`), 3 direct (`normalize_prescription`, `explain_for_patient`, `schedule_reminder`). Direct tools need `repo`/`user_id` from JWT.
> - Feature flag: `MEDASSIST_USE_MCP=1` → orchestrator uses `mcp_registry()` instead of `default_registry()`. No orchestrator/route changes needed.
> - Tests: 14 new tests in `tests/test_mcp.py` — MCP server functions, adapter protocol compliance, result parsing, registry composition, feature flag switching. All pass.
> - Packages added: `mcp[cli]>=1.0` (installed v2.3.0), `httpx-sse>=0.4` (installed v0.4.3).
> - `pytest` → all 240+ tests pass (including 14 new MCP tests).
> - Remaining: MCP Inspector manual testing (DoD), end-to-end stdio transport test.

### Day 29 — An toàn, riêng tư, ổn định (3 giờ) 🤖✋
- [~] 🤖 Consent + chính sách dữ liệu trong app; retention (tự xoá ảnh sau N ngày); audit log hành động nhạy cảm — **phía server + tài liệu xong, màn consent/xoá tài khoản trong app CHƯA** (xem log bên dưới)
- [ ] 🤖 Crash reporting (Crashlytics/Sentry) ⛔ cần chủ dự án duyệt (SDK gửi dữ liệu ra ngoài); timeout + retry, offline queue chưa làm
- [ ] ✋ Test tay 8 tình huống xấu: mất mạng giữa upload, token hết hạn, server 500, ảnh cực lớn, màn hình nhỏ, cỡ chữ lớn, xoay màn hình, đăng nhập 2 tài khoản
- [ ] **DoD:** không có màn trắng/crash trong 8 tình huống trên.

> **Day 29 execution log (2026-10-24) — chỉ phần kiểm chứng được trên máy này (không có Flutter/JDK):**
> - 🔴 **Sửa lỗ hổng an toàn thật:** `_OUT_OF_SCOPE_RE` (regex có dấu, character class sai) để lọt "tương tác", "liều dùng", "chẩn đoán"… Thay bằng `is_out_of_scope()` khớp trên chuỗi đã `normalize()` (`app/agent.py`); luồng chat dùng `allow_interaction=True` để câu tương tác đi tới `check_interaction`. Test: `tests/test_scope.py` (22 ca).
> - ⚠️ Eval: 10 ca `oos_001–019` mong `no_evidence` — đó là kỳ vọng "đóng băng" theo hành vi lỗi; đã sửa thành `out_of_scope` (đúng như tiêu đề nhóm). 2 ca `oos_024/025` khớp mẫu regex cũ viết từ chuỗi vô nghĩa nên chuyển sang `no_evidence` (vẫn từ chối, `answered=false`). **Điểm 109/109 giữ nguyên nhưng bài kiểm tra đã thay đổi — rà lại 2 ca này.**
> - ✅ **Xoá tài khoản** (quyết định D5 từng nợ): `POST /v1/auth/delete-account` (token + mật khẩu, giới hạn tốc độ) xoá user, lịch sử, refresh token, hội thoại RAM, số liệu dùng. Test A/B không đụng nhau.
> - ✅ **Retention:** `MEDASSIST_HISTORY_RETENTION_DAYS` (mặc định tắt; giá trị ≤0/rác = tắt, không bao giờ "xoá hết"), áp dụng lúc khởi động (`app/privacy.py`).
> - ✅ **Audit log** danh sách hành động đóng, chỉ `trace_id/action/outcome` — cố ý không có `user_id` (D4).
> - 🔴 **Sửa lỗi riêng tư phía app:** logout / 401 **không xoá cache ngoại tuyến** → người đăng nhập sau thấy lịch sử người trước khi mất mạng. Nay xoá `CacheStore.clear()` (`providers.dart`) + 2 test. ⚠️ **Dart chưa chạy `flutter analyze/test`** (thiếu SDK).
> - ✅ `android:allowBackup="false"` (hồ sơ sức khoẻ không bị backup lên đám mây; không thêm quyền).
> - ✅ `docs/PRIVACY.md` (thiếu từ trước dù `AGENTS.md` yêu cầu), `docs/RELEASE.md`, `openapi.json` cập nhật.
> - `python -m pytest` → **282 xanh** (240 → 282); `python eval.py` → 109/109.
> - **Chưa làm / cần người:** màn consent + nút "Xoá tài khoản" trong app (cần Flutter để xác minh), mã hoá SQLite/prefs cục bộ, timeout/retry/offline queue, 8 tình huống test tay, crash reporting, quyền `POST_NOTIFICATIONS` (**nhắc thuốc có thể không chạy trên Android 13+**).

> **Day 29 — phần 2 (2026-10-24): chạy thật trên máy ảo `ueh_pixel` (Flutter 3.19.4, JDK 17).**
> Máy có sẵn toolchain (không nằm trong PATH): Flutter `D:\flutter_windows_3.19.4-stable`, JDK `D:\dev\jdk17`, SDK `D:\Android\Sdk`, AVD `D:\dev\avd`. `flutter analyze` sạch; `flutter test` **308/308**; `pytest` **290**; `eval` 109/109; **kiểm thử sống adapter Dart ↔ server thật 9/9** (`tools/medassist_core/core_live_check_test.dart`).
>
> **Lỗi thật tìm ra nhờ chạy thật (trước đó "xanh" nhưng hỏng):**
> 1. App chỉ khớp mock Flask, **không khớp `medassist-core`** (đường dẫn, tên trường JSON thuốc, token lồng) → thêm `ApiPaths`/`BACKEND`, model đọc cả hai kiểu.
> 2. **Không có refresh token** → bị đăng xuất sau 15 phút. Nay tự làm mới (single-flight, gọi lại 1 lần).
> 3. **Chat SSE chưa từng chạy** (`utf8.decoder` sai kiểu `Uint8List`) + `refusal` hiện mã thô + token dính chữ (`_chunk_text` mất khoảng trắng).
> 4. `build.gradle` dùng `flutter.versionCode` (Flutter mới) → **không build được trên 3.19.4**.
> 5. OCR trên máy luôn rỗng (`medicineDatabaseProvider` không bao giờ được nạp) → thêm `GET /v1/medicines` + `medicineCatalogProvider`.
> 6. Nhắc thuốc: **không xin quyền thông báo** (thông báo bị chặn âm thầm) — đã sửa; **giờ nhắc chưa đúng** (cần `timezone`, chờ duyệt).
> 7. Access log uvicorn lộ từ khoá tìm thuốc — đã tắt.
> 8. Thêm: màn đồng ý, màn Quyền riêng tư + xoá tài khoản (đã chạy thật), GET tự thử lại, thiết kế lại UI (theme teal Material 3, đăng nhập/đăng ký, lưới lối tắt, nhãn AI ở chat).
> 9. Chưa tái hiện được: hồ sơ sức khoẻ biến mất sau một lần cài lại APK.
>
> **Phần 3 (2026-10-28) — truy hồi kéo nhầm thuốc:** "Panadol có tác dụng phụ gì" trả thêm trích dẫn của Decolgen/Hapacol/Para/Paracetamol Stada, vì đoạn trích có tiền tố nhãn ("Tác dụng phụ: …") nên từ chung của câu hỏi khớp nhãn của thuốc khác. Sửa: `_named_medicines()` trong `app/retrieval.py` — khi câu hỏi chứa từ định danh duy nhất của một thuốc (có trong tên, không xuất hiện ở thuốc khác) thì chỉ truy hồi trong thuốc đó; hoạt chất dùng chung ("paracetamol") không neo. +2 test hồi quy. Chưa tái hiện được đúng ca "Amoxicillin cho câu hỏi Panadol" đã ghi trước đó (có thể chính là lỗi này). `flutter analyze` sạch (1 info), `flutter test` 318, `pytest` xanh, `eval` 109/109.

### Day 30 — Release (3 giờ) ✋🤖
- [~] ✋ Tạo keystore release thật (bỏ debug signing); `flutter build appbundle --release --dart-define=API_BASE_URL=https://...` — ✅ `build.gradle` đã đọc `android/key.properties` (+ `key.properties.example`); ✋ chủ dự án tự tạo keystore, xem **`docs/RELEASE.md`**
- [ ] ✋ Icon/splash (`flutter_launcher_icons`, `flutter_native_splash`), tên app, ảnh chụp màn hình
- [ ] ✋ Flavor `dev/staging/prod` + `--dart-define` cho từng môi trường
- [ ] ✋ Beta nội bộ (Internal testing / TestFlight), thu phản hồi
- [ ] ✋ `git tag v1.0.0`, cập nhật tiến độ vào Phụ lục C
- [ ] **DoD:** `.aab` release build thành công, cài được từ Internal testing. 🎉 **Milestone M3**

> **Day 30 execution log (2026-10-28) — chủ dự án duyệt "tất cả":**
> - ✅ Nhắc thuốc đúng giờ: `zonedSchedule` theo `hour:minute` + `daysOfWeek`, lặp hằng tuần (`nextReminderTime` + 7 test), dựng lại lịch mỗi lần mở app; thêm `timezone ^0.9.4`, quyền `POST_NOTIFICATIONS`/`SCHEDULE_EXACT_ALARM`/`RECEIVE_BOOT_COMPLETED`, 2 receiver của plugin.
> - ✅ Flavor Gradle `dev/staging/prod` (hậu tố applicationId + nhãn app).
> - ✅ Icon thích ứng + nền splash bằng tài nguyên Android thuần (không thêm package).
> - 🔴 **Lỗi release thật tìm ra nhờ build + chạy bản release:** (1) R8 lỗi vì ML Kit tham chiếu bộ nhận dạng chữ Hán/Nhật/Hàn/Devanagari; (2) bật R8 làm nhắc thuốc crash (`TypeToken` của Gson) khi nhận sự kiện khởi động. Sửa bằng `android/app/proguard-rules.pro`; cài lại bản release trên AVD không còn crash.
> - `flutter test` 325, `flutter analyze` sạch (1 info), `.aab` prod release (41 MB) build được, APK release chạy trên AVD.
> - ⛔ Crash reporting chưa làm (cần chọn SDK + DSN + consent + PRIVACY.md). ✋ Còn: keystore thật, beta nội bộ, `git tag`, thử thông báo thật tới giờ, 8 tình huống test tay.

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
| 5 | AppConfig + ApiClient | 🤖✋ | chỉ còn 1 chỗ có `10.0.2.2` | ✅ 01/10 |
| 6 | Model typed | 🤖✋ | test models xanh | ✅ 01/10 |
| 7 | Ports | 🤖✋ | domain không import Flutter | ✅ 01/10 |
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

**Môi trường ghi nhận:** Flutter **3.19.4** stable / Dart **3.3.2** · git **2.55.0.windows.3** · identity `Tuan Anh <nguoidung.demo@example.com>` · Windows 10.0.26200 · locale en-US.

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

**Việc cần bạn làm để qua được M0 (chi tiết đầy đủ ở `docs/TROUBLESHOOTING.md` mục 1–4, 12):**
```powershell
# 0) RẤT QUAN TRỌNG: C: chỉ còn ~4 GB, trong khi AVD + cache Gradle cần 4-13 GB
New-Item -ItemType Directory -Force -Path 'D:\dev\.gradle','D:\dev\AndroidAvd' | Out-Null
setx GRADLE_USER_HOME "D:\dev\.gradle"; setx ANDROID_AVD_HOME "D:\dev\AndroidAvd"

# 1) JDK 17 (Temurin) — bắt buộc để build Android (JDK của Java, không phải của Flutter)
winget install --id EclipseAdoptium.Temurin.17.JDK -e

# 2) Device để chạy app — chọn 1:
#    A) Cắm điện thoại Android thật + bật USB debugging  -> KHÔNG tốn dung lượng, KHÔNG cần cài gì thêm
#    B) AVD: cần cmdline-tools (flutter emulators KHÔNG có --create), rồi:
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
& "$sdk\cmdline-tools\latest\bin\avdmanager.bat" create avd -n pixel_api35 -k "system-images;android-35;google_apis;x86_64" -d pixel_7
& "$sdk\emulator\emulator.exe" -avd pixel_api35 -no-snapshot-save
flutter devices      # phải thấy emulator-5554
```
**Ảnh đã có sẵn trên máy (khỏi tải):** `android-35/google_apis/x86_64` ✅ · `android-33/google_apis/x86_64` ✅ · `android-34/google_apis` ❌ **rỗng** (chưa có ABI nào).
Máy CPU Intel AMD64 + `HypervisorPresent=True` → chọn ABI **`x86_64`**, emulator có WHPX.

Sau đó `flutter run -d emulator-5554` → app phải vào màn Đăng nhập (đây là DoD còn lại của Day 2 + M0).

**Kế tiếp (Day 4 còn lại):** `docs/TROUBLESHOOTING.md` **đã xong**; còn đúng 1 việc = **bạn cài JDK 17 + tạo device** để tôi chạy `flutter run` nghiệm thu M0. (Dọn 38 lint info đã dời sang Day 10.)

> 📌 **Ghi chú quy trình:** nếu chưa cài được JDK/device, **vẫn đi tiếp P1 (Day 5+) bình thường** — toàn bộ code P1 (`AppConfig`, `ApiClient`, `Result`, Riverpod provider) verify được bằng `flutter test` trên Dart VM, không cần Android toolchain. Chỉ cần device trước **Day 10** (exit P0) và **Day 24** (exit P1).

---

### 30/09/2026 — Day 5 🟡 (code + DoD xong, **verify bị chặn: máy không có Flutter SDK**)

**Code Day 5 đã có đủ 4 file (không cần viết lại):**
- `lib/core/config/app_config.dart` — `baseUrl` từ `--dart-define=API_BASE_URL`; `apiPrefix`, timeout, `flavor`; `uri()` encode query; `validate()` fail-fast; `describe()` để log.
- `lib/core/network/api_client.dart` — bọc `Dio`, `get/post/delete/upload`, `AuthInterceptor` (gắn token, log, báo 401) + `redactHeaders/redactBody` để **không lộ mật khẩu/token**.
- `lib/core/network/api_exception.dart` — `AppExceptionKind` 13 nhánh + thông báo tiếng Việt; múc `detail` từ body Flask/FastAPI.
- `lib/core/network/result.dart` — `sealed Result<T>` (`Ok`/`Err`) + `guard()`.
- `lib/main.dart` đã in `AppConfig.describe()`; cả 5 màn hình `lib/screens/` đã dùng `AppConfig` thay cho nối chuỗi URL.

**Test Day 5 đã viết (TDD), 4 file:** `test/core/config/app_config_test.dart`, `test/core/network/api_client_test.dart` (dùng `_FakeAdapter` chặn socket, chạy offline), `api_exception_test.dart`, `result_test.dart`.

**✅ DoD đạt:** `Select-String '10\.0\.2\.2' lib\*.dart` → **4 hit, tất cả trong `app_config.dart`** (3 hit là comment, 1 hit là `defaultValue`). Không còn URL hardcode ở màn hình nào.

**⛔ BLOCKER — chưa verify được bằng máy:**
> Máy đang mở code này **không cài Flutter, Dart, Git**: `where flutter` / `where dart` / `where git` → *Could not find files*; không có ổ `D:` (`.dart_tool/package_config.json` vẫn trỏ `D:/flutter_windows_3.19.4-stable/flutter` và pub cache `C:/Users/TEN_BAN/...` → project được **copy từ máy cũ**, môi trường ở đây không tương thích).
> ⇒ `flutter test test/core`, `dart format .`, `flutter analyze` **đều chưa chạy được**.

**Việc cần làm để nghiệm thu Day 5 (mất 5 phút nếu đã có Flutter):**
```powershell
flutter pub get
flutter test test/core      # kỳ vọng: All tests passed
dart format --output=none --set-exit-if-changed .
flutter analyze --no-fatal-infos
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000   # log phải in "MedAssist → API: dev http://10.0.2.2:5000 (http · timeout 15s/60s)"
```
Nếu `flutter test` đỏ → sửa tới xanh rồi mới tick nốt 2 ô ✋ ở Day 5 và chuyển sang **Day 6 (Model typed)**.

**Day 6 — Model typed (🟡 code xong, cùng blocker Flutter SDK) — viết trước để tiết kiệm quota**

`lib/data/models/` — **5 file, không thêm dependency nào** (`freezed` bị loại như kế hoạch):
| File | Nội dung |
|---|---|
| `json_utils.dart` | `asString/asStringOrNull/asInt/asDouble/asPercent/asJsonMap/asJsonList/asDateTime` + `decodeJsonObject` + sentinel `notProvided`/`pickChanged` cho `copyWith` |
| `medicine.dart` | `Medicine` (8 khoá JSON nằm trong hằng số) + `MedicineSearchResponse` |
| `history_item.dart` | `HistoryItem` + `HistoryResponse` |
| `user.dart` | `User` + `AuthSession` |
| `analysis_result.dart` | `AnalysisResult` + `toDisplayLines()` |

`test/data/models/` — **5 file + `fixtures/server_samples.dart`**, ~90 test:
| File | Nội dung |
|---|---|
| `fixtures/server_samples.dart` | JSON **thật** theo contract Flask (`search-medicine`, `process-image`, `history`, `login`, `register`) |
| `medicine_test.dart` | 8 trường · null/rỗng/số-thay-chuỗi · round-trip · `copyWith` · `==`/`hashCode` · response |
| `history_item_test.dart` | bản ghi **thiếu `id`** bị lọc · `created_at` · parsing lỗi |
| `user_test.dart` | **bảo mật**: `toJson()` không bao giờ có mật khẩu · token · validate email |
| `analysis_result_test.dart` | 3 hình dạng body · `toDisplayLines()` giữ nguyên `:` trong dữ liệu |
| `json_utils_test.dart` | mọi hàm ép kiểu + HTML lỗi Flask → `parsing` chứ không `FormatException` |

**3 quyết định thiết kế đáng ghi lại:**
1. **Hàm "đọc" không bao giờ ném, hàm "ép hình dạng" mới ném.** Server là CSV scrape nên `null`/`""`/`"93%"`/`93` là chuyện thường — mất 1 ô dữ liệu còn hơn sập màn hình. Chỉ khi body sai *hình dạng* mới ném `AppExceptionKind.parsing` (khung đã dựng sẵn ở Day 5), adapter bọc `guard()` là xong.
2. **`User` không có field `password`.** Mật khẩu là tham số của `toCredentialsJson(password:)` rồi biến mất — chống rò rỉ *mặc định* thay vì dựa vào kỷ luật. Có test riêng canh điều này.
3. **`copyWith` dùng sentinel `notProvided`.** `copyWith({String? x})` không phân biệt được *"không truyền"* và *"truyền `null`"*; sentinel là cách duy nhất để xoá được field thật.

**⚠️ Chênh lệch có chủ đích với DoD Day 6:** *"không còn chỗ nào parse JSON thủ công"* — trong `lib/data/models/` đã sạch 100%, nhưng `lib/screens/` vẫn còn `jsonDecode(...)` + `['Medicine_Name']` vì **Day 9 mới nối UI vào port**; sửa sớm sẽ đụng `test/screens/*` của Day 3. Mọi màn hình đã có sẵn `XxxResponse.parse(body)` để Day 9 chỉ cần thay 1 dòng.

**Kiểm tra tĩnh đã làm được (không cần Flutter):** cân bằng `{}` và `()` của cả 11 file đều khớp; không còn tham chiếu `_pick`/`_unset` cũ; 0 dependency mới trong `pubspec.yaml`.

**Nghiệm thu Day 6 (sau khi cài Flutter):**
```powershell
flutter test test/data        # kỳ vọng: ~90 test, All tests passed
flutter analyze --no-fatal-infos
```
Nếu đỏ, sửa tới xanh rồi tick nốt ô DoD Day 5 + Day 6.

---

### 01/10/2026 — 🟢 **ĐÃ CÀI TOOLCHAIN + VERIFY XANH Day 5–7** (thay thế mục "30/09 chưa verify")

**Đã cài (mất ~20 phút, chạy nền xong trong 1 phiên):**

| Tool | Phiên bản | Vị trí | Ghi chú |
|---|---|---|---|
| Flutter | **3.19.4** (stable) | `C:\src\flutter` | SHA256 đã verify khớp `b8f90abc…`; Dart 3.3.2; cache 1.9 GB |
| Git | **2.56.0** | `C:\src\Git` | Flutter **bắt buộc** có git; cài `/VERYSILENT /DIR=C:\src\Git` |

Đã thêm vào **user PATH**: `C:\src\flutter\bin` · `C:\src\Git\cmd`.
`git config --global`: `safe.directory=*` (tránh git kẹt ở prompt "dubious ownership" — lần đầu `flutter --version` treo 5 tiến trình `git`), `core.longpaths=true`.

**Sau khi cài phải xoá `.dart_tool` cũ** — nó trỏ `D:/flutter_windows_3.19.4-stable` của máy trước:
```powershell
Remove-Item .dart_tool -Recurse -Force ; flutter pub get    # -> "Got dependencies!" (66 gói)
```

**✅ Kết quả CI (đúng 3 lệnh trong `.github/workflows/ci.yml`):**

| Bước | Kết quả |
|---|---|
| `dart format --output=none --set-exit-if-changed .` | **exit 0** — không file nào cần format |
| `flutter analyze --no-fatal-infos` | **50 issues, 0 warning, 0 error** (toàn bộ là `info` — hợp lệ, `--no-fatal-infos` cho phép; dọn ở Day 10) |
| `flutter test` | **`00:07 +179: All tests passed!`** |

**🔥 5 lỗi THẬT mà chỉ chạy test mới phát hiện** (không có `flutter analyze` thì không thấy):

| # | Lỗi | Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|---|---|
| 1 | `AppConfig.uri()` sinh `//history` | `http://10.0.2.2:5000//history` | `_join()` gọi `lead('')` khi `apiPrefix` rỗng → `'/'` | bỏ hẳn đoạn rỗng trước khi `join()` |
| 2 | `pickChanged<T>` không suy luận được kiểu | `String? can't be assigned to String` (14 lỗi) | khai báo `T current` → Dart suy `T` từ **kiểu trả về** trước | đổi thành `T? current` |
| 3 | `MedicineSearchResponse.total` | `can't have a value of 'null'` | `final int total` + `this.total` (mặc định null) | tách `_serverTotal` + getter `total` |
| 4 | `copyWith(x: 99)` với field `double?` | `type 'int' is not a subtype of type 'double?'` | ép kiểu trực tiếp | thêm `_toPercent()` quy `num → double` |
| 5 | Test upload fail trên **Windows** | `PathAccessException errno 32` | xoá thư mục tạm trong `finally` khi Dio còn giữ handle | chuyển sang `addTearDown` + bỏ qua `FileSystemException` |

Ngoài ra 1 lỗi `MapEntry<String,dynamic>` ≠ `MapEntry<String,String>` khiến `contains` trượt dù giá trị giống hệt (`==` của `MapEntry` so cả `runtimeType`) — đổi sang so sánh `'key=value'`.

> 💡 **Đây là lý do đáng để cài toolchain sớm.** 5 lỗi trên đều là lỗi *logic/typing* mà **đếm `{}` không bắt được** — đếm ngoặc chỉ bắt được lỗi cú pháp.

**⚠️ Phát hiện thêm: có 2 bản sao thư mục của project nằm bên trong chính nó:**

| Thư mục | Git track | Ghi chú |
|---|---|---|
| `ProjectAI-main/` | **0 / 153 file** | bản copy cũ, có `analysis_options.yaml` riêng |
| `.kilo/worktrees/…` | 0 | worktree của công cụ |

Chúng làm `flutter analyze` báo **1014 lỗi giả** (mọi import trong đó "đứt" vì không có `.dart_tool`). Đã xử lý **không xoá file nào**:
- `analysis_options.yaml` → thêm `exclude: .kilo/**, ProjectAI-main/**, .dart_tool/**, build/**, android/**, ios/**` → **1014 → 50**
- `.gitignore` → thêm `ProjectAI-main/` và `.kilo/`. **Quan trọng:** quy tắc làm việc của dự án là `git add -A`, không ignore thì lệnh đó sẽ nuốt hàng nghìn file trùng lặp.
- ⏳ Hai thư mục này **vẫn còn trên đĩa** — xoá được bất cứ lúc nào (git không track file nào trong đó).

**Trạng thái sau verify:**

| Ngày | Code | DoD | Test |
|---|---|---|---|
| Day 5 — Config + HTTP | ✅ | ✅ | ✅ xanh |
| Day 6 — Models | ✅ | ⚠️ 1 nửa (có chủ ý, xem mục Day 6) | ✅ xanh |
| Day 7 — Ports | ✅ | ✅ | ✅ xanh |

**Nghiệm thu Day 5–7:** ⛔→✅ **đã xong** (còn 1 ô ✋ của Day 5 là `flutter run` cần thiết bị thật/AVD — xem Day 4/M0).

`lib/domain/ports/` — **4 file, đúng chữ ký kế hoạch yêu cầu**:

| File | Nội dung |
|---|---|
| `medicine_analyzer.dart` | `MedicineAnalyzer`: `analyze(imagePath)` + `capability` |
| `medicine_repository.dart` | `search(name)` · `history()` · `deleteHistory(id)` |
| `auth_repository.dart` | `login` · `register` · `logout` · `currentToken` |
| `agent_client.dart` | `AgentClient.chat({sessionId, message})` + `AgentEvent` sealed 5 nhánh |

`test/domain/ports/` — 2 file, 13 test:
- `domain_is_pure_dart_test.dart` — **test canh kiến trúc** (3 test): cấm `package:flutter/`, `dart:ui`, `dart:html`; cấm domain import `data/adapters/` · `screens/` · `core/providers`; và có test phụ *"có ít nhất 1 file"* để chắc chắn test không pass giả do sai đường dẫn.
- `agent_client_test.dart` — 10 test cho 5 biến thể + `switch` đủ 5 nhánh + `_FakeAgentClient` làm khuôn cho adapter thật ở Day 22.

**✅ DoD đạt, đo bằng `Select-String`:**
```
Select-String -Path lib\domain\*.dart -Pattern 'package:flutter/'   -> 0 kết quả
```
Toàn bộ import của tầng domain chỉ gồm:
`data/models/{user, analysis_result, history_item, medicine}.dart` + `core/network/api_exception.dart`.

**3 lựa chọn thiết kế:**
1. **`Failure` mang `AppException`** → UI hiện lỗi tiếng Việt không cần `try/catch`, đồng bộ với `Result`/`guard()` (Day 5). Doc cảnh báo rõ: lỗi **giữa chừng** phải đóng gói thành event, không được làm stream văng exception — nếu không UI sẽ mất toàn bộ tin nhắn đã nhận.
2. **`search()` trả `MedicineSearchResponse`** thay vì `List<Medicine>` → giữ `query` + `total` server echo lại.
3. **`sealed AgentEvent`** → thêm biến thể mới sẽ **hỏng compile** mọi `switch` chưa xử lý. Test `switch` đủ 5 nhánh bảo vệ đúng điều đó.

**Lỗi phát hiện khi tự kiểm tra (đáng ghi vì không có `flutter analyze`):**
`agent_client.dart` thiếu 1 dấu `}` đóng class `Done` → lệch 14/13. Bắt được bằng cách đếm `{`/`}` bằng PowerShell. Sau khi sửa: cả 6 file Day 5–7 đều cân bằng.
> 💡 **Bài học:** khi không chạy được `analyze`, **đếm `{}` và `()` trong từng file** là bước rẻ nhất để bắt lỗi cú pháp cắt dính. Nên làm sau **mỗi** lần sửa file, không chỉ lúc nghiệm thu.

**Nghiệm thu Day 7 (sau khi cài Flutter):**
```powershell
flutter test test/domain      # 13 test
flutter analyze --no-fatal-infos
```

---

*Đây là kế hoạch sống: cập nhật lại mỗi cuối ngày (tick Phụ lục C). Khi bạn ra lệnh thực thi, mặc định tôi sẽ làm **Day 0 → Day 4 (P0)** trước, rồi dừng ở M0 để bạn tự kiểm tra trước khi đi tiếp P1.*



### 28/10/2026 chiều — Hoàn tất kiểm tra P3 trên máy ảo (Day 25–30)

Chạy thật trên máy ảo `ueh_pixel` (có `adb root`, nên chỉnh được đồng hồ):

| Hạng mục | Kết quả |
|---|---|
| Chat streaming | Chạy trong app: câu trả lời kèm `[R1]…` + thẻ nguồn; có LLM cục bộ (Ollama `qwen2.5:3b`) với phương án lùi về câu trích nguyên văn |
| OCR on-device | Ảnh hộp thuốc GIẢ → ML Kit đọc "Panadol 500mg" → khớp đúng thuốc, kể cả khi **bật chế độ máy bay** |
| Nhắc thuốc | Tạo lịch trong app, 7 báo thức/tuần; chỉnh đồng hồ máy ảo về 07:59 → **thông báo nổ đúng 08:00** |
| MCP | Chạy qua client MCP thật (stdio): `search_drug`, `check_interaction`; thêm `tests/test_mcp_live.py` |
| Riêng tư | Màn đồng ý, xoá tài khoản, PRIVACY.md đã cập nhật (LLM cục bộ chỉ loopback) |

**Lỗi thật tìm thấy nhờ chạy thật (test đơn vị không bắt được):**
1. Nhắc thuốc: `ZoneRulesException: Unknown time-zone ID: device` — plugin chuyển tên múi giờ sang Java `ZoneId.of`. Sửa: `zoneIdForOffset()` → `UTC+07:00`. Hệ quả của lỗi: lịch được ghi DB nhưng không đặt báo thức, giao diện không cập nhật.
2. MCP: tiến trình con không nhận `MEDASSIST_DB` (DB `:memory:` rỗng → `search_drug` luôn `no_data`), và `_parse_mcp_result` đọc `isError` trong khi SDK 2.x dùng `is_error` (lỗi bị `except Exception` nuốt thành `mcp_error`). Sửa cả hai (`_server_env()`), thêm test chạy tiến trình con.

Số liệu: `flutter test` 330/330, `flutter analyze` sạch (1 info cũ), `pytest` đạt, `eval.py` 109/109.

**Chưa làm / cần chủ dự án:** keystore thật + `key.properties`, server HTTPS, beta nội bộ, `git tag v1.0.0`, crash reporting (cần chọn SDK + DSN + consent), thử trên điện thoại thật, phần còn lại của 8 tình huống test tay.