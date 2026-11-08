# TROUBLESHOOTING — Môi trường Windows & lỗi đã gặp

> Ghi lại **lỗi thật + lệnh đo + cách sửa** để lần sau (hoặc máy khác) không mất thời gian dò lại.
> Ngày đo gần nhất: 21/09/2026.
> 📦 **Danh sách link tải (JDK 17 + cmdline-tools) kèm checksum:** xem [`SETUP_WINDOWS.md`](SETUP_WINDOWS.md).

---

## 0. Trạng thái máy — số đo thật (không phải dự đoán)

| Thành phần | Kết quả đo | Kết luận |
|---|---|---|
| Flutter SDK | `D:\flutter_windows_3.19.4-stable\flutter` (3.19.4 / Dart 3.3.2) | ✅ đủ dùng tới Day 25 |
| Git | 2.55.0 | ✅ |
| Python | 3.12.5 | ✅ sẵn sàng cho backend (Day 11) |
| Node | v24.14.1 | ✅ (không dùng tới) |
| Android SDK | `C:\Users\TEN_BAN\AppData\Local\Android\Sdk` — có `platform-tools`(adb), `build-tools`, `emulator`, platform 33/34/35, NDK 24/26/27, system-images | ✅ |
| **JDK (Java)** | `java` không trong PATH, `JAVA_HOME` unset, không có `Android Studio\jbr`, không có `Eclipse Adoptium` | ❌ **PHẢI CÀI** |
| **Android `cmdline-tools`** | không tồn tại → không có `sdkmanager`, `avdmanager` | ❌ **PHẢI CÀI** (accept license + tạo AVD) |
| AVD | `~\.android\avd` không tồn tại → `flutter emulators` trả *No emulators available* | ❌ chưa có device |
| Điện thoại thật | `flutter devices` không thấy device Android | ❌ chưa cắm / chưa bật USB debugging |
| Docker | chưa cài | ⚪ chưa cần (Day 36+ mới tuỳ chọn) |
| VS Build Tools | chưa cài | ⚪ **không cần** (không build target Windows desktop) |
| Xcode / CocoaPods | không có (máy Windows) | ⚪ build iOS phải làm trên macOS |
| Hyper-V / WHPX | `HypervisorPresent = True` | ✅ emulator có tăng tốc phần cứng |
| RAM | 15.6 GB | ✅ đủ cho 1 AVD |
| **Dung lượng C:** | **còn ~4 GB** | ⚠️ **QUAN TRỌNG — xem mục 3** |
| Dung lượng D: | còn ~79 GB | ✅ đích dời cache/AVD |

---

## 1. Lỗi: `ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH`

**Xuất hiện khi:** `flutter build apk --debug`, `flutter run` (bước chạy `gradlew`).
**Nguyên nhân:** module `android/` của app Flutter là project **Gradle/Kotlin**, mà Gradle + Android Gradle Plugin **chạy trên JVM**. JDK là của **Java**, không phải của Flutter — Flutter chỉ sinh ra project Android đó. (iOS/macOS không cần JDK.)
**Cách sửa:**
```powershell
winget install --id EclipseAdoptium.Temurin.17.JDK -e
# MỞ CỬA SỔ PowerShell MỚI (biến môi trường chỉ nạp khi mở shell mới)
java -version            # phải ra: openjdk version "17..."
```
Nếu `JAVA_HOME` vẫn trống, chọn 1 trong 2:
```powershell
setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot"              # cách 1: hệ thống
flutter config --jdk-dir "C:\Program Files\Eclipse Adoptium\jdk-17.0.x-hotspot"     # cách 2: chỉ cho Flutter
```
**Mẹo:** cài **Android Studio** có luôn JDK (thư mục `jbr`) + `cmdline-tools` + GUI tạo AVD — gọn nhất nếu chấp nhận ~1.5 GB disk.

---

## 2. Lỗi: `flutter doctor --android-licenses` báo `sdkmanager not recognized`

**Nguyên nhân:** thiếu `cmdline-tools` (`sdkmanager`, `avdmanager`). `flutter doctor` đã cảnh báo sẵn: *Android SDK install without cmdline-tools → some functionality unavailable*.
**Cách sửa (không cần Android Studio):** tải `commandlinetools-win-13114758_latest.zip` từ
<https://developer.android.com/studio#command-line-tools-only>, giải nén và **đổi tên** thư mục thành `latest` để có đúng đường dẫn:
```
%LOCALAPPDATA%\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat
```
Rồi (cần JDK ở mục 1):
```powershell
& "$env:LOCALAPPDATA\Android\Sdk\cmdline-tools\latest\bin\sdkmanager.bat" --licenses
```
> bẫy hay gặp: nếu thành `cmdline-tools\cmdline-tools\latest\bin` thì **không được nhận ra** — phải là `cmdline-tools\latest\bin`.

## 3. ⚠️ C: chỉ còn ~4 GB — AVD và Gradle sẽ làm đầy ổ

**Vấn đề:** AVD (system image + data) chiếm 3–10 GB, cache Gradle 1–3 GB, **mặc định cả hai nằm trên C:**. Với 4 GB trống thì build đầu tiên/tạo AVD sẽ fail hoặc treo, Windows cũng chậm theo.
**Cách sửa — dời cả hai sang D: (79 GB trống), chạy 1 lần:**
```powershell
New-Item -ItemType Directory -Force -Path 'D:\dev\.gradle','D:\dev\AndroidAvd' | Out-Null
setx GRADLE_USER_HOME "D:\dev\.gradle"
setx ANDROID_AVD_HOME "D:\dev\AndroidAvd"
# MỞ PowerShell MỚI rồi kiểm tra:
$env:GRADLE_USER_HOME; $env:ANDROID_AVD_HOME
```
**Cách thay thế tốn 0 GB:** chạy trên **điện thoại Android thật** qua USB (mục 4) — không cần AVD, không phải cài thêm gì (adb đã có sẵn trong `platform-tools`).

---

## 4. Lỗi: `flutter emulators` → *No emulators available*

**Sự thật đã đo:** `flutter emulators` **chỉ hỗ trợ `--list` và `--launch`**, **không có `--create`** → không tạo được AVD bằng lệnh Flutter. Phải dùng `avdmanager` (cần mục 1 + 2) hoặc GUI Device Manager của Android Studio.
**Tạo AVD bằng dòng lệnh:**
```powershell
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
& "$sdk\cmdline-tools\latest\bin\avdmanager.bat" create avd -n pixel_api35 -k "system-images;android-35;google_apis;x86_64" -d pixel_7
& "$sdk\emulator\emulator.exe" -avd pixel_api35 -no-snapshot-save
flutter devices          # phải thấy "pixel_api35 (mobile) • emulator-5554"
flutter run -d emulator-5554
```
> **Ảnh đã đo trên máy này** (CPU `12th Gen Intel i5-12450H`, `PROCESSOR_ARCHITECTURE = AMD64` → phải dùng ABI **`x86_64`**):
>
> | Package system-image đã tải xong | Dùng được? |
> |---|---|
> | `system-images;android-35;google_apis;x86_64` | ✅ **khuyên dùng — khỏi tải thêm** |
> | `system-images;android-35;google_apis_playstore;x86_64` | ✅ được, nhưng nặng hơn (có Play Store) |
> | `system-images;android-33;google_apis;x86_64` | ✅ được (nếu muốn API thấp hơn) |
> | `android-34\google_apis\` | ❌ **rỗng** — chỉ có thư mục cha, chưa có ABI nào → `avdmanager` sẽ báo không tìm thấy package |
>
> `HypervisorPresent = True` → emulator chạy bằng WHPX (nhanh), RAM 15.6 GB đủ cho 1 AVD.
> `google_apis` (không phải `playstore`) có `adb root` → tiện khi debug, nhẹ hơn, đủ cho app này.

**Chạy bằng điện thoại thật (tiết kiệm disk nhất):** Settings → About phone → bấm 7 lần *Build number* → Developer options → bật **USB debugging** → cắm cable → `flutter devices` → `flutter run`.

---

## 5. Lỗi: `CLEARTEXT communication to 10.0.2.2 not permitted` khi gọi API

**Nguyên nhân:** từ Android 9 (API 28) mặc định **cấm HTTP**; iOS có ATS tương tự.
**Cách sửa (đã làm ở Day 2):** `android/app/src/debug/res/xml/network_security_config.xml` cho phép cleartext với `10.0.2.2`/`localhost`/`127.0.0.1` + khai báo `android:networkSecurityConfig` **trong debug manifest** (release vẫn bắt buộc HTTPS). iOS: `NSExceptionDomains` cho `localhost`/`127.0.0.1` trong `Info.plist`.
**Mẹo:** `10.0.2.2` chỉ là địa chỉ "máy host nhìn từ **emulator**"; **điện thoại thật** phải dùng IP LAN của máy (vd `192.168.1.10:8000`) → vì vậy API base URL phải cấu hình được, không hardcode (Day 5 làm `AppConfig`).

---

## 6. Lỗi build Android: NDK / minSdk / namespace

- **`NDK not configured` hoặc Gradle cố tải NDK rồi fail:** Flutter 3.19 mặc định NDK `25.1.8937393` nhưng máy chỉ có 24.0 / 26.3 / 27.0 → đã **pin `ndkVersion = "26.3.11579264"`** trong `android/app/build.gradle`. App chưa có plugin native nên dùng NDK nào cũng được; thêm plugin native thì kiểm tra lại.
- **`uses-sdk:minSdkVersion 21 cannot be smaller than version XX`** (xung đột khi thêm `firebase_*`, `flutter_tts`…): đã đặt **`minSdk = 26`** (Day 2); khi cần dùng `minSdk = maxOf(26, other)`.
- **`Namespace not specified`:** AGP 8 yêu cầu `namespace` trong `build.gradle` (đã đặt `com.medassist.app`) **và** `MainActivity.kt` phải ở `kotlin/com/medassist/app/` với `package com.medassist.app` — hai chỗ này phải đổi cùng lúc.
- **Warning `enableJetifier`:** đã xoá khỏi `android/gradle.properties` (AndroidX là mặc định từ AGP 7).

## 7. Lỗi runtime iOS: app bị kill khi mở thư viện ảnh

**Nguyên nhân:** thiếu `NSPhotoLibraryUsageDescription` / `NSCameraUsageDescription` → iOS **terminate ngay**, Dart không nhận exception nào nên rất khó debug. Đã thêm ở Day 2.
Máy Windows **không build/test iOS được** → các bước Day 19–23 chỉ kiểm tra bằng code + widget test; build thật chạy ở Day 31 trên macOS.

---

## 8. Cảnh báo Git trên Windows

`warning: in the working copy of 'x.dart', LF will be replaced by CRLF` — **vô hại** với Dart. Muốn hết cảnh báo:
```powershell
git config --global core.autocrlf true
```

---

## 9. `'flutter' is not recognized…` — mở project trên máy KHÔNG cài Flutter (30/09/2026)

**Triệu chứng:** chạy bất kỳ lệnh nào của Flutter/Dart/Git đều báo *not recognized*:
```
'flutter' is not recognized as an internal or external command,
operable program or batch file.
INFO: Could not find files for the given pattern(s).
```
**Nguyên nhân (đã dò ra từ số đo thật, không phải phỏng đoán):**
| Bằng chứng | Kết luận |
|---|---|
| `where flutter` / `where dart` / `where git` → *Could not find files* | Không có toolchain nào trong `PATH` |
| `dir /b /ad D:\` → *The system cannot find the path specified* | **Không có ổ `D:`** |
| `dir /b /ad C:\Users` → chỉ `Administrator`, `Public`, `Default` | Không có profile `Tuan Anh` |
| `.dart_tool/package_config.json` → `file:///D:/flutter_windows_3.19.4-stable/flutter/...` | Project được **copy từ máy cũ**, cache trỏ ổ D: |
| `package_config.json` → pub cache `C:/Users/Tuan%20Anh/AppData/Local/Pub/Cache/...` | Cache thuộc user `Tuan Anh`, **không tồn tại ở đây** |

⇒ Đây **không phải** project hỏng; chỉ là **môi trường máy mới chưa cài toolchain**. Code Day 5 vẫn hợp lệ, chỉ là chưa chạy được lệnh.

**Cách sửa (chọn 1):**
```powershell
# A) Cài Flutter 3.19.4 + Git, rồi chạy lại toàn bộ verify của ngày đang dở
winget install --id Git.Git -e
#    Flutter: tải https://docs.flutter.dev/release/archive?os=windows  -> giải nén, vd C:\src\flutter
#    $env:Path += ";C:\src\flutter\bin;C:\src\flutter\bin\cache\dart-sdk\bin"
flutter --version && flutter doctor

# B) Quay lại máy gốc (có D:\flutter_windows_3.19.4-stable\flutter + JDK + AVD) để tiếp tục
```
Sau đó **bắt buộc** chạy lại vì `package_config.json` đang trỏ sai ổ đĩa:
```powershell
Remove-Item .dart_tool -Recurse -Force
flutter pub get
```

⚠️ **Bài học:** luôn commit `.dart_tool` **không** (`.gitignore` đã loại), và khi copy project sang máy khác thì `flutter pub get` bắt buộc phải chạy lại — đừng tin `.dart_tool` cũ.

---

## 9. Lỗi dependency (đã xử lý ở Day 1 — lưu lại để hiểu vì sao)

| Lỗi `pub get` | Nguyên nhân thật | Cách đã sửa |
|---|---|---|
| `image_picking: Requires Flutter version >=3.22.2` | `image_picking 0.0.9` có plugin native, cần Flutter mới | Xoá — `image_picker` đã có; Day 26 dùng **ML Kit Document Scanner** |
| `image 5.0.9 requires Dart >=3.4` | `image 5.x` mới hơn Dart 3.3.2 của máy | Xoá `image` (không dùng tới) |
| `package_info_plus 1.0.4 <2.0.0` xung đột `shared_preferences` | Package bị khoá ở phiên bản cũ trong `pubspec.lock` | Xoá `package_info_plus` |
| `tflite_flutter_helper` không resolve được | Library **đã ngừng bảo trì/archived**, peg `collection 1.17.0`, không hợp `tflite_flutter >=0.10` | Xoá hẳn; hướng đi mới: **ML Kit (scan) + DocVQA (trắc nghiệm)**, không tự viết pipeline TFLite |
| `sdk: ">=3.4.0 <4.0.0"` không tương thích Dart 3.3.2 | pubspec đòi Dart mới hơn máy | Hạ về `^3.3.0`; **nâng Flutter ở Day 26** rồi mới nâng lại constraint |

---

## 10. Checklist mỗi khi bắt đầu làm việc

```powershell
cd d:\Mobile\ProjectAI-main
flutter --version          # 3.19.4
java -version              # 17.x            <- mục 1
flutter doctor             # không còn "!" ở Android toolchain
flutter devices            # phải có device để chạy
flutter test               # phải XANH trước khi code ngày mới
```

---

## 11. Cái KHÔNG cần cài (đỡ mất thời gian)

- **Visual Studio / Build Tools** — chỉ cần cho target **Windows desktop**; roadmap nhắm Android (chính) + iOS (sau).
- **CocoaPods / Xcode** — chỉ có trên macOS; build iOS thật làm khi có máy Mac.
- **Docker** — tuỳ chọn ở Day 36; khi dev chạy `uvicorn` trực tiếp bằng Python 3.12.5 (đã có).
- **NDK / system-image bổ sung** — máy đã có, không tải thêm.
- **Firebase Console / API Key / `google-services.json`** — tới **Day 26** mới tạo. **Cảnh báo:** hiện repo *không* có `google-services.json` và `GoogleService-Info.plist`, nên **không được** thêm plugin `firebase_*` vào `pubspec.yaml` trước Day 26 — Gradle sẽ fail ngay ở bước `processDebugGoogleServices`.

---

## 12. Việc môi trường còn tồn (tính tới Day 3)

| # | Việc | Lệnh/Ghi chú | Ai làm |
|---|---|---|---|
| 1 | Cài JDK 17 | `winget install --id EclipseAdoptium.Temurin.17.JDK -e` | 🧑 bạn (cần quyền admin) |
| 2 | Dời cache sang D: (C: chỉ còn 4 GB) | `setx GRADLE_USER_HOME`/`ANDROID_AVD_HOME` (mục 3) | 🧑 bạn |
| 3 | Cài `cmdline-tools` | mục 2 — hoặc cài Android Studio cho đủ bộ | 🧑 bạn |
| 4 | Có device: AVD **hoặc** điện thoại thật | mục 4 | 🧑 bạn |
| 5 | Chạy `flutter run` + `flutter doctor --android-licenses` | DoD còn treo của Day 2 / M0 | 🤖 sau khi 1–4 xong |


