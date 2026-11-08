# SETUP Windows — Danh sách file cần tải (đã kiểm chứng 21/09/2026)

> Mọi link dưới đây được lấy từ **repository chính thức** (`dl.google.com/android/repository/repository2-3.xml`,
> `api.adoptium.net`) và đã test trả về **HTTP 200**. Dung lượng/ checksum lấy theo dữ liệu repo, không ước lượng.
> Máy này: C: còn **~5.2 GB**, D: còn **~79 GB** → **tải hết về `D:\dev`** rồi mới cài.

---

## A. BẮT BUỘC — thiếu là không build/chạy được Android

### 1) JDK 17 — Temurin `17.0.20.1+1` (Windows x64, ZIP) → `D:\dev\jdk17`
- **ZIP (khuyên dùng, không cần admin, không đụng C:):**
  `https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20.1%2B1/OpenJDK17U-jdk_x64_windows_hotspot_17.0.20.1_1.zip`
- Dung lượng: **190,817,615 B ≈ 182 MB** · **SHA-256** `e53a79c3c3d86865bd7e787903884331068e71321714ffd44f145785affc7cb0`
- Trang release: `https://github.com/adoptium/temurin17-binaries/releases/tag/jdk-17.0.20.1%2B1`
- **Vì sao 17 mà không phải 21:** `android/gradle/wrapper/gradle-wrapper.properties` đang là **Gradle 8.3**; Gradle 8.3 chỉ chạy tới JDK 20 → JDK 21 sẽ fail. (JDK 21 cần Gradle ≥ 8.5.)
- Bản MSI (nếu thích installer, cần admin): `.../OpenJDK17U-jdk_x64_windows_hotspot_17.0.20.1_1.msi` (168 MB), hoặc
  `winget install --id EclipseAdoptium.Temurin.17.JDK -e --override "/qn INSTALLDIR=\"D:\dev\jdk17\" ADD_PATH=1 ENABLE_JAVA_HOME=1"`

```powershell
New-Item -ItemType Directory -Force -Path 'D:\dev' | Out-Null
Invoke-WebRequest 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.20.1%2B1/OpenJDK17U-jdk_x64_windows_hotspot_17.0.20.1_1.zip' -OutFile 'D:\dev\jdk17.zip'
(Get-FileHash 'D:\dev\jdk17.zip' -Algorithm SHA256).Hash.ToLower()   # phải = e53a79c3...c7cb0
Expand-Archive 'D:\dev\jdk17.zip' -DestinationPath 'D:\dev' -Force
Rename-Item 'D:\dev\jdk-17.0.20.1+1' 'jdk17'    # zip có 1 thư mục cha, đổi tên cho gọn
setx JAVA_HOME 'D:\dev\jdk17'
setx PATH "$([Environment]::GetEnvironmentVariable('Path','User'));%JAVA_HOME%\bin"
# MỞ PowerShell MỚI:
java -version
```

### 2) Android **command line tools** — revision **23**, build `16111833` (Windows)
- `https://dl.google.com/android/repository/commandlinetools-win-16111833_latest.zip`
- Dung lượng: **154,957,218 B ≈ 148 MB** · **SHA-1** `57d04f2d75eb8e8fffc5000a987e5de4b5a63e9d` (Google publish SHA-1 cho gói này)
- Cần để có `sdkmanager` / `avdmanager` (chấp nhận license + tạo AVD). `flutter emulators` **không** có `--create`.
- ⚠️ Chạy `sdkmanager` **yêu cầu JDK** → phải xong mục 1 trước.
- ⚠️ **Bẫy thư mục:** phải thành `...\Sdk\cmdline-tools\latest\bin\sdkmanager.bat`. Nếu thành `cmdline-tools\cmdline-tools\latest` thì **không được nhận ra**. SDK hiện ở C: (nơi duy nhất Flutter tìm) → gói này ~300 MB sau giải nén, C: còn 5.2 GB nên vừa.

```powershell
$sdk = "$env:LOCALAPPDATA\Android\Sdk"
Invoke-WebRequest 'https://dl.google.com/android/repository/commandlinetools-win-16111833_latest.zip' -OutFile 'D:\dev\cmdline-tools.zip'
(Get-FileHash 'D:\dev\cmdline-tools.zip' -Algorithm SHA1).Hash.ToLower()   # phải = 57d04f2d...e830
Expand-Archive 'D:\dev\cmdline-tools.zip' -DestinationPath 'D:\dev\ct' -Force
New-Item -ItemType Directory -Force -Path "$sdk\cmdline-tools" | Out-Null
Move-Item 'D:\dev\ct\cmdline-tools' "$sdk\cmdline-tools\latest"      # => Sdk\cmdline-tools\latest\bin
Test-Path "$sdk\cmdline-tools\latest\bin\sdkmanager.bat"             # phải là True
& "$sdk\cmdline-tools\latest\bin\sdkmanager.bat" --licenses          # trả 'y' hết
```

---

## B. KHÔNG cần tải — máy đã có (đừng mất công)

| Thứ | Trạng thái trên máy | Ghi chú |
|---|---|---|
| Flutter SDK | `D:\flutter_windows_3.19.4-stable\flutter` (3.19.4 / Dart 3.3.2) | dùng tới Day 25 |
| Android SDK + `platform-tools`(adb) + `build-tools` + `emulator` | `C:\Users\TEN_BAN\AppData\Local\Android\Sdk` | đầy đủ |
| **System image cho AVD** | `android-35\google_apis\x86_64` ✅ · `android-33\google_apis\x86_64` ✅ | **khỏi tải**; `android-34\google_apis` **rỗng** → đừng chọn |
| NDK | 24.0 / 26.3 / 27.0 (build đang pin `26.3.11579264`) | app chưa có plugin native |
| Python / Git / Node | 3.12.5 / 2.55 / v24 | Python dùng cho backend FastAPI (Day 11) |
| 7-Zip | **không cần** — đã có `tar` (bsdtar 3.8.8) và `Expand-Archive` | |

## C. Tuỳ chọn — chỉ tải khi cần

| Khi nào | Tải gì | Link |
|---|---|---|
| Muốn có GUI tạo AVD/debugger (thay cho mục A2) | **Android Studio** (bản mới nhất: Quail 4 \| 2026.1.4 Patch 1) — kèm sẵn JDK + cmdline-tools, nhưng **~2 GB trên C: đang chật** → chỉ nên dùng nếu chấp nhận | `https://developer.android.com/studio` |
| **Day 26** (cần Flutter ≥ 3.22 cho ML Kit Document Scanner) | **Flutter 3.47.5 stable / Dart 3.13.4** — *chưa tải bây giờ*, đổi Flutter giữa chừng phải test lại toàn bộ | `https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.47.5-stable.zip` |
| Day 36 (deploy backend, tuỳ chọn) | Docker Desktop | `https://www.docker.com/products/docker-desktop/` |

## D. Tổng dung lượng phải tải để thông M0

| File | Về đâu | Dung lượng |
|---|---|---|
| `OpenJDK17U-jdk_x64_windows_hotspot_17.0.20.1_1.zip` | `D:\dev\jdk17` | 182 MB (giải nén ~300 MB, **toàn bộ trên D:**) |
| `commandlinetools-win-16111833_latest.zip` | `...\Android\Sdk\cmdline-tools\latest` | 148 MB (giải nén ~350 MB, **nằm trên C:**) |
| **Tổng** | | **~330 MB tải / ~650 MB ổ cứng** |

> Trước khi giải nén mục A2, **dời cache Gradle/AVD sang D:** (xem `TROUBLESHOOTING.md` mục 3) để C: không bị build đầu tiên làm đầy.
