# Mở máy ảo + mở app bản dev nối server demo (10.0.2.2 = máy host).
#
# Mặc định KHÔNG build lại (Gradle ngốn RAM, dễ lỗi "paging file is too small" khi chạy
# cùng máy ảo + Ollama): dùng APK đã build sẵn, cài nếu máy ảo chưa có rồi mở app.
# Thêm -Build khi vừa sửa code Dart và cần build lại (đóng bớt Chrome/IDE trước).
param([switch]$Build)

$env:JAVA_HOME = "D:\dev\jdk17\jdk-17.0.20.1+1"
$env:ANDROID_HOME = "D:\Android\Sdk"
$env:ANDROID_SDK_ROOT = "D:\Android\Sdk"
$env:ANDROID_AVD_HOME = "D:\dev\avd"
$env:GRADLE_OPTS = "-Dorg.gradle.offline=true"
$env:Path = "D:\flutter_windows_3.19.4-stable\flutter\bin;$env:JAVA_HOME\bin;D:\Android\Sdk\platform-tools;$env:Path"
$adb = "D:\Android\Sdk\platform-tools\adb.exe"
$pkg = "com.medassist.app.dev"
$apk = "$PSScriptRoot\..\build\app\outputs\flutter-apk\app-dev-debug.apk"
Set-Location "$PSScriptRoot\.."

# Chỉ mở máy ảo nếu chưa có thiết bị nào.
if (-not ((& $adb devices) -match "emulator-\d+\s+device")) {
    Start-Process "D:\Android\Sdk\emulator\emulator.exe" -ArgumentList "-avd ueh_pixel -no-snapshot-save"
}
& $adb wait-for-device
# Chờ hệ điều hành khởi động xong hẳn (mở app quá sớm sẽ lỗi "Activity class does not exist").
while ((& $adb shell getprop sys.boot_completed 2>$null).Trim() -ne "1") { Start-Sleep -Seconds 2 }
Start-Sleep -Seconds 3

if ($Build) {
    flutter run --flavor dev --dart-define=APP_FLAVOR=dev --dart-define=API_BASE_URL=http://10.0.2.2:8000
    if ($LASTEXITCODE -eq 0) { return }
}

$installed = (& $adb shell pm list packages $pkg) -match $pkg
# ⚠️ Cài đè bản MỚI HƠN ngay cả khi đã cài: nếu không, sau khi build lại (vd thêm tính
# năng) app trên máy ảo vẫn là bản cũ mà không ai nhận ra — đã gặp thật khi thêm mục
# Tin tức: script chỉ "đã cài -> mở luôn", bỏ qua bản build mới.
$apkTime = if (Test-Path $apk) { (Get-Item $apk).LastWriteTimeUtc } else { $null }
$stampFile = "$PSScriptRoot\.last_installed_apk_time"
$installedTime = if (Test-Path $stampFile) { [datetime](Get-Content $stampFile -Raw) } else { $null }
$needInstall = (-not $installed) -or (-not $installedTime) -or ($apkTime -gt $installedTime)
if ($needInstall) {
    if (-not (Test-Path $apk)) { Write-Host "Chua co APK: chay lai voi -Build"; return }
    & $adb install -r $apk
    if ($LASTEXITCODE -eq 0) { $apkTime.ToString("o") | Set-Content $stampFile }
}
& $adb shell monkey -p $pkg -c android.intent.category.LAUNCHER 1 | Out-Null
Write-Host "Da mo app $pkg tren may ao."
