# Đưa ảnh từ máy tính vào máy ảo / điện thoại để chọn trong màn "Quét thuốc".
#
# Ảnh chỉ được chép sang thiết bị qua cáp/adb (không đi qua mạng, không lên server);
# OCR vẫn chạy ngay trên thiết bị. Script KHÔNG đọc nội dung ảnh.
#
# Cách dùng:
#   .\tools\push_image.ps1 -Path C:\anh\hop-thuoc.jpg
#   .\tools\push_image.ps1 -Path C:\anh\a.jpg, C:\anh\b.png
#   .\tools\push_image.ps1 -Folder C:\anh\hop-thuoc          # mọi ảnh trong thư mục
#   .\tools\push_image.ps1 -Path .\a.jpg -Serial emulator-5554   # khi có nhiều thiết bị
#
# Trong app: Quét thuốc → Chọn ảnh → chọn ảnh vừa đẩy (nằm trong album "MedAssist").
param(
    [string[]]$Path = @(),
    [string]$Folder,
    [string]$Serial,
    [switch]$Clean   # xoá các ảnh đã đẩy trước đó trên thiết bị (chỉ thư mục MedAssist)
)

$adb = "D:\Android\Sdk\platform-tools\adb.exe"
if (-not (Test-Path $adb)) { $adb = "adb" }
$remote = "/sdcard/Pictures/MedAssist"
$ext = @(".jpg", ".jpeg", ".png", ".webp")

function Invoke-Adb {
    if ($Serial) { & $adb -s $Serial @args } else { & $adb @args }
}

# --- Thiết bị
$devices = @(& $adb devices | Select-String "\sdevice$" | ForEach-Object { ($_ -split "\s+")[0] })
if ($devices.Count -eq 0) { Write-Error "Khong thay thiet bi nao. Mo may ao (tools\demo_app.ps1) hoac cam dien thoai (bat Debug USB)."; exit 1 }
if ($devices.Count -gt 1 -and -not $Serial) {
    Write-Error ("Co nhieu thiet bi: " + ($devices -join ", ") + ". Chay lai voi -Serial <ten>.")
    exit 1
}

if ($Clean) {
    # ⚠️ Xoá bản ghi trong thư viện ảnh (MediaStore) TRƯỚC khi xoá file. Nếu chỉ `rm`, Android vẫn giữ
    # "ảnh ma" (ô trống) trong bộ chọn ảnh và app không chọn được ảnh.
    Invoke-Adb shell "content delete --uri content://media/external/images/media --where `"relative_path LIKE 'Pictures/MedAssist%'`"" | Out-Null
    Invoke-Adb shell "rm -rf $remote" | Out-Null
    Write-Host "Da xoa $remote tren thiet bi."
    if (-not $Path -and -not $Folder) { return }
}

# --- Gom danh sách file
$files = @()
foreach ($p in $Path) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { Write-Warning "Bo qua (khong thay file): $p"; continue }
    $files += Get-Item -LiteralPath $p
}
if ($Folder) {
    if (-not (Test-Path -LiteralPath $Folder -PathType Container)) { Write-Error "Khong thay thu muc: $Folder"; exit 1 }
    $files += Get-ChildItem -LiteralPath $Folder -File | Where-Object { $ext -contains $_.Extension.ToLower() }
}
if ($files.Count -eq 0) { Write-Error "Khong co anh nao de day. Dung -Path hoac -Folder."; exit 1 }

# --- Đẩy từng ảnh
Invoke-Adb shell "mkdir -p $remote" | Out-Null
$n = 0
foreach ($f in $files) {
    if ($ext -notcontains $f.Extension.ToLower()) { Write-Warning "Bo qua (khong phai anh jpg/png/webp): $($f.Name)"; continue }
    if ($f.Length -gt 30MB) { Write-Warning "Bo qua (qua 30 MB): $($f.Name)"; continue }
    # Tên an toàn: bỏ ký tự lạ/dấu cách để adb không hiểu nhầm.
    $safe = ($f.BaseName -replace "[^A-Za-z0-9_-]", "_") + $f.Extension.ToLower()
    $dest = "$remote/$safe"
    $out = Invoke-Adb push $f.FullName $dest 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Warning "Loi khi day $($f.Name): $out"; continue }
    # Báo hệ thống quét để ảnh hiện trong bộ chọn ảnh.
    Invoke-Adb shell "am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://$dest" | Out-Null
    Write-Host "OK  $($f.Name)  ->  $dest"
    $n++
}
Write-Host ""
Write-Host "Da day $n anh. Trong app: Quet thuoc -> Chon anh -> album 'MedAssist' (hoac Recent)."
