# Tự đồng bộ ảnh từ một thư mục trên MÁY TÍNH sang máy ảo / điện thoại.
# Bỏ ảnh vào thư mục -> vài giây sau ảnh hiện trong app: Quét thuốc -> Chọn ảnh -> album "MedAssist".
#
#   .\tools\sync_images.ps1                      # theo dõi thư mục mặc định: <dự án>\anh_thuoc
#   .\tools\sync_images.ps1 -Folder "D:\anh"     # theo dõi thư mục khác (ảnh của bạn)
#   .\tools\sync_images.ps1 -Once                # đẩy các ảnh hiện có rồi thoát
#   .\tools\sync_images.ps1 -Serial emulator-5554   # khi có nhiều thiết bị
#
# Bấm Ctrl+C để dừng. Ảnh chỉ được chép qua adb sang thiết bị (không lên mạng, không lên server);
# OCR chạy ngay trên thiết bị. Script không đọc nội dung ảnh.
param(
    [string]$Folder,
    [string]$Serial,
    [switch]$Once,
    [int]$IntervalSeconds = 2
)

$root = Resolve-Path "$PSScriptRoot\.."
if (-not $Folder) { $Folder = Join-Path $root "anh_thuoc" }
if (-not (Test-Path -LiteralPath $Folder)) {
    New-Item -ItemType Directory -Force $Folder | Out-Null
    Write-Host "Da tao thu muc: $Folder"
}
$Folder = (Resolve-Path -LiteralPath $Folder).Path
$push = Join-Path $PSScriptRoot "push_image.ps1"
$ext = @(".jpg", ".jpeg", ".png", ".webp")

Write-Host "Theo doi thu muc: $Folder"
Write-Host "Tha anh (jpg/png/webp) vao thu muc nay. Ctrl+C de dung."

# Ghi nhớ ảnh đã đẩy: "đường dẫn|ngày sửa|dung lượng". Ảnh sửa lại sẽ được đẩy lại.
$pushed = @{}
$lastSize = @{}

function Get-Images {
    Get-ChildItem -LiteralPath $Folder -File -ErrorAction SilentlyContinue |
        Where-Object { $ext -contains $_.Extension.ToLower() }
}

function Test-Device {
    $d = @(& "D:\Android\Sdk\platform-tools\adb.exe" devices | Select-String "\sdevice$")
    return ($d.Count -gt 0)
}

do {
    if (-not (Test-Device)) {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] Chua co thiet bi - mo may ao (tools\demo_app.ps1) hoac cam dien thoai."
        if ($Once) { exit 1 }
        Start-Sleep -Seconds 5
        continue
    }
    foreach ($f in Get-Images) {
        $key = "$($f.FullName)|$($f.LastWriteTimeUtc.Ticks)|$($f.Length)"
        if ($pushed.ContainsKey($key)) { continue }
        # Chờ chép xong: dung lượng phải giữ nguyên qua 2 lần quét (tránh đẩy file đang chép dở).
        if (-not $Once -and $lastSize[$f.FullName] -ne $f.Length) {
            $lastSize[$f.FullName] = $f.Length
            continue
        }
        if ($Serial) { & $push -Path $f.FullName -Serial $Serial | Out-Null } else { & $push -Path $f.FullName | Out-Null }
        if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
            $pushed[$key] = $true
            Write-Host "[$(Get-Date -Format HH:mm:ss)] Da day: $($f.Name)"
        } else {
            Write-Host "[$(Get-Date -Format HH:mm:ss)] Loi khi day: $($f.Name) (se thu lai)"
        }
    }
    if (-not $Once) { Start-Sleep -Seconds $IntervalSeconds }
} while (-not $Once)

Write-Host "Xong. Trong app: Quet thuoc -> Chon anh -> album 'MedAssist'."
