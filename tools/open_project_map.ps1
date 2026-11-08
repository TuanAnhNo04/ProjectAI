# Quét lại dự án và mở sơ đồ tương tác (docs\project-map.html) trong trình duyệt.
# Chạy hoàn toàn cục bộ, không gọi mạng. Chạy lại mỗi khi bạn sửa code để sơ đồ luôn cập nhật.
#
#   .\tools\open_project_map.ps1              # quét + mở
#   .\tools\open_project_map.ps1 -NoOpen      # chỉ quét
param([switch]$NoOpen)

$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$root = (Resolve-Path "$PSScriptRoot\..").Path

& $py -I (Join-Path $root "tools\project_map\build_map.py")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Quet that bai (xem thong bao o tren). Thuong la do luong mo ta (FLOWS trong build_map.py) khong con khop voi ma."
    exit $LASTEXITCODE
}
if (-not $NoOpen) { Start-Process (Join-Path $root "docs\project-map.html") }
