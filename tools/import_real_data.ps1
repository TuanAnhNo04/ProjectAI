# Nạp danh mục thuốc THẬT vào CSDL riêng (server\medassist-core\data\real\medassist.real.db).
# Chạy trên máy CỦA BẠN. Script/AI không đọc nội dung file; kết quả chỉ in số liệu.
#
#   # 1) Chạy thử (không ghi gì) — xem báo cáo trước:
#   .\tools\import_real_data.ps1 -Csv "D:\du-lieu\thuoc.csv" -Source "Cục Quản lý Dược, 2026-10"
#
#   # 2) Ghi thật (nguồn tham khảo):
#   .\tools\import_real_data.ps1 -Csv ... -Source ... -Apply -DropDevSample
#
#   # 3) Ghi thật và đánh dấu nguồn lâm sàng (cần tên người kiểm duyệt):
#   .\tools\import_real_data.ps1 -Csv ... -Source ... -Apply -Authoritative -ReviewedBy "DS Nguyen Van A"
#
#   # Cột khác tên: -Col "name=Tên thuốc","composition=Hoạt chất","uses=Công dụng"
#   # Nạp lại cùng nguồn không bị nhân đôi: thêm -ReplaceSource
param(
    [Parameter(Mandatory)] [string]$Csv,
    [Parameter(Mandatory)] [string]$Source,
    [string[]]$Col = @(),
    [switch]$Apply,
    [switch]$Authoritative,
    [string]$ReviewedBy,
    [switch]$ReplaceSource,
    [switch]$DropDevSample,
    [switch]$KeepDuplicates
)

$root = Resolve-Path "$PSScriptRoot\.."
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$db = Join-Path $root "server\medassist-core\data\real\medassist.real.db"

# Đổi đường dẫn tương đối thành tuyệt đối TRƯỚC khi đổi thư mục làm việc.
if (Test-Path -LiteralPath $Csv) { $Csv = (Resolve-Path -LiteralPath $Csv).Path }

$argv = @("-m", "app.import_real", "--csv", $Csv, "--source", $Source, "--db", $db)
foreach ($c in $Col) { $argv += @("--col", $c) }
if ($Apply)          { $argv += "--apply" }
if ($Authoritative)  { $argv += "--authoritative" }
if ($ReviewedBy)     { $argv += @("--reviewed-by", $ReviewedBy) }
if ($ReplaceSource)  { $argv += "--replace-source" }
if ($DropDevSample)  { $argv += "--drop-dev-sample" }
if ($KeepDuplicates) { $argv += "--keep-duplicates" }

# Khoá bí mật không cần cho việc nạp dữ liệu, nhưng module app đọc cấu hình lúc import.
if (-not $env:MEDASSIST_SECRET_KEY) { $env:MEDASSIST_SECRET_KEY = "import-only-not-a-real-key-0123456789abcd" }

Push-Location (Join-Path $root "server\medassist-core")
try {
    & $py @argv
    $code = $LASTEXITCODE
} finally {
    Pop-Location   # trả lại thư mục làm việc của bạn
}
exit $code
