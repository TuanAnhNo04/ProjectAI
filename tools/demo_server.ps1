# Chạy server DEMO trên localhost với khoá GIẢ (không dùng dữ liệu thật).
# DB nằm ở server\medassist-core\data\medassist.dev.db (đã được .gitignore chặn *.db),
# nên tài khoản demo không mất khi khởi động lại; thuốc mẫu được nạp nếu bảng còn rỗng.
#
# Mặc định dùng mock (ổn định cho buổi demo). Thêm -Llm để bật LLM thật chạy TRÊN MÁY
# qua Ollama (xem docs\LLM_LOCAL.md; cần đã cài Ollama + `ollama pull qwen2.5:3b`).
#
# -Real: chạy trên CSDL dữ liệu THẬT (data\real\medassist.real.db, nạp bằng import_real_data.ps1).
#   Bắt buộc bạn tự đặt MEDASSIST_SECRET_KEY thật (>= 32 ký tự) trong môi trường — script
#   KHÔNG dùng khoá demo, KHÔNG nạp thuốc mẫu. Vẫn chỉ nghe 127.0.0.1; muốn cho máy khác
#   truy cập phải đặt sau reverse proxy HTTPS (xem docs\REAL_DATA.md).
#
# -Claude: dùng Claude API (đám mây). Thứ tự tìm khoá: (1) biến môi trường ANTHROPIC_API_KEY,
#   (2) file đã lưu bằng -SaveKey, (3) hỏi ẩn ký tự (không hiện trên màn hình).
# -SaveKey: (dùng kèm -Claude) hỏi khoá MỘT lần rồi lưu MÃ HOÁ vào
#   %APPDATA%\MedAssist\anthropic_key.xml (Windows DPAPI: chỉ đúng tài khoản Windows này, trên máy
#   này, giải mã được). File nằm NGOÀI thư mục project nên không bao giờ bị `git push`.
# -ForgetKey: xoá file khoá đã lưu rồi thoát.
# -News: bật tin thật cho mục Tin tức từ ĐÚNG MỘT nguồn chủ dự án đã duyệt (30/10/2026):
#   FDA Recalls/Safety (public domain). Server chỉ GET đọc RSS, không gửi dữ liệu gì đi.
#   Không có -News -> bài mẫu, không gọi mạng. Xem docs\NEWS.md.
param([switch]$Llm, [switch]$Real, [switch]$Claude, [switch]$SaveKey, [switch]$ForgetKey, [switch]$News)

$keyFile = Join-Path $env:APPDATA "MedAssist\anthropic_key.xml"
if ($ForgetKey) {
    Remove-Item $keyFile -ErrorAction SilentlyContinue
    Write-Host "Da xoa khoa da luu (neu co)."
    exit 0
}

if ($Real) {
    if (-not $env:MEDASSIST_SECRET_KEY -or $env:MEDASSIST_SECRET_KEY.Length -lt 32 -or $env:MEDASSIST_SECRET_KEY -like "*demo-fake*") {
        Write-Error "Che do -Real can bien moi truong MEDASSIST_SECRET_KEY that (>= 32 ky tu, khong phai khoa demo)."
        exit 1
    }
    $env:MEDASSIST_DB = "$PSScriptRoot\..\server\medassist-core\data\real\medassist.real.db"
    if (-not (Test-Path $env:MEDASSIST_DB)) {
        Write-Error "Chua co CSDL that. Nap du lieu truoc: .\tools\import_real_data.ps1 -Csv ... -Source ... -Apply"
        exit 1
    }
    Write-Host "CHE DO DU LIEU THAT: $env:MEDASSIST_DB"
} else {
    $env:MEDASSIST_SECRET_KEY = "demo-fake-key-demo-fake-key-demo-fake-key-0123"
    $env:MEDASSIST_DB = "$PSScriptRoot\..\server\medassist-core\data\medassist.dev.db"
}
if ($Claude) {
    if (-not $env:ANTHROPIC_API_KEY -and -not $SaveKey -and (Test-Path $keyFile)) {
        try {
            $secure = Import-Clixml $keyFile
            $env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new('', $secure).Password
            Write-Host "Da nap khoa tu file ma hoa $keyFile"
        } catch {
            Write-Host "Khong doc duoc file khoa da luu (khac tai khoan/may?). Se hoi lai."
        }
    }
    if (-not $env:ANTHROPIC_API_KEY) {
        # -AsSecureString chạy được cả PowerShell 5.1 lẫn 7 (khác -MaskInput chỉ có ở 7.1+).
        $secure = Read-Host "Dan khoa Anthropic (an ky tu)" -AsSecureString
        $env:ANTHROPIC_API_KEY = [System.Net.NetworkCredential]::new('', $secure).Password
        if ($SaveKey -and $env:ANTHROPIC_API_KEY) {
            New-Item -ItemType Directory -Force (Split-Path $keyFile) | Out-Null
            $secure | Export-Clixml $keyFile
            Write-Host "Da luu khoa (ma hoa) vao $keyFile. Lan sau chi can: .\tools\demo_server.ps1 -Claude"
        }
    }
    if (-not $env:ANTHROPIC_API_KEY) { Write-Error "Chua co khoa ANTHROPIC_API_KEY."; exit 1 }
    $env:MEDASSIST_LLM = "claude"
    Write-Host "LLM: claude (khoa doc tu bien moi truong, khong in ra)"
} elseif ($Llm) {
    $env:MEDASSIST_LLM = "ollama"
    if (-not $env:MEDASSIST_OLLAMA_MODEL) { $env:MEDASSIST_OLLAMA_MODEL = "qwen2.5:3b" }
    Write-Host "LLM: ollama ($env:MEDASSIST_OLLAMA_MODEL) tren 127.0.0.1"
    # Nạp sẵn model vào bộ nhớ (lần đầu mất ~1 phút) để câu hỏi đầu tiên khi demo không bị chậm.
    # Chỉ gọi 127.0.0.1; lỗi (Ollama chưa chạy) thì bỏ qua, server vẫn chạy và dùng câu trả lời trích nguyên văn.
    try {
        $warm = @{ model = $env:MEDASSIST_OLLAMA_MODEL; keep_alive = "30m" } | ConvertTo-Json
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method Post -Body $warm `
            -ContentType "application/json" -TimeoutSec 180 | Out-Null
        Write-Host "Da nap model vao bo nho."
    } catch {
        Write-Host "Canh bao: chua goi duoc Ollama (da chay chua?). Se dung cau tra loi trich nguyen van."
    }
} else {
    Remove-Item Env:\MEDASSIST_LLM -ErrorAction SilentlyContinue
    Write-Host "LLM: mock (mac dinh)"
}
if ($News) {
    $env:MEDASSIST_NEWS_FEEDS = "FDA Recalls|https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/recalls/rss.xml"
    Write-Host "Tin tuc: FDA Recalls (RSS that)"
} else {
    Remove-Item Env:\MEDASSIST_NEWS_FEEDS -ErrorAction SilentlyContinue
}
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
Set-Location "$PSScriptRoot\..\server\medassist-core"
if (-not $Real) { & $py -m app.seed }   # dữ liệu thật: không nạp thuốc mẫu
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
