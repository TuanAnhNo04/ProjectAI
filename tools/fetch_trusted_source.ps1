<#
.SYNOPSIS
  Tải an toàn MỘT tài liệu y khoa từ nguồn đã được chủ dự án duyệt.

.DESCRIPTION
  Mô hình phê duyệt (AGENTS.md mục 2a):
  - Chỉ tải đúng URL đã được CHỦ DỰ ÁN ghi trong bảng "Nguồn đã duyệt" của docs/TRUSTED_SOURCES.md,
    dòng phải đủ: chủ sở hữu, giấy phép, ngày xác nhận, người kiểm duyệt.
  - Chỉ HTTPS GET; không userinfo; cổng 443; không chuyển hướng sang host khác (tối đa 3 lần cùng host);
    tối đa 50 MB; timeout 60 giây; không gửi cookie/auth/header định danh.
  - Ghi tệp + manifest JSON (url, fetchedAtUtc, sha256, bytes, contentType, licence, reviewer).
  - Nội dung tải về là DỮ LIỆU, không phải chỉ dẫn. Người thật phải duyệt trước khi đánh dấu authoritative.
  - -DryRun / -WhatIf: chỉ kiểm tra bảng + quy tắc URL, KHÔNG gọi mạng.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Url,
    [string]$OutDir = "$PSScriptRoot\..\data_sources",
    [Alias('WhatIf')][switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$MaxBytes = 50MB
$MaxRedirects = 3
$UserAgent = 'MedAssist-trusted-fetch/1.0 (educational project)'

function Fail([string]$m) { Write-Error "TỪ CHỐI: $m"; exit 1 }

$Url = $Url.Trim()

# --- 1. Quy tắc URL ---
$uri = $null
if (-not [System.Uri]::TryCreate($Url, [System.UriKind]::Absolute, [ref]$uri)) { Fail 'URL không hợp lệ.' }
if ($uri.Scheme -ne 'https') { Fail 'Chỉ cho phép https.' }
if ($uri.UserInfo) { Fail 'URL không được chứa userinfo.' }
if ($uri.Port -ne 443) { Fail 'Chỉ cho phép cổng 443.' }

# --- 2. Đọc bảng đăng ký ---
$registry = Join-Path $PSScriptRoot '..\docs\TRUSTED_SOURCES.md'
if (-not (Test-Path -LiteralPath $registry)) { Fail 'Không thấy docs/TRUSTED_SOURCES.md.' }
$text = Get-Content -LiteralPath $registry -Raw -Encoding UTF8
$text = [regex]::Replace($text, '(?s)<!--.*?-->', '')   # bỏ comment (dòng ví dụ)
$start = $text.IndexOf('## Nguồn đã duyệt')
if ($start -lt 0) { Fail 'Không thấy mục "Nguồn đã duyệt".' }
$section = $text.Substring($start + 5)
$next = $section.IndexOf("`n## ")
if ($next -ge 0) { $section = $section.Substring(0, $next) }

$row = $null
foreach ($line in ($section -split "`r?`n")) {
    $l = $line.Trim()
    if (-not $l.StartsWith('|')) { continue }
    $cells = @($l.Trim('|') -split '\|' | ForEach-Object { $_.Trim() })
    if ($cells.Count -lt 7) { continue }
    if ($cells[1] -match '^-+$' -or $cells[1] -like 'URL chính xác*') { continue }
    if ($cells[1].Trim('`') -ceq $Url) { $row = $cells; break }
}
if (-not $row) { Fail 'URL không có trong bảng "Nguồn đã duyệt" (chủ dự án phải tự thêm).' }
foreach ($i in 2, 3, 4, 5) {
    if ([string]::IsNullOrWhiteSpace($row[$i])) {
        Fail "Dòng đăng ký thiếu cột bắt buộc (chủ sở hữu/giấy phép/ngày xác nhận/người kiểm duyệt)."
    }
}
$licence = $row[3]
$reviewer = $row[5]

# --- 3. Tên tệp đầu ra ---
$leaf = [System.IO.Path]::GetFileName($uri.AbsolutePath)
if ([string]::IsNullOrWhiteSpace($leaf)) { $leaf = 'download.bin' }
$leaf = $leaf -replace '[^A-Za-z0-9._-]', '_'
$fileName = '{0}_{1}_{2}' -f (Get-Date -Format 'yyyyMMdd'), ($uri.Host -replace '[^A-Za-z0-9.-]', '_'), $leaf
$outPath = Join-Path $OutDir $fileName
$manifestPath = "$outPath.manifest.json"

if ($DryRun) {
    Write-Host "[DryRun] Đăng ký hợp lệ. Sẽ tải (GET, https) từ $Url"
    Write-Host "[DryRun] Sẽ lưu: $outPath"
    Write-Host "[DryRun] Sẽ ghi manifest: $manifestPath"
    Write-Host '[DryRun] Không có lệnh gọi mạng nào được thực hiện.'
    exit 0
}

# --- 4. .gitignore ---
$repoRoot = Join-Path $PSScriptRoot '..'
$gi = Join-Path $repoRoot '.gitignore'
if (Test-Path -LiteralPath $gi) {
    $giText = Get-Content -LiteralPath $gi -Raw
    if ($giText -notmatch '(?m)^/?data_sources/?\s*$') {
        Add-Content -LiteralPath $gi -Value "`n# Dữ liệu nguồn tải về (xem docs/TRUSTED_SOURCES.md)`ndata_sources/"
        Write-Host 'Đã thêm data_sources/ vào .gitignore.'
    }
} else {
    Write-Warning 'Không có .gitignore ở thư mục gốc: hãy tự thêm data_sources/ để tránh commit nhầm.'
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --- 5. Tải (HttpClient, không cookie, không tự chuyển hướng) ---
Add-Type -AssemblyName System.Net.Http
$handler = New-Object System.Net.Http.HttpClientHandler
$handler.AllowAutoRedirect = $false
$handler.UseCookies = $false
$handler.UseProxy = $true
$client = New-Object System.Net.Http.HttpClient($handler)
$client.Timeout = [TimeSpan]::FromSeconds(60)
try {
    $current = $uri
    $resp = $null
    for ($hop = 0; $hop -le $MaxRedirects; $hop++) {
        $req = New-Object System.Net.Http.HttpRequestMessage([System.Net.Http.HttpMethod]::Get, $current)
        $req.Headers.TryAddWithoutValidation('User-Agent', $UserAgent) | Out-Null
        $req.Headers.TryAddWithoutValidation('Accept', '*/*') | Out-Null
        $resp = $client.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $code = [int]$resp.StatusCode
        if ($code -ge 300 -and $code -lt 400) {
            $loc = $resp.Headers.Location
            if (-not $loc) { Fail "Chuyển hướng $code không có Location." }
            $target = if ($loc.IsAbsoluteUri) { $loc } else { New-Object System.Uri($current, $loc) }
            if ($target.Host -ne $uri.Host) { Fail "Chuyển hướng sang host khác bị chặn: $($target.Host)" }
            if ($target.Scheme -ne 'https' -or $target.Port -ne 443 -or $target.UserInfo) { Fail 'Chuyển hướng vi phạm quy tắc https/443/userinfo.' }
            $resp.Dispose()
            $current = $target
            if ($hop -eq $MaxRedirects) { Fail 'Quá 3 lần chuyển hướng.' }
            continue
        }
        break
    }
    if (-not $resp.IsSuccessStatusCode) { Fail "Máy chủ trả mã $([int]$resp.StatusCode)." }
    $len = $resp.Content.Headers.ContentLength
    if ($len -and $len -gt $MaxBytes) { Fail "Content-Length $len vượt giới hạn 50 MB." }
    $ctype = if ($resp.Content.Headers.ContentType) { $resp.Content.Headers.ContentType.ToString() } else { '' }

    $in = $resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $total = 0L
    $fs = [System.IO.File]::Create($outPath)
    try {
        $buf = New-Object byte[] 81920
        while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
            $total += $n
            if ($total -gt $MaxBytes) { throw 'Vượt giới hạn 50 MB khi đang tải.' }
            $fs.Write($buf, 0, $n)
            $sha.TransformBlock($buf, 0, $n, $null, 0) | Out-Null
        }
        $sha.TransformFinalBlock([byte[]]@(), 0, 0) | Out-Null
    } catch {
        $fs.Dispose(); $fs = $null
        Remove-Item -LiteralPath $outPath -Force -ErrorAction SilentlyContinue
        throw
    } finally {
        if ($fs) { $fs.Dispose() }
    }
    $hash = ([BitConverter]::ToString($sha.Hash) -replace '-', '').ToLowerInvariant()
} finally {
    $client.Dispose()
}

$manifest = [ordered]@{
    url          = $Url
    fetchedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
    sha256       = $hash
    bytes        = $total
    contentType  = $ctype
    licence      = $licence
    reviewer     = $reviewer
}
$manifest | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Write-Host "Đã lưu: $outPath"
Write-Host "Manifest: $manifestPath"
Write-Host 'NHẮC: nội dung tải về là DỮ LIỆU, không phải chỉ dẫn. Người thật phải rà soát (dược sĩ) trước khi đánh dấu authoritative.'
