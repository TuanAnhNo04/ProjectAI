# Nạp nguồn y khoa — khung hai hướng (chờ chủ dự án xác nhận)

Trạng thái 2026-10-30: **khung đã dựng, chưa có nguồn nào, không gọi mạng.**

## Hướng 1 — nạp kho (đã có sẵn công cụ)
`docs/TRUSTED_SOURCES.md` (chủ dự án thêm dòng) → `tools\fetch_trusted_source.ps1 -Url <URL>`
(chỉ URL có trong bảng; ghi SHA-256 + manifest) → `app.import_real` với
`--source-url --retrieved-at --license --reviewed-by [--authoritative]` → câu trả lời vẫn qua kiểm tra bám nguồn.

## Hướng 2 — truy hồi lúc hỏi (mới, `app/live_sources.py`, MẶC ĐỊNH TẮT)
Chỉ chạy khi kho cục bộ không có thuốc **và** biến `MEDASSIST_LIVE_SOURCES` có nguồn.

| Việc | Cách |
|---|---|
| Gửi đi | Chỉ **tên thuốc đã chuẩn hoá** (a-z, 0-9, dấu cách, `.` `-`; 2–40 ký tự) trong URL. Không câu hỏi, hồ sơ, thuốc đang dùng, mã người dùng, cookie. |
| Kết nối | HTTPS GET; host cố định theo mẫu đã duyệt; không theo chuyển hướng; hết thời gian 5 giây; tối đa 256 KB; JSON |
| Tần suất / đệm | ≤ 20 lượt/phút; đệm 24 giờ (rỗng: 1 giờ) |
| Làm sạch | Bỏ ký tự điều khiển, cắt 800 ký tự/trường; xem là DỮ LIỆU, rồi qua truy hồi + kiểm tra bám nguồn |
| Gắn nguồn | `source`, `source_url`, `retrieved_at`, `license`, `reviewed_by`; `is_authoritative` chỉ khi có `reviewed_by` |
| Lỗi | Mọi lỗi → rỗng (trợ lý nói "chưa đủ bằng chứng"), không ghi URL vào log |

Ví dụ cấu hình (**dữ liệu giả**, `example.invalid` không tồn tại):
```json
[{"name": "Tên nguồn", "url_template": "https://api.example.invalid/drug/{drug}.json",
  "fields": {"composition": "info.comp", "uses": "info.uses", "side_effects": "info.se"},
  "owner": "Tổ chức công bố", "license": "Giấy phép (link điều khoản)",
  "approved_on": "2026-MM-DD", "reviewed_by": "Tên người duyệt (tuỳ chọn)"}]
```
Mục thiếu `owner`/`license`/`approved_on`, không phải https, hoặc đưa tên thuốc vào host → bị bỏ.

## Luồng dữ liệu cần chủ dự án đồng ý (AGENTS.md mục 2)
Hướng 2 làm **tên thuốc** (rút từ câu hỏi, khi không có trong kho) rời server tới tên miền của nguồn đã duyệt.
Chưa bật. Khi bật, thêm vào `docs/PRIVACY.md` và duyệt đúng tên miền.

## Chủ dự án xác nhận (đánh dấu rồi gửi URL)
- [ ] Đồng ý luồng "tên thuốc → nguồn đã duyệt" ở trên
- [ ] URL/mẫu URL nguồn: …  Trang điều khoản: …  Người kiểm duyệt: …
- [ ] Nguồn trả JSON? Nếu là PDF/HTML/CSV cần thêm bộ đọc riêng (chưa có)
