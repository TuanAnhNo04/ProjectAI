# CHUYỂN SANG DỮ LIỆU THẬT

> Hiện tại MedAssist chạy trên **10 thuốc mẫu** (`data/medicines.dev.csv`, đánh dấu `is_authoritative=false`)
> và tài khoản giả. Tài liệu này là quy trình chuyển sang dữ liệu thật **do bạn thực hiện** — theo
> `AGENTS.md`, AI/công cụ trợ lý không đọc CSV, ảnh hay đơn thuốc thật; các công cụ dưới đây chỉ in số liệu.

## Đã làm thử (28/10/2026) với bộ `G2_Final`

Từ thư mục Drive `G2_Final` (nhóm G2), đã tải `FINAL REPORT (1).pdf` và `processed_data.zip` về
`server\medassist-core\data\real\raw\` (thư mục bị git chặn). Trong zip có 2 CSV:

| File | Cột | Dòng | Đã dùng? |
|---|---|---|---|
| `processed_tdung _chinh_phu.csv` | `Medicine_Name, Composition, Uses, Side_effects, Manufacturer, Excellent/Average/Poor Review %` | 11.825 (11.501 sau khi loại 324 dòng trùng) | ✅ đã nạp vào `data\real\medassist.real.db`, nguồn ghi "nguồn gốc chưa xác minh", **không** đánh dấu lâm sàng |
| `processed_phan_loai.csv` | `Name, Category, Dosage Form, Strength, Manufacturer, Indication, Classification` | 50.000 | ⛔ **KHÔNG nạp** — dữ liệu sinh ngẫu nhiên (xem dưới) |

**Vì sao không nạp `processed_phan_loai.csv`:** chỉ có 64 tên thuốc nhưng **mỗi** tên đều ghép với đủ 8 chỉ định, 8 nhóm, 8 dạng bào chế, 2 phân loại và 20 nhà sản xuất;
giá trị phổ biến nhất chiếm đúng ≈ 1/8 số dòng (phân bố đều). Đó là các cột được gán độc lập ngẫu nhiên, không phải thông tin thuốc.
Nạp vào sẽ khiến app khẳng định sai (một thuốc "chỉ định" cho mọi bệnh) kèm trích dẫn nguồn, vi phạm `AGENTS.md` (không bịa thông tin y khoa).
Quyết định của chủ dự án (28/10/2026): giữ file ở `data\real\raw\`, không đưa vào CSDL.
Muốn dùng để thử tải/hiệu năng thì nạp vào CSDL tạm riêng, không dùng cho câu trả lời trong app.

⚠️ **Chưa xác minh nguồn gốc/giấy phép.** `FINAL REPORT (1).pdf` chưa đọc được (máy chưa có thư viện đọc PDF).
Tên cột và số dòng giống một bộ dữ liệu thuốc công khai phổ biến, rất có thể là thuốc **thị trường Ấn Độ** (tìm "amoxicillin" không ra,
có thể do dữ liệu viết "Amoxycillin") — nếu đúng, nó **không phù hợp để khuyên người dùng Việt Nam** về thuốc đang lưu hành ở Việt Nam.
Dùng cho học thuật/demo thì ổn; dùng cho người dùng thật thì cần nguồn Việt Nam đã kiểm duyệt.

Chạy server với dữ liệu này: `$env:MEDASSIST_SECRET_KEY = "<chuỗi bất kỳ >= 32 ký tự>"; .\tools\demo_server.ps1 -Real`
(CSDL thật **chưa có tài khoản** — hãy đăng ký tài khoản mới trong app.)

**Lỗi quy mô đã sửa nhờ chạy trên dữ liệu này:** `/agent/ask` và công cụ `explain_for_patient` chỉ nhìn **100 thuốc đầu theo tên**,
nên với 11.501 thuốc hầu hết câu hỏi bị từ chối. Đã thay bằng `candidate_medicines()` (chỉ mục từ khoá, tự làm mới khi có tiến trình khác nạp thêm).

## 0. "Dữ liệu thật" gồm ba thứ khác nhau

| Loại | Ví dụ | Trạng thái |
|---|---|---|
| **Danh mục thuốc** (nội dung y khoa) | tên, thành phần, công dụng, tác dụng phụ | ✅ Có công cụ nạp (mục 1–4) |
| **Dữ liệu người dùng** | tài khoản, lịch sử, hồ sơ sức khoẻ, ảnh hộp thuốc | Chỉ phát sinh khi có người dùng thật; xem mục 6 |
| **Hạ tầng thật** | server HTTPS, khoá bí mật, CSDL mã hoá | Chưa có; xem mục 5 |

## 1. Chọn nguồn danh mục thuốc

Nguồn phải **hợp pháp và trỏ về được một nơi cụ thể** (app luôn trích dẫn nguồn). Kiểm tra trước khi nạp:

- [ ] Có quyền sử dụng (giấy phép/điều khoản) — đặc biệt nếu app được phát hành rộng.
- [ ] Phù hợp người dùng Việt Nam (thuốc lưu hành tại Việt Nam; tên và hoạt chất tiếng Việt hoặc quốc tế).
- [ ] Có ngày cập nhật; thuốc thu hồi/hết hiệu lực được loại ra.
- [ ] Có người am hiểu (dược sĩ) xem qua mẫu dữ liệu.

Ví dụ nguồn để bạn cân nhắc: cơ sở dữ liệu thuốc đã đăng ký của cơ quan quản lý dược, hoặc tờ hướng dẫn sử dụng chính thức do nhà sản xuất công bố. (Tôi không xác nhận nguồn cụ thể nào đủ điều kiện — đó là quyết định của bạn.)

## 2. Chuẩn bị file CSV

UTF-8 (Excel: "CSV UTF-8"), có dòng tiêu đề. Cột mặc định:

```
Medicine_Name, Composition, Uses, Side_effects, Manufacturer
```

Chỉ `Medicine_Name` là bắt buộc. Cột tên khác thì ánh xạ bằng `-Col` (mục 3). Ô để trống sẽ hiện là
"không có thông tin" — **không** được điền bù.

⚠️ Đặt file ở nơi **ngoài repo** hoặc trong `server\medassist-core\data\real\` (đã bị `.gitignore` chặn). Đừng dán nội dung vào chat.

## 3. Nạp vào CSDL (chạy trên máy của bạn)

```powershell
# (a) CHẠY THỬ — không ghi gì, chỉ in số liệu (số dòng, trùng, thiếu tên, ô quá dài)
.\tools\import_real_data.ps1 -Csv "D:\du-lieu\thuoc.csv" -Source "Tên nguồn, 2026-10"

# (b) Cột khác tên
.\tools\import_real_data.ps1 -Csv ... -Source ... -Col "name=Tên thuốc","composition=Hoạt chất","uses=Công dụng","side_effects=Tác dụng phụ","manufacturer=Nhà sản xuất"

# (c) GHI THẬT — nguồn tham khảo; gỡ 10 thuốc mẫu cho khỏi lẫn dữ liệu giả
.\tools\import_real_data.ps1 -Csv ... -Source ... -Apply -DropDevSample

# (d) Nạp lại cùng một nguồn (không nhân đôi)
.\tools\import_real_data.ps1 -Csv ... -Source ... -Apply -ReplaceSource
```

Công cụ tự **sao lưu** CSDL trước khi ghi (`medassist.real.db.bak-<thời gian>`); muốn quay lại thì chép bản sao lưu đè lên.
CSDL thật nằm ở `server\medassist-core\data\real\medassist.real.db`, tách hẳn CSDL demo.

### Nguồn lâm sàng (`-Authoritative`)

Chỉ dùng khi **một người có chuyên môn đã kiểm duyệt nguồn** — bắt buộc ghi tên:

```powershell
.\tools\import_real_data.ps1 -Csv ... -Source ... -Apply -Authoritative -ReviewedBy "DS Nguyễn Văn A"
```

Tên người kiểm duyệt được ghi vào nhãn nguồn của từng bản ghi. Khi dữ liệu là `authoritative`, app bỏ dòng cảnh báo
"dữ liệu mẫu" trên câu trả lời. **Đừng bật nếu chưa có ai chịu trách nhiệm nội dung.**

## 4. Chạy server với dữ liệu thật

```powershell
$env:MEDASSIST_SECRET_KEY = "<chuỗi-ngẫu-nhiên-từ-32-ký-tự>"   # KHOÁ THẬT của bạn, đừng dùng khoá demo
.\tools\demo_server.ps1 -Real            # thêm -Llm nếu dùng LLM cục bộ
```

`-Real` từ chối chạy nếu thiếu khoá thật hoặc chưa có CSDL thật, và **không** nạp thuốc mẫu. Server vẫn chỉ nghe `127.0.0.1`.
Kiểm tra sau khi nạp: hỏi vài thuốc quen thuộc trong app, xác nhận câu trả lời trích đúng nguồn bạn đã gắn.

## 5. Điều chưa mở — và vì sao

| Việc | Tình trạng |
|---|---|
| **Câu hỏi liều dùng / tương tác thuốc** | Vẫn bị chặn (`is_out_of_scope`, `has_interaction_data()`), **kể cả khi** nguồn là `authoritative`. Muốn mở phải có dược sĩ duyệt quy tắc và nguồn tương tác riêng, rồi sửa code có chủ đích — đừng bật chỉ bằng cờ. |
| **Server cho máy khác truy cập** | Phải đặt sau reverse proxy **HTTPS**; bản release của app bắt buộc HTTPS. Không mở cổng 8000 ra LAN/Internet trực tiếp. |
| **Mã hoá CSDL** | SQLite phía server và phía máy chưa mã hoá. Dùng mã hoá ổ đĩa hoặc chuyển Postgres có mã hoá trước khi chứa dữ liệu người dùng thật. |
| **Sao lưu, retention** | Đặt `MEDASSIST_HISTORY_RETENTION_DAYS`; lên lịch sao lưu CSDL. |
| **LLM đám mây** | Vẫn chưa làm (cần đồng ý trong UI + khử định danh + ghi PRIVACY). LLM cục bộ (Ollama) thì dùng được. |

## 6. Khi có người dùng thật

- Cập nhật `docs/PRIVACY.md` cho đúng thực tế (ai lưu gì, bao lâu) và kiểm tra màn đồng ý.
- Thử trên **điện thoại thật** (xem `docs/RELEASE.md`): thông báo nhắc, OCR, đăng nhập qua HTTPS.
- Ảnh hộp thuốc chỉ xử lý trên máy (ML Kit), **không** gửi lên server; đừng đổi điều này mà chưa đủ ba điều kiện ở `AGENTS.md`.
- Chưa có báo lỗi crash; nếu thêm, cần đồng ý của người dùng và che dữ liệu.

## 7. Kiểm tra nhanh trước khi báo "đã chạy dữ liệu thật"

- [ ] Nạp xong: số bản ghi khớp số dòng trong file (báo cáo cuối của công cụ).
- [ ] Không còn nhãn `dev-sample:` trong danh mục.
- [ ] Nguồn trong câu trả lời đúng tên bạn đặt; có người kiểm duyệt nếu bật `-Authoritative`.
- [ ] Server chạy bằng khoá thật; không dùng tài khoản/khoá demo.
- [ ] Dữ liệu thật không nằm trong git (`data/real/`, `*.real.csv`, `*.db.bak-*` đã bị chặn).
