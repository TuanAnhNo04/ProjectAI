# Mock server (Flask) — cầu tạm cho dev

> ⚠️ **Đây KHÔNG phải backend thật.** Đây là server Flask tối giản dựng tại chỗ để
> Day 8 test tay được 5 luồng và Day 9 nối UI. Day 15 sẽ thay bằng **FastAPI Core**
> thật (`medassist-core`, `/v1/...`) — lúc đó xoá thư mục này.

## Vì sao cần

Repo app **không chứa** mã nguồn server Flask cũ. Đã kiểm tra: cả ổ đĩa không có
file `.py` nào (chỉ có tài liệu *nhắc tới* Flask). Trong khi đó Day 8 cần:

- login sai → hiện lỗi, không crash
- login đúng → vào Home
- upload ảnh → hiện thông tin thuốc
- search "panadol" → ra kết quả
- xoá 1 mục history → biến mất

→ Không có server thì 5 việc trên không test được. File này lấp đúng chỗ đó, và
trả **đúng tên trường** mà app đang parse (xem `lib/data/models/*.dart`).

## Cài & chạy

Python 3.12 và Flask đã có sẵn trên máy dev, nên thường **không cần cài gì**:

```powershell
# chỉ chạy khi máy khác chưa có
python -m pip install -r tools/mock_server/requirements.txt

# chạy server (Ctrl+C để dừng)
python tools/mock_server/app.py
```

Kiểm tra server sống:

```powershell
curl.exe -s http://127.0.0.1:5000/health
# {"history":0,"medicines":10,"service":"mock-flask","status":"ok"}
```

Tài khoản thử:

- **`demo` / `demo123`** — đăng nhập được, trả token.
- **`legacy` / bất kỳ mật khẩu nào** — luôn thất bại nhưng bằng
  `HTTP 200 + {"success": false}` (kiểu phản hồi của Flask bản cũ). Có tài khoản
  này để kiểm tra được CẢ HAI nhánh thất bại mà `RemoteAuthRepository` phải xử
  lý, không chỉ nhánh 401.

## Trỏ app vào server

| Chạy app ở đâu | `API_BASE_URL` | Ghi chú |
|---|---|---|
| Android **emulator** | `http://10.0.2.2:5000` | mặc định của `AppConfig`, đã cho phép cleartext |
| **Điện thoại thật** (cùng WiFi) | `http://<IP-LAN>:5000` | phải thêm IP vào `network_security_config.xml` (xem dưới) |
| Windows desktop | `http://127.0.0.1:5000` | không cần cấu hình gì thêm |

```powershell
# emulator
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:5000

# điện thoại thật: tìm IP LAN
ipconfig | Select-String "IPv4"
flutter run --dart-define=API_BASE_URL=http://192.168.1.10:5000
```

Vì sao điện thoại thật cần thêm bước: từ Android 9, HTTP (cleartext) **bị chặn
mặc định**; dự án chỉ mở cho `10.0.2.2`, `localhost`, `127.0.0.1` trong
`android/app/src/debug/res/xml/network_security_config.xml`. Dùng IP LAN thì mở
file đó và thêm 1 dòng `<domain includeSubdomains="true">192.168.1.10</domain>`.
(Windows Firewall cũng sẽ hỏi cho phép Python — chọn *Allow*.)

## Endpoint

| Method | Đường dẫn | Trả về |
|---|---|---|
| POST | `/process-image` | multipart field `image` → `{"medicine_info":[...]}` |
| GET | `/search-medicine?name=` | `{"results":[...], "search_results":[...], "query", "total"}` |
| GET | `/history` | `{"history":[{id, filepath, search_results, created_at}]}` |
| DELETE | `/history/<id>` | `{"success":true}` · **404** nếu không có |
| POST | `/login` | `{"success":true,"token","user"}` · **401** nếu sai · riêng `legacy` trả **200** + `success:false` |
| POST | `/register` | `{"success":true,"message"}` |
| POST | `/logout` | `{"success":true}` |
| GET | `/health` | `{"status":"ok",...}` |

Dữ liệu thuốc nằm ở `medicines.csv` — sửa file này rồi chạy lại server là có dữ
liệu mới, không cần đụng code. Dữ liệu mẫu **cố ý có cả tên tiếng Việt có dấu**
(`Thuốc ho Bảo Thanh`, `Dầu gió xanh`) để phần tìm kiếm không dấu được kiểm tra
thật: gõ `thuoc ho` hoặc `dau gio` (không dấu) vẫn ra đúng thuốc.

Tìm kiếm khớp **tên + hoạt chất + nhà sản xuất**, không phân biệt dấu và hoa
thường; nhiều từ thì phải khớp hết (AND), nên `panadol`, `para 500`,
`thuoc ho`, `dau gio` đều ra kết quả.

Ví dụ đã chạy thật trên máy dev:

| `?name=` | Kết quả |
|---|---|
| `panadol` | 1 — Panadol 500mg |
| `para 500` | 4 — Panadol 500mg, Hapacol 500, Para 500, Paracetamol Stada 500mg |
| `thuoc ho` | 1 — Thuốc ho Bảo Thanh (gõ **không dấu**, khớp tên **có dấu**) |
| `dau gio` | 1 — Dầu gió xanh |
| `khong co thuoc nay` | 0 — màn "không tìm thấy" |

### Ảnh upload không có OCR

Mock không đọc được chữ trong ảnh, nên quy ước theo **tên file** để test được cả
2 nhánh UI:

- tên file chứa `empty` hoặc `khong-thay` → `{"medicine_info": []}` (màn "không
  tìm thấy thuốc")
- tên file khác → trả về `Panadol 500mg`

Muốn thử nhánh "không tìm thấy" bằng curl thì đổi tên file trước:

```powershell
Copy-Item pubspec.yaml "$env:TEMP\empty_scan.jpg"
curl.exe -s -X POST http://127.0.0.1:5000/process-image -F "image=@$env:TEMP\empty_scan.jpg"
# {"medicine_info": [], ...}
```

## Kiểm tra 5 luồng — tự động (khuyến nghị)

`tools/mock_server/live_check_test.dart` chạy **adapter thật của app**
(`RemoteAuthRepository`, `RemoteAnalyzer`, `RemoteMedicineRepository`) đối chiếu
với server này — 15 test, phủ cả 5 luồng Day 8:

```powershell
# cửa sổ 1 — server
python tools/mock_server/app.py

# cửa sổ 2 — gọi ĐÍCH DANH đường dẫn (file không nằm trong test/ nên
# `flutter test` một mình sẽ KHÔNG chạy nó)
flutter test tools/mock_server/live_check_test.dart --dart-define=API_BASE_URL=http://127.0.0.1:5000
# -> All tests passed! (15)
```

**Vì sao không đặt trong `test/`:** file này cần server đang chạy, còn 200 test
trong `test/` phải chạy offline (CI không có server). Nằm ngoài `test/` nên nó
không bao giờ tự chạy trong `flutter test` — đã kiểm tra: `flutter test` vẫn đúng
**200 test**.

**Vì sao cần nó, khi đã có fixture JSON:** fixture ở `test/data/models/fixtures/`
là JSON **chép tay**, nên nó không thể phát hiện server trả *sai tên trường*, *sai
mã HTTP* hay *sai kiểu dữ liệu*. Ở đây thì có — đó là lý do nó tồn tại.

## Kiểm tra 5 luồng — bằng tay (curl)

Dùng khi muốn soi từng phản hồi thô:

```powershell
# 1) login sai -> 401
curl.exe -s -i -X POST http://127.0.0.1:5000/login -H "Content-Type: application/json" -d "{\"username\":\"demo\",\"password\":\"sai\"}"

# 2) login đúng -> 200 + token
curl.exe -s -X POST http://127.0.0.1:5000/login -H "Content-Type: application/json" -d "{\"username\":\"demo\",\"password\":\"demo123\"}"

# 3) search (không dấu vẫn ra)
curl.exe -s "http://127.0.0.1:5000/search-medicine?name=panadol"
curl.exe -s "http://127.0.0.1:5000/search-medicine?name=para%20500"

# 4) upload ảnh -> lưu history, rồi xem history
curl.exe -s -X POST http://127.0.0.1:5000/process-image -F "image=@pubspec.yaml"
curl.exe -s http://127.0.0.1:5000/history

# 5) xoá history vừa tạo -> 200; xoá lại -> 404 (id lấy từ kết quả bước 4)
curl.exe -s -X DELETE http://127.0.0.1:5000/history/1
curl.exe -s -i -X DELETE http://127.0.0.1:5000/history/1
```

Dữ liệu chỉ nằm trong RAM: **tắt server là mất history** (đúng ý đồ — Day 15 mới
có Postgres). Ảnh upload được lưu tạm ở `tools/mock_server/uploads/` (đã gitignore).