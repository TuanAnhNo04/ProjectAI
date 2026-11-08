# Postman cho MedAssist (chạy trên máy bạn)

Ba file trong thư mục này (nhập bằng **Import**):

| File | Để làm gì |
|---|---|
| `MedAssist-Local.postman_collection.json` | **33 request** (mọi endpoint của backend + tình huống lỗi + chat triệu chứng/khẩn cấp, lời khuyên, tin tức), mỗi request có test và ví dụ phản hồi (tạo lại 30/10/2026) |
| `MedAssist-Local.postman_environment.json` | biến môi trường: `baseUrl`, `username`, `email`, `password`, `sessionId` |
| `search-terms.json` | dữ liệu cho **Collection Runner** (tìm thuốc hàng loạt) |

> Chỉ dùng dữ liệu GIẢ. Tài khoản demo: `demo` / `Demo@12345678` (không phải tài khoản thật).

## Cách dùng

1. Chạy server: `.\tools\demo_server.ps1` (khoá giả, chỉ nghe `127.0.0.1:8000`).
2. Postman → **Import** → chọn cả 2 file `.postman_collection.json` và `.postman_environment.json`.
3. Góc trên bên phải chọn environment **MedAssist Local**.
4. Chạy lần lượt: **1. Health** → **2. Tài khoản › Đăng nhập** (hoặc Đăng ký nếu chưa có tài khoản). Token tự lưu vào biến `accessToken`/`refreshToken`.
5. Các thư mục 3 → 8 chạy theo bất kỳ thứ tự nào sau khi đăng nhập (**8. Tin tức** không cần đăng nhập; chạy server với `-News` để có tin thật FDA). Mở tab **Examples** của request để xem phản hồi mẫu.
6. Chạy cả bộ: nút **Run collection**. Với thư mục **7. Chạy hàng loạt**, chọn **Data → `search-terms.json`**.
7. Thư mục **9. Dọn dẹp** (đăng xuất, xoá tài khoản) luôn để **cuối cùng**.

## Lưu ý

- Chạy với **dữ liệu thật** (`.\tools\demo_server.ps1 -Real`): CSDL thật chưa có tài khoản → đổi `username`/`password` trong environment rồi chạy **Đăng ký**; và thay `search-terms.json` bằng tên thuốc của bạn (file mẫu chỉ có tên thuốc mẫu giả).
- Chat (`/v1/agent/chat`) trả luồng SSE; xem dần ở tab Response.
- `Xoá một bản ghi lịch sử` cần `historyId` do request **Danh sách lịch sử** lưu lại; `Xoá phiên hội thoại` cần chạy **Chat** trước.
- Hỏi liều dùng/tương tác thuốc bị từ chối (`out_of_scope`) — đó là thiết kế an toàn, không phải lỗi.
- Không dán token hay email thật vào collection.

## Tạo lại bộ file khi API thay đổi

```powershell
.\tools\demo_server.ps1                          # cửa sổ 1
python tools\postman\build_collection.py        # cửa sổ 2 — chạy thật từng request để lấy ví dụ (tài khoản tạm, tự xoá)
python tools\postman\build_collection.py --no-examples   # không cần server
```

Script báo lỗi nếu `openapi.json` có endpoint mà collection chưa có request (không bị thiếu im lặng).
