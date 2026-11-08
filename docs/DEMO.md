# KỊCH BẢN DEMO (~7 phút)

Chạy: `tools\demo_server.ps1` (cửa sổ 1) rồi `tools\demo_app.ps1` (cửa sổ 2). Chỉ dùng dữ liệu GIẢ.
Server tự nạp 10 thuốc mẫu (Panadol 500mg, Efferalgan, Hapacol 500, Para 500, Decolgen, Amoxicillin 500mg, Strepsils,
Paracetamol Stada 500mg, Thuốc ho Bảo Thanh, Dầu gió xanh) và lưu tài khoản vào `server\medassist-core\data\medassist.dev.db`.
Tài khoản demo: tên đăng nhập `demo` / mật khẩu `Demo@12345678` (ô đăng nhập dùng TÊN ĐĂNG NHẬP, không phải email).
Về AI: hiện KHÔNG có LLM thật — agent là truy hồi + kiểm chứng trích dẫn, mô hình sinh chữ là mock (xem `docs/architecture.html`).

## Giới thiệu (30 giây)
MedAssist: app tra cứu thuốc cho người Việt. Mọi câu trả lời y khoa đều có trích nguồn;
không có nguồn thì nói "không đủ bằng chứng" và khuyên hỏi dược sĩ. Dữ liệu sức khoẻ ưu tiên xử lý trên máy.

## Luồng demo
1. **Đăng ký / đăng nhập** (tài khoản giả, vd `demo@example.com`). Nhấn mạnh: mật khẩu băm bcrypt, token lưu secure storage.
2. **Tìm thuốc**: gõ "Panadol" → thông tin thuốc. Gõ "paracetamol" → nhiều thuốc cùng hoạt chất.
3. **Hỏi agent (chat SSE)**: "Panadol có tác dụng phụ gì" → câu trả lời chạy từng chữ, có nguồn trích dẫn.
   Hỏi thứ không có trong dữ liệu → app trả "không đủ bằng chứng", không bịa.
4. **Câu hỏi y khoa bị từ chối**: hỏi "liều dùng Panadol cho trẻ em" hoặc "Panadol và Amoxicillin có tương tác không"
   → agent từ chối (`out_of_scope`) và khuyên hỏi dược sĩ. Đây là thiết kế an toàn: dữ liệu hiện chỉ là 10 thuốc MẪU,
   chưa phải nguồn lâm sàng, nên agent không kết luận liều dùng/tương tác.
5. **Hồ sơ sức khoẻ** (có đồng ý, lưu SQLite trên máy) → thuốc đang dùng được agent tính đến.
6. **Nhắc uống thuốc**: tạo lịch nhắc → thông báo đúng giờ (đặt trước 1–2 phút để cho xem).
7. **OCR hộp thuốc** (ML Kit chạy trên máy, không gửi ảnh đi) — dùng ảnh mẫu GIẢ. Đưa ảnh từ máy tính vào máy ảo/điện thoại:
   **Tự đồng bộ (khuyên dùng):** `.\tools\sync_images.ps1` rồi bỏ ảnh vào thư mục `anh_thuoc\` (hoặc `-Folder "D:\anh"`); vài giây sau ảnh
   hiện trong app. Đẩy tay một lần: `.\tools\push_image.ps1 -Path C:\anh\hop-thuoc.jpg` (hoặc `-Folder <thư mục>`; `-Clean` để xoá ảnh đã đẩy). Trong app:
   Quét thuốc → Chọn ảnh → album "MedAssist". Ảnh chỉ chép qua adb sang thiết bị, không lên server.
8. **Xoá tài khoản** → xoá toàn bộ dữ liệu (quyền riêng tư).

## Điểm kỹ thuật để nói khi báo cáo
- Flutter + Riverpod, kiến trúc ports & adapters; FastAPI backend; agent có tool registry + MCP.
- Retrieval BM25 có neo theo tên thuốc; bộ đánh giá `eval.py` 109/109.
- Kiểm thử: `flutter test` 325, `pytest` xanh. Build release có R8, `.aab` 41 MB, 3 flavor dev/staging/prod.
- Bảo mật/riêng tư: không log token/email/thuốc, HTTPS bắt buộc ở release, phân quyền theo `user_id`,
  `docs/PRIVACY.md`.

## Câu hỏi thường gặp
- *Có thay thế bác sĩ không?* Không — luôn khuyên hỏi dược sĩ/bác sĩ, có disclaimer.
- *Dữ liệu đi đâu?* Hồ sơ + OCR ở máy; server chỉ nhận câu hỏi tra cứu, có retention.
- *Còn thiếu gì?* Crash reporting (cần chọn SDK + consent), ký khoá release, beta nội bộ.

## Nếu có sự cố khi demo
- App báo lỗi mạng → kiểm tra cửa sổ server còn chạy; máy ảo gọi host qua `10.0.2.2:8000`.
- Thông báo nhắc không hiện → cấp quyền thông báo trong cài đặt máy ảo.
- Chuẩn bị sẵn ảnh chụp màn hình/quay video cả luồng làm phương án dự phòng.
