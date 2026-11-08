# Danh sách cơ sở y tế mẫu (`assets/data/medical_facilities.json`)

> **CẢNH BÁO: DỮ LIỆU MẪU, CHƯA KIỂM CHỨNG.** Không dùng cho tình huống thật trước
> khi chủ dự án xác minh từng dòng.

## Hiện trạng
- 9 bệnh viện lớn (7 ở TP.HCM, 2 ở Hà Nội), do AI soạn từ hiểu biết chung.
- Toạ độ (`lat`, `lng`) là **gần đúng**; địa chỉ dùng tên quận/huyện cũ (trước đợt
  sắp xếp đơn vị hành chính) và có thể đã thay đổi.
- Mọi dòng có `"verified": false`.
- **Không có số điện thoại** của bệnh viện — cố ý, để tránh số bịa. Số duy nhất trong
  app là cấp cứu quốc gia **115** (mở trình quay số, người dùng tự bấm gọi).

## Việc chủ dự án cần làm trước khi dùng thật
1. Đối chiếu tên, địa chỉ, toạ độ với nguồn chính thức (Sở Y tế / website bệnh viện).
2. Thay/bổ sung danh sách cho khu vực triển khai; đặt `"verified": true` cho dòng đã kiểm.
3. Nếu thêm số điện thoại, chỉ lấy từ nguồn chính thức và ghi nguồn + ngày kiểm.
4. Cập nhật banner "Dữ liệu mẫu, cần kiểm chứng" trong app khi dữ liệu đã được xác minh.

## Quyền riêng tư
- Vị trí người dùng lấy qua `LocationManager` (MethodChannel `medassist/device`), chỉ
  dùng để tính khoảng cách haversine **trên máy**; không lưu, không ghi log, không gửi đi.
- Chỉ đường / tìm thêm mở app bản đồ bằng intent `geo:` — app bản đồ chỉ nhận toạ độ
  của bệnh viện (hoặc từ khoá tìm kiếm), không nhận vị trí người dùng từ MedAssist.
