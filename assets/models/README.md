# assets/models

Thư mục chứa model on-device (`.tflite`) — theo kế hoạch `docs/ROADMAP.md` sẽ dùng từ **Day 26**
(ML Kit OCR + TFLite, adapter `OnDeviceAnalyzer` / `HybridAnalyzer`).

Hiện **chưa có model nào**: app đang chạy theo hướng *thin client* (gọi API Flask).
Thư mục này tồn tại vì `pubspec.yaml` đã khai báo `assets: - assets/models/`;
nếu để trống, `flutter analyze` sẽ báo `asset_directory_does_not_exist`.

Khi thêm model thật, bỏ file README này (hoặc giữ cũng không sao).
