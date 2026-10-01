import 'package:image_processing_app/data/models/analysis_result.dart';

/// Cổng trừu tượng cho việc **nhận diện thuốc từ ảnh**.
///
/// Đây là hợp đồng mà tầng UI/domain dùng, còn *cách thức* nhận diện thì tùy
/// adapter — đúng nguyên tắc ports & adapters:
///
/// | Adapter (Day)         | `capability` | Hoạt động khi mất mạng |
/// |-----------------------|--------------|-------------------------|
/// | `RemoteAnalyzer` (8)  | Flask server | ❌ không                 |
/// | `OnDeviceAnalyzer` (26)| ML Kit OCR  | ✅ có                    |
/// | `HybridAnalyzer` (26) | local→remote| ⚠️ có (dùng cache)      |
///
/// **Day 26 chỉ cần thêm adapter — sửa orchestrator/UI ở đây.**
abstract interface class MedicineAnalyzer {
  /// Phân tích ảnh tại [imagePath] (đường dẫn **trên thiết bị**).
  ///
  /// Ném [AppException] khi lỗi — bọc bằng `guard()` ở tầng trên để thành
  /// `Result<AnalysisResult>`; UI không bao giờ phải `try/catch`.
  Future<AnalysisResult> analyze(String imagePath);

  /// Nhãn ngắn mô tả bản hiện tại (`'remote-flask'`, `'ondevice-mlkit'`, ...).
  ///
  /// Dùng cho màn "Chẩn đoán" và để ghi log xem app đang chạy tầng nào —
  /// cần thiết khi Day 26 bật/tắt bằng `--dart-define=ANALYZER=hybrid`.
  String get capability;
}