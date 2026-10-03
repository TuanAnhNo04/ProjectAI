import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/domain/ports/medicine_analyzer.dart';

/// Adapter gọi **Flask hiện tại** để nhận diện thuốc từ ảnh.
///
/// Đây là adapter đầu tiên của cổng [MedicineAnalyzer] (Day 7). Nó chỉ biết
/// hợp đồng HTTP cũ: `POST /process-image` là `multipart/form-data` với field
/// `image`, trả `{"medicine_info": [...]}`. Việc UI/domain không hề biết tới
/// Flask là cả mục đích của ports & adapters — Day 26 thêm `OnDeviceAnalyzer`
/// mà không phải sửa dòng nào ở tầng trên.
class RemoteAnalyzer implements MedicineAnalyzer {
  RemoteAnalyzer(
    this._api, {
    String path = '/process-image',
    String field = 'image',
  })  : _path = path,
        _field = field;

  final ApiClient _api;

  /// Đường dẫn endpoint (đổi được để test/Đổi backend sau này).
  final String _path;

  /// Tên form field — Flask cũ đọc `request.files['image']`.
  final String _field;

  /// Nhãn dùng cho log/màn "Chẩn đoán": đang chạy tầng từ xa nào.
  @override
  String get capability => 'remote-flask';

  /// Upload ảnh tại [imagePath] và map `medicine_info` thành [AnalysisResult].
  ///
  /// Ném [AppException] khi mất mạng / server lỗi / body sai hình dạng — tầng
  /// UI bọc bằng `guard()` nên không cần `try/catch`.
  @override
  Future<AnalysisResult> analyze(String imagePath) async {
    // Chặn sớm: gửi file rỗng chỉ tổ tốn một vòng request vô nghĩa.
    if (imagePath.trim().isEmpty) {
      throw const AppException(
        kind: AppExceptionKind.validation,
        message: 'Chưa có ảnh để phân tích.',
      );
    }

    final response = await _api.upload(
      _path,
      files: [UploadFile(path: imagePath, field: _field)],
    );

    // Dio đã decode JSON thành Map; fromJson chấp nhận cả 3 kiểu body
    // (`medicines` / `medicine_info` / `search_results`) — xem AnalysisResult.
    return AnalysisResult.fromJson(asJsonMap(response.data, _path));
  }
}
