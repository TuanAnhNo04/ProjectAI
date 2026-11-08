import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/medicine.dart';

/// Tải **toàn bộ** danh mục thuốc từ server (phân trang) cho OCR trên máy.
///
/// Danh mục là dữ liệu công khai, không gắn người dùng nào — khác với lịch sử.
class RemoteMedicineCatalog {
  RemoteMedicineCatalog(this._api, {required String path, this.pageSize = 100})
      : _path = path;

  final ApiClient _api;
  final String _path;
  final int pageSize;

  /// Giới hạn cứng số trang: server lỗi trả `total` vô lý không được làm app
  /// lặp mãi.
  static const int maxPages = 50;

  Future<List<Medicine>> fetchAll() async {
    final all = <Medicine>[];
    var offset = 0;
    for (var page = 0; page < maxPages; page++) {
      final response = await _api.get(
        _path,
        query: {'limit': pageSize, 'offset': offset},
      );
      final json = asJsonMap(response.data, _path);
      final batch = Medicine.listFromJson(json['results'], 'results');
      all.addAll(batch);
      final total = asIntOrNull(json['total']) ?? all.length;
      offset += batch.length;
      if (batch.isEmpty || offset >= total) break;
    }
    return List<Medicine>.unmodifiable(all);
  }
}
