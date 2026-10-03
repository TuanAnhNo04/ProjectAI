import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';

/// Adapter dữ liệu thuốc gọi **Flask hiện tại** (`/search-medicine`, `/history`).
///
/// Mọi URL đi qua [ApiClient] nên tên thuốc được **encode chuẩn** (sửa bug cũ
/// `?name=$ten` nối chuỗi tay), và mọi lỗi mạng thành [AppException].
class RemoteMedicineRepository implements MedicineRepository {
  RemoteMedicineRepository(
    this._api, {
    String searchPath = '/search-medicine',
    String historyPath = '/history',
  })  : _searchPath = searchPath,
        _historyPath = historyPath;

  final ApiClient _api;
  final String _searchPath;
  final String _historyPath;

  /// `GET /search-medicine?name=<name>` → bọc trong [MedicineSearchResponse]
  /// để giữ luôn `query`/`total` mà server echo lại.
  @override
  Future<MedicineSearchResponse> search(String name) async {
    final response = await _api.get(_searchPath, query: {'name': name});
    return MedicineSearchResponse.fromJson(
      asJsonMap(response.data, _searchPath),
    );
  }

  /// `GET /history` → `{"history": [...]}`; bản ghi thiếu `id` bị lọc sẵn
  /// trong [HistoryItem.listFromJson].
  @override
  Future<List<HistoryItem>> history() async {
    final response = await _api.get(_historyPath);
    return HistoryResponse.fromJson(
      asJsonMap(response.data, _historyPath),
    ).items;
  }

  /// `DELETE /history/{id}` — ném `AppExceptionKind.notFound` nếu server không
  /// còn bản ghi đó.
  @override
  Future<void> deleteHistory(int id) => _api.delete('$_historyPath/$id');
}
