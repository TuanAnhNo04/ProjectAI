import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';

/// Cổng trừu tượng cho dữ liệu thuốc: tìm kiếm + lịch sử quét ảnh.
///
/// Adapter từ xa (`RemoteMedicineRepository`, Day 8) nối Flask; sau này có thể
/// thêm adapter cache ngoại tuyến (Day 12) mà UI không đổi dòng nào.
abstract interface class MedicineRepository {
  /// Tìm thuốc theo [name] — **server tự encode**, không nối chuỗi tay.
  ///
  /// Trả về [MedicineSearchResponse] thay vì `List<Medicine>` để giữ luôn
  /// `query` và `total` mà server echo lại (hữu ích cho "tìm 1/24").
  Future<MedicineSearchResponse> search(String name);

  /// Lịch sử các lần đã quét, mới nhất trước. Bản ghi thiếu `id` đã bị lọc
  /// sẵn trong model.
  Future<List<HistoryItem>> history();

  /// Xoá một bản ghi. [id] phải khác `null` — xem `HistoryItem.id`.
  ///
  /// Ném `AppExceptionKind.notFound` nếu server không còn bản ghi đó.
  Future<void> deleteHistory(int id);
}
