import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/medicine.dart';

/// Một lần quét ảnh đã lưu trong `GET /history`.
///
/// Cấu trúc server: `{"id": 1, "filepath": "...", "search_results": [...]}`.
class HistoryItem {
  const HistoryItem({
    this.id,
    required this.filepath,
    this.searchResults = const <Medicine>[],
    this.createdAt,
  });

  factory HistoryItem.fromJson(Map<String, dynamic> json) => HistoryItem(
        id: asIntOrNull(json[keyId]),
        filepath: asString(json[keyFilepath]),
        searchResults:
            Medicine.listFromJson(json[keySearchResults], keySearchResults),
        createdAt: asDateTimeOrNull(json[keyCreatedAt]),
      );

  static const String keyId = 'id';
  static const String keyFilepath = 'filepath';
  static const String keySearchResults = 'search_results';
  static const String keyCreatedAt = 'created_at';

  /// Khoá xoá ở `DELETE /history/{id}`. `null` nếu server không trả id —
  /// đúng lý do màn cũ hiện `Invalid ID`.
  final int? id;

  /// Đường dẫn ảnh **trên máy chủ** (nơi file đã lưu), không phải trên điện
  /// thoại — dùng để đối chiếu, không mở trực tiếp được.
  final String filepath;

  final List<Medicine> searchResults;

  /// Server cũ chưa trả trường này; FastAPI (Day 15) sẽ có.
  final DateTime? createdAt;

  bool get isValid => id != null && filepath.isNotEmpty;

  HistoryItem copyWith({
    Object? id = notProvided,
    String? filepath,
    List<Medicine>? searchResults,
    Object? createdAt = notProvided,
  }) =>
      HistoryItem(
        id: pickChanged(id, this.id),
        filepath: filepath ?? this.filepath,
        searchResults: searchResults ?? this.searchResults,
        createdAt: pickChanged(createdAt, this.createdAt),
      );

  Map<String, dynamic> toJson() => <String, dynamic>{
        keyId: id,
        keyFilepath: filepath,
        keySearchResults: searchResults.map((m) => m.toJson()).toList(),
        keyCreatedAt: createdAt?.toIso8601String(),
      };

  /// Bỏ bản ghi không xoá được (thiếu `id`) khỏi danh sách — tránh hiện nút xoá
  /// chết trong UI.
  static List<HistoryItem> listFromJson(Object? raw, String field) => asJsonList(
        raw,
        field,
      )
          .map((e) => HistoryItem.fromJson(asJsonMap(e, field)))
          .where((item) => item.isValid)
          .toList(growable: false);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is HistoryItem &&
          other.id == id &&
          other.filepath == filepath &&
          other.createdAt == createdAt &&
          _sameMedicines(other.searchResults, searchResults);

  @override
  int get hashCode =>
      Object.hash(id, filepath, createdAt, Object.hashAll(searchResults));

  @override
  String toString() => 'HistoryItem($id, $filepath)';
}

/// Phản hồi `GET /history` → `{"history": [...]}`.
class HistoryResponse {
  const HistoryResponse({required this.items});

  factory HistoryResponse.fromJson(Map<String, dynamic> json) => HistoryResponse(
        items: HistoryItem.listFromJson(json[keyHistory], keyHistory),
      );

  factory HistoryResponse.parse(String body) =>
      HistoryResponse.fromJson(decodeJsonObject(body));

  static const String keyHistory = 'history';

  final List<HistoryItem> items;

  bool get isEmpty => items.isEmpty;

  @override
  String toString() => 'HistoryResponse(${items.length})';
}

bool _sameMedicines(List<Medicine> a, List<Medicine> b) {
  if (identical(a, b)) return true;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}