import 'package:image_processing_app/data/models/json_utils.dart';

/// Một dòng thuốc trong `medicines.csv` mà server Flask đang phục vụ.
///
/// Toàn bộ **tên trường JSON gom vào hằng số** ở đây — gõ sai tên trường sẽ
/// bị compiler bắt ngay thay vì âm thầm ra `null` lúc chạy (đúng tinh thần
/// "test bằng JSON thật để phát hiện sai tên trường ngay lập tức").
///
/// Tên trường **giữ nguyên** chữ hoa/dấu cách của server (`Medicine_Name`,
/// `Excellent Review %`) vì đó là hợp đồng đã có; phía app mới là phần có
/// phong cách `camelCase`.
class Medicine {
  const Medicine({
    this.name = '',
    this.composition,
    this.uses,
    this.sideEffects,
    this.manufacturer,
    this.excellentReviewPercent,
    this.averageReviewPercent,
    this.poorReviewPercent,
  });

  factory Medicine.fromJson(Map<String, dynamic> json) => Medicine(
        name: asString(json[keyName]),
        composition: asStringOrNull(json[keyComposition]),
        uses: asStringOrNull(json[keyUses]),
        sideEffects: asStringOrNull(json[keySideEffects]),
        manufacturer: asStringOrNull(json[keyManufacturer]),
        excellentReviewPercent: asPercent(json[keyExcellent]),
        averageReviewPercent: asPercent(json[keyAverage]),
        poorReviewPercent: asPercent(json[keyPoor]),
      );

  /// --- Tên trường JSON của server (hợp đồng, đừng đổi) ---
  static const String keyName = 'Medicine_Name';
  static const String keyComposition = 'Composition';
  static const String keyUses = 'Uses';
  static const String keySideEffects = 'Side_effects';
  static const String keyManufacturer = 'Manufacturer';
  static const String keyExcellent = 'Excellent Review %';
  static const String keyAverage = 'Average Review %';
  static const String keyPoor = 'Poor Review %';

  /// Tên thuốc. Rỗng khi server không có dữ liệu (không dùng `null` để UI
  /// không phải `?? 'Không có thông tin'` ở mọi chỗ).
  final String name;

  final String? composition;
  final String? uses;
  final String? sideEffects;
  final String? manufacturer;

  /// Tỉ lệ đánh giá (0–100), `null` khi server không có cột này.
  final double? excellentReviewPercent;
  final double? averageReviewPercent;
  final double? poorReviewPercent;

  Medicine copyWith({
    String? name,
    Object? composition = notProvided,
    Object? uses = notProvided,
    Object? sideEffects = notProvided,
    Object? manufacturer = notProvided,
    Object? excellentReviewPercent = notProvided,
    Object? averageReviewPercent = notProvided,
    Object? poorReviewPercent = notProvided,
  }) =>
      Medicine(
        name: name ?? this.name,
        composition: pickChanged(composition, this.composition),
        uses: pickChanged(uses, this.uses),
        sideEffects: pickChanged(sideEffects, this.sideEffects),
        manufacturer: pickChanged(manufacturer, this.manufacturer),
        excellentReviewPercent: _toPercent(
            pickChanged(excellentReviewPercent, this.excellentReviewPercent)),
        averageReviewPercent: _toPercent(
            pickChanged(averageReviewPercent, this.averageReviewPercent)),
        poorReviewPercent:
            _toPercent(pickChanged(poorReviewPercent, this.poorReviewPercent)),
      );

  Map<String, dynamic> toJson() => <String, dynamic>{
        keyName: name,
        keyComposition: composition,
        keyUses: uses,
        keySideEffects: sideEffects,
        keyManufacturer: manufacturer,
        keyExcellent: formatPercent(excellentReviewPercent),
        keyAverage: formatPercent(averageReviewPercent),
        keyPoor: formatPercent(poorReviewPercent),
      };

  /// Đọc mảng thuốc từ server (dùng chung cho `results` và `search_results`).
  static List<Medicine> listFromJson(Object? raw, String field) =>
      asJsonList(raw, field)
          .map((e) => Medicine.fromJson(asJsonMap(e, field)))
          .toList(growable: false);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Medicine &&
          other.name == name &&
          other.composition == composition &&
          other.uses == uses &&
          other.sideEffects == sideEffects &&
          other.manufacturer == manufacturer &&
          other.excellentReviewPercent == excellentReviewPercent &&
          other.averageReviewPercent == averageReviewPercent &&
          other.poorReviewPercent == poorReviewPercent;

  @override
  int get hashCode => Object.hash(
        name,
        composition,
        uses,
        sideEffects,
        manufacturer,
        excellentReviewPercent,
        averageReviewPercent,
        poorReviewPercent,
      );

  @override
  String toString() => 'Medicine($name)';
}

/// `copyWith(poorReviewPercent: 99)` truyền `int` — Dart sẽ ném
/// *"type 'int' is not a subtype of type 'double?'"* nếu ép trực tiếp.
/// Quy về `double` ở đây để người gọi không phải nhớ viết `99.0`.
double? _toPercent(Object? value) =>
    value is num ? value.toDouble() : value as double?;

/// Phản hồi của `GET /search-medicine?name=...`.
///
/// Bọc cả gói `{"results": [...]}` để adapter (Day 8) không phải tự mở `Map`
/// rồi ép kiểu — đó chính là chỗ "parse JSON thủ công" mà Day 6 loại bỏ.
class MedicineSearchResponse {
  const MedicineSearchResponse({
    required this.results,
    this.query,
    int? total,
  }) : _serverTotal = total;

  factory MedicineSearchResponse.fromJson(Map<String, dynamic> json) {
    final results = Medicine.listFromJson(json[keyResults], keyResults);
    return MedicineSearchResponse(
      results: results,
      // Server có thể echo lại tên đã tìm; không có thì `null`.
      query: asStringOrNull(json[keyQuery]),
      total: asIntOrNull(json[keyTotal]),
    );
  }

  /// Giải nén thẳng từ body text: `MedicineSearchResponse.parse(body)`.
  factory MedicineSearchResponse.parse(String body) =>
      MedicineSearchResponse.fromJson(decodeJsonObject(body));

  static const String keyResults = 'results';
  static const String keyQuery = 'query';
  static const String keyTotal = 'total';

  final List<Medicine> results;

  /// Tên đã tìm (server echo lại), `null` nếu không có.
  final String? query;

  /// `total` do server trả; `null` khi server không có trường này.
  final int? _serverTotal;

  /// Tổng số kết quả — **luôn có giá trị**: lấy của server nếu có, nếu không
  /// thì bằng số kết quả vừa nhận (mọt trường hợp server cũ đều rơi vào đây).
  int get total => _serverTotal ?? results.length;

  bool get isEmpty => results.isEmpty;

  @override
  String toString() => 'MedicineSearchResponse(${results.length}/$total)';
}
