import 'package:image_processing_app/data/models/json_utils.dart';
import 'package:image_processing_app/data/models/medicine.dart';

/// Kết quả phân tích ảnh đơn thuốc — đầu ra của tầng `MedicineAnalyzer`
/// (Day 7 định nghĩa port, Day 8 có adapter gọi Flask).
///
/// Nhận **3 kiểu body** vì server đã đổi hình dạng theo từng endpoint:
/// - `POST /process-image` → `{"medicine_info": [...]}`
/// - `GET /history/{id}` / cache Day 12 → `{"search_results": [...]}`
/// - `POST /process-image` bản mới → `{"text": "...", "medicines": [...]}`
class AnalysisResult {
  const AnalysisResult({
    this.medicines = const <Medicine>[],
    this.text,
    this.imagePath,
    this.analyzedAt,
  });

  /// Rỗng — dùng khi ảnh không nhận ra thuốc nào.
  const AnalysisResult.empty() : this();

  factory AnalysisResult.fromJson(Map<String, dynamic> json) => AnalysisResult(
        // Ưu tiên `medicines` (hợp đồng mới) -> `medicine_info` (Flask cũ)
        // -> `search_results` (dạng lưu trong history/cache).
        medicines: Medicine.listFromJson(
          json[keyMedicines] ??
              json[keyMedicineInfo] ??
              json[keySearchResults],
          keyMedicineInfo,
        ),
        text: asStringOrNull(json[keyText]),
        imagePath: asStringOrNull(json[keyImagePath]),
        analyzedAt: asDateTimeOrNull(json[keyAnalyzedAt]),
      );

  factory AnalysisResult.parse(String body) =>
      AnalysisResult.fromJson(decodeJsonObject(body));

  static const String keyMedicineInfo = 'medicine_info';
  static const String keyMedicines = 'medicines';
  static const String keySearchResults = 'search_results';
  static const String keyText = 'text';
  static const String keyImagePath = 'image_path';
  static const String keyAnalyzedAt = 'analyzed_at';

  final List<Medicine> medicines;

  /// Text OCR thô nếu server trả kèm — Day 26 dùng cho hiển thị nguồn.
  final String? text;

  /// Đường dẫn ảnh **trên máy chủ** (đã lưu), dùng để xoá/đối chiếu.
  final String? imagePath;

  final DateTime? analyzedAt;

  bool get isEmpty => medicines.isEmpty;
  bool get isNotEmpty => medicines.isNotEmpty;

  /// Dòng hiển thị `"Nhãn: giá trị"` — **thay cho** `split(': ')` của code cũ.
  ///
  /// Code cũ làm `details[0].split(': ').last`, nên tên thuốc chứa `: `
  /// (hoặc `Composition` chứa `:`) sẽ hiển thị sai. Ở đây ghép trực tiếp
  /// từ field đã tách sẵn nên không còn cách nào vỡ.
  List<String> toDisplayLines() => <String>[
        for (final m in medicines) ...<String>[
          '$labelName: ${m.name}',
          if (m.composition != null) '$labelComposition: ${m.composition}',
          if (m.uses != null) '$labelUses: ${m.uses}',
          if (m.sideEffects != null) '$labelSideEffects: ${m.sideEffects}',
        ],
      ];

  static const String labelName = 'Tên';
  static const String labelComposition = 'Thành phần';
  static const String labelUses = 'Công dụng';
  static const String labelSideEffects = 'Tác dụng phụ';

  AnalysisResult copyWith({
    List<Medicine>? medicines,
    Object? text = notProvided,
    Object? imagePath = notProvided,
    Object? analyzedAt = notProvided,
  }) =>
      AnalysisResult(
        medicines: medicines ?? this.medicines,
        text: pickChanged(text, this.text),
        imagePath: pickChanged(imagePath, this.imagePath),
        analyzedAt: pickChanged(analyzedAt, this.analyzedAt),
      );

  Map<String, dynamic> toJson() => <String, dynamic>{
        keyMedicines: medicines.map((m) => m.toJson()).toList(),
        keyText: text,
        keyImagePath: imagePath,
        keyAnalyzedAt: analyzedAt?.toIso8601String(),
      };

  @override
  String toString() =>
      'AnalysisResult(${medicines.length} thuốc${isEmpty ? ', rỗng' : ''})';
}