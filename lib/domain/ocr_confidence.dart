/// Đánh giá độ chắc chắn của một dòng thuốc đọc từ OCR + so khớp với danh mục.
///
/// Thuần Dart, chạy trên máy. Chỉ là **tín hiệu để người dùng kiểm tra lại** — không
/// tự sửa tên thuốc, không tự chọn thuốc thay người dùng (đọc nhầm một ký tự có thể
/// ghi nhầm thuốc/liều). Không ghi log tên thuốc.
library;

import 'package:image_processing_app/domain/prescription_parser.dart'
    show foldVietnamese;

/// Lý do khiến một dòng OCR đáng ngờ.
enum OcrIssue {
  /// Không đọc rõ số lần dùng (parser đã đánh dấu `needsReview`).
  unclearFrequency,

  /// Không đọc được liều mỗi lần.
  missingDose,

  /// Tên quá ngắn để là tên thuốc.
  shortName,

  /// Tên có ký tự lạ hoặc chữ-số lẫn nhau kiểu OCR đọc nhầm ("Paracetam0l").
  suspiciousChars,

  /// Đã đối chiếu danh mục nhưng không tìm thấy.
  notInCatalog,
}

/// Các lý do cần **khoá xác nhận** (phải tick "Tôi đã kiểm tra" mới lưu được).
/// Thiếu liều không khoá: nhiều đơn ghi liều ở dòng riêng, người dùng tự điền.
const Set<OcrIssue> lockingIssues = {
  OcrIssue.unclearFrequency,
  OcrIssue.shortName,
  OcrIssue.suspiciousChars,
  OcrIssue.notInCatalog,
};

final _oddChars = RegExp(r'[|@#~_{}\[\]<>\\^`$]');
// Chữ-số-chữ trong cùng một từ: "paracetam0l", "amoxici1lin". Không bắt "500mg".
final _mixedInWord = RegExp(r'[a-z][0-9]+[a-z]');

/// Các vấn đề của một dòng OCR. [match] là kết quả đối chiếu danh mục nếu người
/// dùng đã bấm đối chiếu, `null` nếu chưa.
List<OcrIssue> assessOcrItem({
  required String name,
  required String dose,
  bool needsReview = false,
  CatalogMatch? match,
}) {
  final folded = foldVietnamese(name).toLowerCase().trim();
  return [
    if (needsReview) OcrIssue.unclearFrequency,
    if (dose.trim().isEmpty) OcrIssue.missingDose,
    if (folded.length < 3) OcrIssue.shortName,
    if (_oddChars.hasMatch(folded) || _mixedInWord.hasMatch(folded))
      OcrIssue.suspiciousChars,
    if (match == CatalogMatch.none) OcrIssue.notInCatalog,
  ];
}

/// Kết quả đối chiếu tên đọc được với danh mục thuốc.
enum CatalogMatch {
  /// Trùng tên (sau khi bỏ dấu, khoảng trắng, ký tự đặc biệt).
  exact,

  /// Có thuốc gần giống (tên này là phần đầu của tên kia hoặc ngược lại).
  similar,

  /// Không có kết quả nào.
  none,
}

/// Bỏ dấu, hạ chữ thường, chỉ giữ chữ và số — để so "Panadol 500mg" với "panadol500mg".
String catalogKey(String s) =>
    foldVietnamese(s).toLowerCase().replaceAll(RegExp(r'[^a-z0-9]'), '');

/// So [name] với tên các thuốc [candidates] do danh mục trả về.
CatalogMatch matchCatalog(String name, Iterable<String> candidates) {
  final key = catalogKey(name);
  if (key.isEmpty) return CatalogMatch.none;
  var similar = false;
  for (final c in candidates) {
    final ck = catalogKey(c);
    if (ck.isEmpty) continue;
    if (ck == key) return CatalogMatch.exact;
    final shorter = ck.length < key.length ? ck : key;
    if (shorter.length >= 4 && (ck.startsWith(key) || key.startsWith(ck))) {
      similar = true;
    }
  }
  if (similar) return CatalogMatch.similar;
  // Danh mục có trả kết quả (tìm theo từ khoá) nhưng tên khác hẳn -> vẫn là "gần
  // giống" để người dùng xem gợi ý, không coi là "không có".
  return candidates.isEmpty ? CatalogMatch.none : CatalogMatch.similar;
}
