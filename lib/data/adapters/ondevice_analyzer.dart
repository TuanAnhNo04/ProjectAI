import 'dart:io';

import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/domain/ports/medicine_analyzer.dart';

/// Adapter nhận diện thuốc **trên thiết bị** bằng ML Kit OCR (Day 26).
///
/// Quy trình: ảnh → ML Kit Text Recognition → chuẩn hoá chuỗi → tra cứu
/// trong danh sách thuốc đã cache (không cần mạng).
///
/// Giới hạn: OCR chỉ đọc chữ in; chữ viết tay và ảnh mờ cho kết quả kém.
/// Khi không match được thuốc nào, trả [AnalysisResult] rỗng (không ném lỗi)
/// để [HybridAnalyzer] có thể fallback sang remote.
class OnDeviceAnalyzer implements MedicineAnalyzer {
  OnDeviceAnalyzer({required this.medicineDatabase});

  /// Danh sách thuốc đã cache — truyền vào từ provider để test được.
  final List<Medicine> medicineDatabase;

  final _recognizer = TextRecognizer();

  @override
  String get capability => 'ondevice-mlkit';

  @override
  Future<AnalysisResult> analyze(String imagePath) async {
    if (imagePath.trim().isEmpty) {
      throw const AppException(
        kind: AppExceptionKind.validation,
        message: 'Chưa có ảnh để phân tích.',
      );
    }

    final file = File(imagePath);
    if (!file.existsSync()) {
      throw const AppException(
        kind: AppExceptionKind.validation,
        message: 'Không tìm thấy file ảnh.',
      );
    }

    final inputImage = InputImage.fromFilePath(imagePath);
    final RecognizedText recognized;
    try {
      recognized = await _recognizer.processImage(inputImage);
    } catch (e) {
      throw AppException(
        kind: AppExceptionKind.unknown,
        message: 'Không nhận diện được chữ trong ảnh.',
        cause: e,
      );
    }

    final ocrText = recognized.text;
    if (ocrText.trim().isEmpty) {
      return const AnalysisResult.empty();
    }

    final matches = _matchMedicines(ocrText);

    return AnalysisResult(
      medicines: matches,
      text: ocrText,
    );
  }

  /// Tìm thuốc trong [medicineDatabase] dựa trên OCR text.
  ///
  /// Dùng fuzzy matching đơn giản: chuẩn hoá cả hai rồi kiểm tra substring.
  List<Medicine> _matchMedicines(String ocrText) {
    final normalized = _normalize(ocrText);
    final found = <Medicine>[];

    for (final med in medicineDatabase) {
      final medName = _normalize(med.name);
      if (medName.isEmpty) continue;

      if (normalized.contains(medName)) {
        found.add(med);
      } else {
        // Tách từng token của tên thuốc, nếu >= 60% token khớp thì coi là match.
        final tokens = medName.split(RegExp(r'\s+'));
        if (tokens.length >= 2) {
          final matchCount = tokens
              .where((t) => t.length > 2 && normalized.contains(t))
              .length;
          if (matchCount / tokens.length >= 0.6) {
            found.add(med);
          }
        }
      }
    }

    return found;
  }

  /// Chuẩn hoá: lowercase, bỏ dấu tiếng Việt, bỏ ký tự đặc biệt.
  static String _normalize(String input) {
    var s = input.toLowerCase();
    s = _removeDiacritics(s);
    s = s.replaceAll(RegExp(r'[^\w\s]'), ' ');
    s = s.replaceAll(RegExp(r'\s+'), ' ').trim();
    return s;
  }

  static String _removeDiacritics(String s) {
    const map = {
      'à': 'a',
      'á': 'a',
      'ả': 'a',
      'ã': 'a',
      'ạ': 'a',
      'ă': 'a',
      'ằ': 'a',
      'ắ': 'a',
      'ẳ': 'a',
      'ẵ': 'a',
      'ặ': 'a',
      'â': 'a',
      'ầ': 'a',
      'ấ': 'a',
      'ẩ': 'a',
      'ẫ': 'a',
      'ậ': 'a',
      'đ': 'd',
      'è': 'e',
      'é': 'e',
      'ẻ': 'e',
      'ẽ': 'e',
      'ẹ': 'e',
      'ê': 'e',
      'ề': 'e',
      'ế': 'e',
      'ể': 'e',
      'ễ': 'e',
      'ệ': 'e',
      'ì': 'i',
      'í': 'i',
      'ỉ': 'i',
      'ĩ': 'i',
      'ị': 'i',
      'ò': 'o',
      'ó': 'o',
      'ỏ': 'o',
      'õ': 'o',
      'ọ': 'o',
      'ô': 'o',
      'ồ': 'o',
      'ố': 'o',
      'ổ': 'o',
      'ỗ': 'o',
      'ộ': 'o',
      'ơ': 'o',
      'ờ': 'o',
      'ớ': 'o',
      'ở': 'o',
      'ỡ': 'o',
      'ợ': 'o',
      'ù': 'u',
      'ú': 'u',
      'ủ': 'u',
      'ũ': 'u',
      'ụ': 'u',
      'ư': 'u',
      'ừ': 'u',
      'ứ': 'u',
      'ử': 'u',
      'ữ': 'u',
      'ự': 'u',
      'ỳ': 'y',
      'ý': 'y',
      'ỷ': 'y',
      'ỹ': 'y',
      'ỵ': 'y',
    };
    final buf = StringBuffer();
    for (final c in s.split('')) {
      buf.write(map[c] ?? c);
    }
    return buf.toString();
  }

  void dispose() {
    _recognizer.close();
  }
}
