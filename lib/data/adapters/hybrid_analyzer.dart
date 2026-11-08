import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/domain/ports/medicine_analyzer.dart';

/// Adapter lai: thử on-device trước, rơi về remote nếu cần (Day 26).
///
/// Luồng:
/// 1. [_local].analyze() — ML Kit OCR → tra DB local.
/// 2. Nếu kết quả **rỗng** (OCR không đọc được hoặc không match thuốc):
///    → thử [_remote].analyze() (cần mạng).
/// 3. Nếu remote cũng lỗi → tìm trong cache offline.
/// 4. Kết quả thành công luôn được ghi cache.
class HybridAnalyzer implements MedicineAnalyzer {
  HybridAnalyzer({
    required MedicineAnalyzer local,
    required MedicineAnalyzer remote,
    required OfflineCache cache,
  })  : _local = local,
        _remote = remote,
        _cache = cache;

  final MedicineAnalyzer _local;
  final MedicineAnalyzer _remote;
  final OfflineCache _cache;

  static const String _cachePrefix = 'analysis.';

  @override
  String get capability => 'hybrid';

  @override
  Future<AnalysisResult> analyze(String imagePath) async {
    // Bước 1: thử on-device.
    try {
      final local = await _local.analyze(imagePath);
      if (local.isNotEmpty) {
        _saveCache(imagePath, local);
        return local;
      }
    } on AppException {
      // On-device lỗi → tiếp tục sang remote.
    }

    // Bước 2: fallback remote.
    try {
      final remote = await _remote.analyze(imagePath);
      _saveCache(imagePath, remote);
      return remote;
    } on AppException {
      // Remote cũng lỗi → thử cache.
    }

    // Bước 3: cache.
    final cached = await _readCache(imagePath);
    if (cached != null && cached.isNotEmpty) return cached;

    // Hoàn toàn không có kết quả.
    throw const AppException(
      kind: AppExceptionKind.network,
      message:
          'Không nhận diện được thuốc. Kiểm tra ảnh rõ nét và thử lại khi có mạng.',
    );
  }

  String _cacheKey(String imagePath) =>
      '$_cachePrefix${imagePath.hashCode.toRadixString(36)}';

  Future<void> _saveCache(String imagePath, AnalysisResult result) async {
    try {
      await _cache.save(_cacheKey(imagePath), result.toJson());
    } on Object {
      // Cache là tiện ích, lỗi ghi không làm hỏng luồng chính.
    }
  }

  Future<AnalysisResult?> _readCache(String imagePath) async {
    try {
      final raw = await _cache.load(_cacheKey(imagePath));
      if (raw == null) return null;
      return AnalysisResult.fromJson(raw);
    } on Object {
      return null;
    }
  }
}
