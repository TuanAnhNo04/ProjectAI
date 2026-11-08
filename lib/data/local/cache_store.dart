import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

/// Cache JSON tối giản cho dữ liệu **không nhạy cảm** (Day 12).
///
/// ⚠️ **Tuyệt đối không** cache token, mật khẩu hay email vào đây: file
/// `shared_preferences` nằm thẳng trên đĩa, không mã hoá. Token thuộc về
/// `SecureTokenStore` (Day 11) chứ không phải ở đây.
abstract interface class CacheStore {
  /// Đọc giá trị đã cache theo [key]; `null` nếu chưa có.
  Future<Object?> read(String key);

  /// Ghi [value] (phải là JSON-encodable) kèm mốc thời gian.
  Future<void> write(String key, Object? value);

  /// Xoá một mục.
  Future<void> remove(String key);

  /// Xoá toàn bộ cache của app.
  Future<void> clear();
}

/// Cache dùng `shared_preferences` — **chỉ dành cho dữ liệu công khai**.
class PreferencesCacheStore implements CacheStore {
  PreferencesCacheStore({SharedPreferences? preferences})
      : _preferences = preferences;

  SharedPreferences? _preferences;

  /// Tiền tố khoá để không đụng với setting khác của app.
  static const String keyPrefix = 'cache.';

  /// Khoá đi kèm mỗi mục: `{'v': <json>, 'at': <ISO8601>}`.
  ///
  /// Nhờ vậy UI biết dữ liệu ngoại tuyến đã cũ bao lâu mà không cần gọi mạng.
  static const String fieldValue = 'v';
  static const String fieldCachedAt = 'at';

  /// Nền tảng lưu trữ; `null` nếu chưa có (lần đầu mở app, hoặc thiếu plugin).
  Future<SharedPreferences?> _tryPrefs() async {
    try {
      return _preferences ??= await SharedPreferences.getInstance();
    } on Object {
      // Thiếu plugin (test), hoặc lỗi nền tảng -> coi như **không có** kho.
      // Cache chỉ là tiện ích: hỏng thì mất cache, không được làm hỏng app.
      return null;
    }
  }

  @override
  Future<Object?> read(String key) async {
    final prefs = await _tryPrefs();
    if (prefs == null) return null;
    final raw = prefs.getString('$keyPrefix$key');
    if (raw == null) return null;
    try {
      final decoded = jsonDecode(raw);
      if (decoded is! Map) return null;
      final value = decoded[fieldValue];
      return value is Map ? Map<String, dynamic>.from(value) : value;
    } on FormatException {
      // Cache hỏng (đổi phiên bản app, ghi tay) -> coi như không có.
      return null;
    }
  }

  /// Mốc thời gian cache mục [key] gần nhất, `null` nếu chưa cache.
  ///
  /// Dùng cho banner "Dữ liệu ngoại tuyến · cập nhật ...".
  Future<DateTime?> cachedAt(String key) async {
    final prefs = await _tryPrefs();
    if (prefs == null) return null;
    final raw = prefs.getString('$keyPrefix$key');
    if (raw == null) return null;
    try {
      final decoded = jsonDecode(raw);
      if (decoded is! Map) return null;
      final at = decoded[fieldCachedAt];
      return at is String ? DateTime.tryParse(at) : null;
    } on FormatException {
      return null;
    }
  }

  @override
  Future<void> write(String key, Object? value) async {
    final prefs = await _tryPrefs();
    if (prefs == null) return;
    try {
      final payload = jsonEncode(<String, Object?>{
        fieldValue: value,
        fieldCachedAt: DateTime.now().toIso8601String(),
      });
      await prefs.setString('$keyPrefix$key', payload);
    } on Object {
      // Ghi lỗi (đủ dung lượng, quyền...) -> bỏ qua, lần sau vẫn dùng cache cũ.
    }
  }

  @override
  Future<void> remove(String key) async {
    final prefs = await _tryPrefs();
    if (prefs == null) return;
    try {
      await prefs.remove('$keyPrefix$key');
    } on Object {
      // Bỏ qua: không xoá được cache không làm hỏng luồng chính.
    }
  }

  @override
  Future<void> clear() async {
    final prefs = await _tryPrefs();
    if (prefs == null) return;
    // Chỉ xoá khoá có tiền tố của mình, không đụng setting khác.
    try {
      for (final key in prefs.getKeys().toList()) {
        if (key.startsWith(keyPrefix)) await prefs.remove(key);
      }
    } on Object {
      // Bỏ qua.
    }
  }
}

/// Cache trong RAM — dùng cho test và khi `shared_preferences` lỗi.
class InMemoryCacheStore implements CacheStore {
  final Map<String, Object?> _values = <String, Object?>{};
  final Map<String, DateTime> _timestamps = <String, DateTime>{};

  @override
  Future<Object?> read(String key) async => _values[key];

  /// Bản sao của [cachedAt] để test offline banner.
  DateTime? cachedAtOf(String key) => _timestamps[key];

  @override
  Future<void> write(String key, Object? value) async {
    _values[key] = value;
    _timestamps[key] = DateTime.now();
  }

  @override
  Future<void> remove(String key) async {
    _values.remove(key);
    _timestamps.remove(key);
  }

  @override
  Future<void> clear() async {
    _values.clear();
    _timestamps.clear();
  }
}

/// Khoá cache dùng trong app (đặt tên tập trung ở đây để khỏi gõ sai chuỗi).
class CacheKeys {
  const CacheKeys._();

  /// Danh sách lần quét ảnh (`GET /history`).
  static const String history = 'history';

  /// Danh mục thuốc cho OCR trên máy (`GET /v1/medicines`) — dữ liệu công khai.
  static const String catalog = 'catalog';

  /// Tin tức thuốc/y tế (`GET /v1/news`) — dữ liệu công khai, dùng khi mất mạng.
  static const String news = 'news';

  /// Lời khuyên lối sống gần nhất (`POST /v1/agent/lifestyle`) — xoá khi đăng xuất.
  static const String lifestyle = 'lifestyle';

  /// Kết quả tìm thuốc, tính theo từ khoá đã chuẩn hoá (`GET /search-medicine`).
  static const String searchPrefix = 'search.';

  static String search(String normalizedQuery) =>
      '$searchPrefix${normalizedQuery.toLowerCase()}';
}
