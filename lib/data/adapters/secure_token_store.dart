import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';

/// Lưu token phiên vào kho bí mật của hệ điều hành (Day 11).
///
/// Vì sao **không** dùng `shared_preferences`: file preferences nằm thẳng trên
/// đĩa và bất kỳ app nào cũng đọc được. Token là thứ tương đương mật khẩu nên
/// phải nằm trong Keychain (iOS/macOS) hoặc Keystore/DPAPI (Android/Windows).
///
/// ⚠️ Bên trên **web** plugin này chỉ dùng `localStorage` (không mã hoá) — vì
/// vậy adapter này **không dùng cho web build**; xem `isSecurelyAvailable`.
class SecureTokenStore implements TokenStore {
  SecureTokenStore({FlutterSecureStorage? storage})
      : _storage = storage ??
            const FlutterSecureStorage(
              aOptions: AndroidOptions(encryptedSharedPreferences: true),
              iOptions: IOSOptions(
                accessibility: KeychainAccessibility.first_unlock,
              ),
            );

  final FlutterSecureStorage _storage;

  /// AndroidOptions dùng `encryptedSharedPreferences: true` nên khoá được mã
  /// hoá bằng keystore phần cứng, không phải AES tự quản lý trong prefs.
  static const String tokenKey = 'medassist.auth_token';
  static const String refreshKey = 'medassist.refresh_token';

  /// `false` trên web — nơi cần adapter khác (in-memory) hoặc chỉ đăng nhập.
  static bool get isSecurelyAvailable => !kIsWeb;

  @override
  Future<String?> read() async {
    try {
      final value = await _storage.read(key: tokenKey);
      // Token rỗng coi như chưa đăng nhập — tránh gửi "Bearer " rỗng.
      return (value == null || value.isEmpty) ? null : value;
    } catch (_) {
      // Kho bí mật hỏng (đổi máy, khôi phục backup...) — coi như chưa đăng nhập
      // để app chạy tiếp bằng màn Login thay vì crash.
      return null;
    }
  }

  @override
  Future<void> write(String? token) async {
    try {
      if (token == null || token.isEmpty) {
        await _storage.delete(key: tokenKey);
      } else {
        await _storage.write(key: tokenKey, value: token);
      }
    } catch (_) {
      // Xem ghi chú ở read(): nuốt lỗi để UI không mất phiên đang dùng.
    }
  }

  @override
  Future<String?> readRefresh() async {
    try {
      final value = await _storage.read(key: refreshKey);
      return (value == null || value.isEmpty) ? null : value;
    } catch (_) {
      return null;
    }
  }

  @override
  Future<void> writeRefresh(String? token) async {
    try {
      if (token == null || token.isEmpty) {
        await _storage.delete(key: refreshKey);
      } else {
        await _storage.write(key: refreshKey, value: token);
      }
    } catch (_) {
      // Như write(): không để lỗi kho bí mật làm sập luồng đăng nhập.
    }
  }

  @override
  Future<void> clear() async {
    await write(null);
    await writeRefresh(null);
  }
}
