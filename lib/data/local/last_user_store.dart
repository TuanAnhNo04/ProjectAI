import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Nhớ id người dùng đăng nhập gần nhất trên máy (không phải bí mật), để phát
/// hiện đổi tài khoản trên thiết bị dùng chung và xoá dữ liệu sức khoẻ cục bộ
/// của người trước (xem `AuthController`).
abstract interface class LastUserStore {
  Future<String?> read();
  Future<void> write(String userId);
  Future<void> clear();
}

class SecureLastUserStore implements LastUserStore {
  SecureLastUserStore({FlutterSecureStorage? storage})
      : _storage = storage ??
            const FlutterSecureStorage(
              aOptions: AndroidOptions(encryptedSharedPreferences: true),
            );

  final FlutterSecureStorage _storage;
  static const String _key = 'medassist.last_user_id';

  @override
  Future<String?> read() async {
    try {
      return await _storage.read(key: _key);
    } catch (_) {
      return null;
    }
  }

  @override
  Future<void> write(String userId) async {
    try {
      await _storage.write(key: _key, value: userId);
    } catch (_) {}
  }

  @override
  Future<void> clear() async {
    try {
      await _storage.delete(key: _key);
    } catch (_) {}
  }
}

class InMemoryLastUserStore implements LastUserStore {
  InMemoryLastUserStore([this.value]);
  String? value;

  @override
  Future<String?> read() async => value;

  @override
  Future<void> write(String userId) async => value = userId;

  @override
  Future<void> clear() async => value = null;
}
