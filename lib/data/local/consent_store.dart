import 'package:shared_preferences/shared_preferences.dart';

/// Phiên bản chính sách mà người dùng đã đồng ý. Đổi nội dung chính sách đáng kể
/// thì tăng số này để hỏi lại.
const int kPrivacyPolicyVersion = 1;

/// Lưu việc người dùng đã đồng ý chính sách dữ liệu (xem `docs/PRIVACY.md`).
///
/// Chỉ là một cờ + số phiên bản, **không** phải dữ liệu nhạy cảm nên dùng
/// `shared_preferences`; token/bí mật không bao giờ nằm ở đây.
abstract interface class ConsentStore {
  /// `true` nếu đã đồng ý đúng phiên bản chính sách hiện hành. Không bao giờ ném.
  Future<bool> hasAccepted();

  Future<void> accept();

  /// Thu hồi đồng ý (lần mở app sau sẽ hỏi lại).
  Future<void> revoke();
}

class PreferencesConsentStore implements ConsentStore {
  static const String key = 'consent.privacy_version';

  Future<SharedPreferences?> _prefs() async {
    try {
      return await SharedPreferences.getInstance();
    } on Object {
      return null;
    }
  }

  @override
  Future<bool> hasAccepted() async {
    final prefs = await _prefs();
    return prefs?.getInt(key) == kPrivacyPolicyVersion;
  }

  @override
  Future<void> accept() async {
    final prefs = await _prefs();
    await prefs?.setInt(key, kPrivacyPolicyVersion);
  }

  @override
  Future<void> revoke() async {
    final prefs = await _prefs();
    await prefs?.remove(key);
  }
}

/// Bản trong RAM — dùng cho test.
class InMemoryConsentStore implements ConsentStore {
  InMemoryConsentStore({bool accepted = false}) : _accepted = accepted;

  bool _accepted;

  @override
  Future<bool> hasAccepted() async => _accepted;

  @override
  Future<void> accept() async => _accepted = true;

  @override
  Future<void> revoke() async => _accepted = false;
}
