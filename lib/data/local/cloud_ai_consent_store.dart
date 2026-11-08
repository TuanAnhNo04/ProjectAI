import 'package:shared_preferences/shared_preferences.dart';

/// Phiên bản chính sách "AI đám mây" (xem `docs/PRIVACY.md`). Tăng số này khi
/// nội dung gửi đi thay đổi để hỏi đồng ý lại.
const int kCloudAiPolicyVersion = 1;

/// Bản ghi đồng ý dùng AI đám mây (tách riêng khỏi đồng ý hồ sơ sức khoẻ).
class CloudAiConsent {
  const CloudAiConsent({required this.grantedAt, required this.policyVersion});

  final DateTime grantedAt;
  final int policyVersion;
}

/// Lưu đồng ý gửi câu hỏi (đã xoá định danh) tới AI đám mây. Mặc định TẮT.
abstract interface class CloudAiConsentStore {
  /// Bản đồng ý còn hiệu lực (đúng [kCloudAiPolicyVersion]); `null` nếu chưa/hết hạn.
  Future<CloudAiConsent?> read();

  Future<void> grant();

  Future<void> revoke();
}

class PreferencesCloudAiConsentStore implements CloudAiConsentStore {
  static const String keyVersion = 'cloud_ai_consent.policy_version';
  static const String keyGrantedAt = 'cloud_ai_consent.granted_at';

  Future<SharedPreferences?> _prefs() async {
    try {
      return await SharedPreferences.getInstance();
    } on Object {
      return null;
    }
  }

  @override
  Future<CloudAiConsent?> read() async {
    final prefs = await _prefs();
    if (prefs == null) return null;
    if (prefs.getInt(keyVersion) != kCloudAiPolicyVersion) return null;
    final at = DateTime.tryParse(prefs.getString(keyGrantedAt) ?? '');
    if (at == null) return null;
    return CloudAiConsent(grantedAt: at, policyVersion: kCloudAiPolicyVersion);
  }

  @override
  Future<void> grant() async {
    final prefs = await _prefs();
    if (prefs == null) return;
    await prefs.setInt(keyVersion, kCloudAiPolicyVersion);
    await prefs.setString(keyGrantedAt, DateTime.now().toIso8601String());
  }

  @override
  Future<void> revoke() async {
    final prefs = await _prefs();
    if (prefs == null) return;
    await prefs.remove(keyVersion);
    await prefs.remove(keyGrantedAt);
  }
}

/// Bản trong RAM — dùng cho test.
class InMemoryCloudAiConsentStore implements CloudAiConsentStore {
  InMemoryCloudAiConsentStore({CloudAiConsent? initial}) : _value = initial;

  CloudAiConsent? _value;

  @override
  Future<CloudAiConsent?> read() async =>
      _value?.policyVersion == kCloudAiPolicyVersion ? _value : null;

  @override
  Future<void> grant() async => _value = CloudAiConsent(
        grantedAt: DateTime.now(),
        policyVersion: kCloudAiPolicyVersion,
      );

  @override
  Future<void> revoke() async => _value = null;
}
