import 'dart:convert';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import 'package:image_processing_app/data/models/health_profile.dart';

/// Kho "két sắt" không đọc được (hỏng/khoá/lỗi giải mã).
///
/// **Fail-closed**: tuyệt đối KHÔNG coi là "trống" rồi ghi đè — UI hiện
/// "không đọc được dữ liệu an toàn" và chỉ cho phép xoá hẳn ([HealthVault.wipe]).
class VaultUnreadable implements Exception {
  const VaultUnreadable();

  @override
  String toString() => 'VaultUnreadable';
}

/// Giới hạn độ dài để dữ liệu lưu luôn nhỏ và có cấu trúc.
const int kVaultMaxEpisodes = 100;
const int kVaultMaxTerms = 5;
const int kVaultTermLen = 60;
const int kVaultMaxMeds = 10;
const int kVaultMedLen = 100;
const int kVaultProfileItems = 100;
const int kVaultProfileItemLen = 100;

/// Một lần ghi nhớ triệu chứng — KHÔNG chứa nội dung câu hỏi gốc.
class SymptomEpisode {
  const SymptomEpisode({
    required this.at,
    this.terms = const [],
    this.meds = const [],
    this.recurring = false,
  });

  final DateTime at;
  final List<String> terms;
  final List<String> meds;
  final bool recurring;

  /// Bản đã cắt độ dài/số lượng theo giới hạn của kho.
  SymptomEpisode sanitized() => SymptomEpisode(
        at: at,
        terms: _clean(terms, kVaultMaxTerms, kVaultTermLen),
        meds: _clean(meds, kVaultMaxMeds, kVaultMedLen),
        recurring: recurring,
      );

  Map<String, dynamic> toJson() => {
        'at': at.toIso8601String(),
        'terms': terms,
        'meds': meds,
        'recurring': recurring,
      };

  static SymptomEpisode? fromJson(Object? raw) {
    if (raw is! Map) return null;
    final at = DateTime.tryParse(raw['at'] as String? ?? '');
    if (at == null) return null;
    return SymptomEpisode(
      at: at,
      terms: _strings(raw['terms']),
      meds: _strings(raw['meds']),
      recurring: raw['recurring'] == true,
    ).sanitized();
  }
}

List<String> _strings(Object? v) => v is List
    ? [
        for (final e in v)
          if (e is String) e
      ]
    : const [];

List<String> _clean(List<String> items, int maxItems, int maxLen) => items
    .map((e) => e.trim())
    .where((e) => e.isNotEmpty)
    .map((e) => e.length > maxLen ? e.substring(0, maxLen) : e)
    .take(maxItems)
    .toList();

/// Các triệu chứng khác nhau trong [days] ngày gần nhất, mới nhất trước, tối đa [max].
List<String> recentSymptomTerms(
  List<SymptomEpisode> episodes,
  DateTime now, {
  int days = 14,
  int max = 5,
}) {
  final cutoff = now.subtract(Duration(days: days));
  final sorted = [...episodes]..sort((a, b) => b.at.compareTo(a.at));
  final seen = <String>{};
  final out = <String>[];
  for (final e in sorted) {
    if (e.at.isBefore(cutoff)) continue;
    for (final t in e.terms) {
      if (seen.add(t.toLowerCase())) {
        out.add(t);
        if (out.length >= max) return out;
      }
    }
  }
  return out;
}

/// Kho mã hoá trên máy cho dữ liệu nhạy cảm của trợ lý: hồ sơ sức khoẻ + nhật
/// ký triệu chứng. Chỉ nằm trên máy; không bao giờ gửi đi nguyên vẹn.
abstract interface class HealthVault {
  /// `null` = chưa có hồ sơ. Ném [VaultUnreadable] nếu không đọc được.
  Future<HealthProfile?> readProfile();
  Future<void> writeProfile(HealthProfile profile);
  Future<void> deleteProfile();

  /// Mới nhất trước.
  Future<List<SymptomEpisode>> readEpisodes();
  Future<void> appendEpisode(SymptomEpisode episode);

  /// Xoá mọi lần ghi có đúng mốc [at]. Trả `true` nếu có xoá.
  Future<bool> deleteEpisode(DateTime at);
  Future<void> clearEpisodes();

  /// Xoá sạch (cả khi dữ liệu hỏng) — đường phục hồi duy nhất cho kho hỏng.
  Future<void> wipe();
}

/// Logic chung (JSON + giới hạn) — lớp con chỉ cung cấp đọc/ghi chuỗi.
abstract class _KvHealthVault implements HealthVault {
  static const String profileKey = 'medassist.vault.profile';
  static const String episodesKey = 'medassist.vault.episodes';

  Future<String?> _get(String key);
  Future<void> _put(String key, String value);
  Future<void> _del(String key);

  Future<String?> _safeGet(String key) async {
    try {
      return await _get(key);
    } on Object {
      throw const VaultUnreadable();
    }
  }

  Future<void> _safePut(String key, String value) async {
    try {
      await _put(key, value);
    } on Object {
      throw const VaultUnreadable();
    }
  }

  Future<void> _safeDel(String key) async {
    try {
      await _del(key);
    } on Object {
      throw const VaultUnreadable();
    }
  }

  // --- Profile ---------------------------------------------------------------

  @override
  Future<HealthProfile?> readProfile() async {
    final raw = await _safeGet(profileKey);
    if (raw == null || raw.isEmpty) return null;
    try {
      final m = jsonDecode(raw) as Map<String, dynamic>;
      return HealthProfile(
        allergies: _clean(
            _strings(m['allergies']), kVaultProfileItems, kVaultProfileItemLen),
        conditions: _clean(_strings(m['conditions']), kVaultProfileItems,
            kVaultProfileItemLen),
        currentMedications: _clean(_strings(m['current_medications']),
            kVaultProfileItems, kVaultProfileItemLen),
        consentGivenAt:
            DateTime.tryParse(m['consent_given_at'] as String? ?? ''),
      );
    } on Object {
      throw const VaultUnreadable();
    }
  }

  @override
  Future<void> writeProfile(HealthProfile p) async {
    await readProfile(); // dữ liệu hiện có hỏng -> ném, KHÔNG ghi đè
    await _safePut(
      profileKey,
      jsonEncode({
        'allergies':
            _clean(p.allergies, kVaultProfileItems, kVaultProfileItemLen),
        'conditions':
            _clean(p.conditions, kVaultProfileItems, kVaultProfileItemLen),
        'current_medications': _clean(
            p.currentMedications, kVaultProfileItems, kVaultProfileItemLen),
        'consent_given_at': p.consentGivenAt?.toIso8601String(),
      }),
    );
  }

  @override
  Future<void> deleteProfile() => _safeDel(profileKey);

  // --- Episodes (lưu cũ -> mới) ----------------------------------------------

  Future<List<SymptomEpisode>> _load() async {
    final raw = await _safeGet(episodesKey);
    if (raw == null || raw.isEmpty) return [];
    try {
      final list = jsonDecode(raw) as List;
      final out = <SymptomEpisode>[];
      for (final e in list) {
        final ep = SymptomEpisode.fromJson(e);
        if (ep != null) out.add(ep);
      }
      return out;
    } on Object {
      throw const VaultUnreadable();
    }
  }

  Future<void> _save(List<SymptomEpisode> eps) async {
    final capped = eps.length > kVaultMaxEpisodes
        ? eps.sublist(eps.length - kVaultMaxEpisodes)
        : eps;
    await _safePut(
        episodesKey, jsonEncode([for (final e in capped) e.toJson()]));
  }

  @override
  Future<List<SymptomEpisode>> readEpisodes() async {
    final eps = await _load();
    return eps.reversed.toList();
  }

  @override
  Future<void> appendEpisode(SymptomEpisode episode) async {
    final clean = episode.sanitized();
    if (clean.terms.isEmpty) return;
    final eps = await _load();
    await _save([...eps, clean]);
  }

  @override
  Future<bool> deleteEpisode(DateTime at) async {
    final eps = await _load();
    final kept = eps.where((e) => !e.at.isAtSameMomentAs(at)).toList();
    if (kept.length == eps.length) return false;
    await _save(kept);
    return true;
  }

  @override
  Future<void> clearEpisodes() => _safeDel(episodesKey);

  @override
  Future<void> wipe() async {
    var failed = false;
    for (final k in [profileKey, episodesKey]) {
      try {
        await _del(k);
      } on Object {
        failed = true;
      }
    }
    if (failed) throw const VaultUnreadable();
  }
}

/// Kho thật: `flutter_secure_storage` (Keystore/Keychain/DPAPI), cùng tuỳ chọn
/// với `SecureAppLockStore`.
///
/// Giới hạn: lịch nhắc/tuân thủ/đơn thuốc vẫn nằm trong SQLite **chưa mã hoá**.
/// Khuyến nghị SQLCipher sau này (cần chủ dự án duyệt thêm gói).
class SecureHealthVault extends _KvHealthVault {
  SecureHealthVault({FlutterSecureStorage? storage})
      : _storage = storage ??
            const FlutterSecureStorage(
              aOptions: AndroidOptions(encryptedSharedPreferences: true),
              iOptions: IOSOptions(
                accessibility: KeychainAccessibility.first_unlock,
              ),
            );

  final FlutterSecureStorage _storage;

  @override
  Future<String?> _get(String key) => _storage.read(key: key);

  @override
  Future<void> _put(String key, String value) =>
      _storage.write(key: key, value: value);

  @override
  Future<void> _del(String key) => _storage.delete(key: key);
}

/// Bản trong RAM — dùng cho test.
class InMemoryHealthVault extends _KvHealthVault {
  final Map<String, String> data = {};

  /// Test: giả lập kho không đọc/ghi được.
  bool unreadable = false;

  @override
  Future<String?> _get(String key) async {
    if (unreadable) throw StateError('unreadable');
    return data[key];
  }

  @override
  Future<void> _put(String key, String value) async {
    if (unreadable) throw StateError('unreadable');
    data[key] = value;
  }

  @override
  Future<void> _del(String key) async {
    if (unreadable) throw StateError('unreadable');
    data.remove(key);
  }
}
