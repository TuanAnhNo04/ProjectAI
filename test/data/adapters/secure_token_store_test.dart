// Day 14 — test cho tầng `data`: cache và kho token.
//
// Đây là hai chỗ **quan trọng về bảo mật** nên phải test kỹ:
// - token phải không bao giờ rò ra `null`/`''` rác, và lỗi kho không làm crash;
// - cache phải chịu được JSON hỏng (đổi phiên bản app, ghi tay).
//
// `SharedPreferences` trong `flutter test` cần `setMockInitialValues` — không có
// plugin nền tảng thật.

import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:image_processing_app/data/adapters/secure_token_store.dart';
import 'package:image_processing_app/data/local/cache_store.dart';

void main() {
  setUp(() {
    // Bắt đầu mỗi test với kho trống sạch.
    // `flutter_secure_storage` cũng là plugin nền tảng -> cần mock riêng.
    FlutterSecureStorage.setMockInitialValues(<String, String>{});
    SharedPreferences.setMockInitialValues(<String, Object>{});
  });

  group('PreferencesCacheStore', () {
    test('ghi rồi đọc lại được Map', () async {
      final store = PreferencesCacheStore();

      expect(await store.read('history'), isNull);

      await store.write('history', <String, dynamic>{'a': 1});
      expect(await store.read('history'), <String, dynamic>{'a': 1});
    });

    test('ghi xong có mốc thời gian để dựng banner ngoại tuyến', () async {
      final store = PreferencesCacheStore();
      await store.write('history', <String, dynamic>{'a': 1});

      final at = await store.cachedAt('history');
      expect(at, isNotNull);
      // Không ở tương lai -> tránh lỗi đồng hồ máy.
      expect(
          at!.isAfter(DateTime.now().add(const Duration(minutes: 1))), isFalse);
    });

    test('chưa ghi -> cachedAt = null (không bịa mốc thời gian)', () async {
      expect(await PreferencesCacheStore().cachedAt('chua-co'), isNull);
    });

    test('JSON hỏng -> coi như không có cache, KHÔNG crash', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        // Ghi thẳng chuỗi rác, bỏ qua lớp bọc JSON.
        'cache.hong': 'KHONG-PHOI-JSON{{{',
      });

      final store = PreferencesCacheStore();
      expect(await store.read('hong'), isNull);
      expect(await store.cachedAt('hong'), isNull);
    });

    test('remove() chỉ xoá đúng khoá của mình', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        'setting_khac': 'giu-nguyen',
        'cache.history': '{"v":{"a":1},"at":"2026-10-05T10:00:00.000"}',
      });

      final store = PreferencesCacheStore();
      await store.remove('history');

      expect(await store.read('history'), isNull);
      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getString('setting_khac'), 'giu-nguyen');
    });

    test('clear() xoá mọi khoá `cache.` nhưng giữ setting khác', () async {
      SharedPreferences.setMockInitialValues(<String, Object>{
        'setting_khac': 'giu-nguyen',
        'cache.a': '{"v":1,"at":"2026-10-05T10:00:00.000"}',
        'cache.b': '{"v":2,"at":"2026-10-05T10:00:00.000"}',
      });

      final store = PreferencesCacheStore();
      await store.clear();

      expect(await store.read('a'), isNull);
      expect(await store.read('b'), isNull);
      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getString('setting_khac'), 'giu-nguyen');
    });

    test('khoá được tiền tố `cache.` (không đụng setting app)', () {
      expect(PreferencesCacheStore.keyPrefix, 'cache.');
    });
  });

  group('CacheKeys', () {
    test('history là hằng cố định', () {
      expect(CacheKeys.history, 'history');
    });

    test('search() chuẩn hoá về chữ thường để không trùng khoá', () {
      expect(CacheKeys.search('Paracetamol'), 'search.paracetamol');
      expect(CacheKeys.search('paracetamol'), CacheKeys.search('PARACETAMOL'));
    });
  });

  group('SecureTokenStore', () {
    test('ghi rồi đọc lại được token', () async {
      final store = SecureTokenStore();

      expect(await store.read(), isNull);
      await store.write('tok-123');
      expect(await store.read(), 'tok-123');
    });

    test('token rỗng -> coi như chưa đăng nhập (không gửi "Bearer " rỗng)',
        () async {
      final store = SecureTokenStore();

      await store.write('tok');
      await store.write('');
      expect(await store.read(), isNull);

      await store.write('tok');
      await store.write(null);
      expect(await store.read(), isNull);
    });

    test('clear() xoá token', () async {
      final store = SecureTokenStore();
      await store.write('tok');
      await store.clear();
      expect(await store.read(), isNull);
    });

    test('khoá token có tiền tố riêng, không trùng cache', () {
      expect(SecureTokenStore.tokenKey, 'medassist.auth_token');
      expect(
          SecureTokenStore.tokenKey.startsWith(PreferencesCacheStore.keyPrefix),
          isFalse,
          reason: 'token TUYỆT ĐỐI không được nằm trong kho không mã hoá');
    });
  });

  group('SecureTokenStore — kho hỏng', () {
    // Dùng storage giả luôn ném để mô phỏng kho bí mật hỏng (đổi máy, hỏng
    // keystore). Nguyên tắc: nuốt lỗi, không crash — app phải quay về màn Login.
    test('read/write lỗi -> trả null / bỏ qua, KHÔNG ném ra ngoài', () async {
      final store = SecureTokenStore(storage: _ThrowingStorage());

      expect(await store.read(), isNull);
      await store.write('tok'); // không được ném
      await store.clear(); // không được ném
    });
  });
}

/// Storage luôn ném lỗi — dùng để kiểm tra đường lỗi của [SecureTokenStore].
class _ThrowingStorage implements FlutterSecureStorage {
  @override
  dynamic noSuchMethod(Invocation invocation) =>
      throw StateError('kho bi mat hong');
}
