// Day 9 — test cho tầng DI/state (`lib/core/providers.dart`).
//
// Đây là lớp "bột" của Riverpod: logic nẩy trong AsyncNotifier nên test được
// bằng cách override adapter bằng fake — **không chạm socket**, không cần
// server Flask thật, và kiểm tra được cả các trường hợp lỗi mà trước Day 9
// không test nổi (UI gọi mạng trực tiếp).

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/data/models/user.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/widgets/offline_banner.dart';
import 'package:image_processing_app/domain/ports/auth_repository.dart';
import 'package:image_processing_app/domain/ports/medicine_analyzer.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';

/// ProviderContainer cho test — không cần widget, chạy trên Dart VM thuần.
ProviderContainer _container({
  AuthRepository? auth,
  MedicineRepository? medicine,
  MedicineAnalyzer? analyzer,
  CacheStore? cache,
}) {
  final container = ProviderContainer(
    overrides: [
      if (auth != null) authRepositoryProvider.overrideWithValue(auth),
      if (medicine != null)
        medicineRepositoryProvider.overrideWithValue(medicine),
      if (analyzer != null) analyzerProvider.overrideWithValue(analyzer),
      // Cache mặc định dùng `shared_preferences` — thứ **không** tồn tại trong
      // test, nên luôn thay bằng bản trong RAM cho test chạy độc lập.
      cacheStoreProvider.overrideWithValue(cache ?? InMemoryCacheStore()),
    ],
  );
  addTearDown(container.dispose);
  return container;
}

class _FakeAuthRepository implements AuthRepository {
  _FakeAuthRepository({this.fail = false, this.token = 'tok', this.me});

  final bool fail;
  final String? token;

  /// Người dùng mà `/me` trả về (null = backend không có endpoint / lỗi).
  final User? me;

  @override
  Future<AuthSession> login(String username, String password) async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.unauthorized,
        message: 'Sai tài khoản.',
      );
    }
    return AuthSession(user: User(username: username), token: 'tok');
  }

  @override
  Future<AuthSession> register(String u, String p, String e) async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.conflict,
        message: 'Tên đã tồn tại.',
      );
    }
    return AuthSession(user: User(username: u), token: 'tok');
  }

  @override
  Future<void> logout() async {}

  @override
  Future<String?> currentToken() async => token;

  @override
  Future<User?> currentUser() async => me;

  @override
  Future<bool> refreshSession() async => false;

  @override
  Future<void> deleteAccount(String password) async {
    if (password == 'sai') {
      throw const AppException(
        kind: AppExceptionKind.unauthorized,
        message: 'Mật khẩu không đúng.',
      );
    }
  }
}

class _FakeMedicineRepository implements MedicineRepository {
  _FakeMedicineRepository({this.fail = false});

  final bool fail;
  final List<int> deleted = <int>[];

  @override
  Future<MedicineSearchResponse> search(String name) async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.network,
        message: 'Mất mạng.',
      );
    }
    return MedicineSearchResponse(
      results: [Medicine(name: 'A $name', excellentReviewPercent: 10)],
      query: name,
      total: 1,
    );
  }

  @override
  Future<List<HistoryItem>> history() async => const [
        HistoryItem(id: 1, filepath: 'a.jpg'),
        HistoryItem(id: 2, filepath: 'b.jpg'),
      ];

  @override
  Future<void> deleteHistory(int id) async => deleted.add(id);
}

class _FakeAnalyzer implements MedicineAnalyzer {
  _FakeAnalyzer({this.fail = false});

  final bool fail;

  @override
  String get capability => 'fake-analyzer';

  @override
  Future<AnalysisResult> analyze(String imagePath) async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.parsing,
        message: 'Ảnh không hợp lệ.',
      );
    }
    return const AnalysisResult(medicines: [Medicine(name: 'Thuốc A')]);
  }
}

/// Repository **luôn** ném lỗi mạng — mô phỏng "đã tắt wifi".
class _OfflineMedicineRepository implements MedicineRepository {
  @override
  Future<MedicineSearchResponse> search(String name) async {
    throw const AppException(
      kind: AppExceptionKind.network,
      message: 'Không kết nối được.',
    );
  }

  @override
  Future<List<HistoryItem>> history() async {
    throw const AppException(
      kind: AppExceptionKind.network,
      message: 'Không kết nối được.',
    );
  }

  @override
  Future<void> deleteHistory(int id) async {}
}

/// Container dùng chung kho cache cho 2 lượt (có mạng -> tắt wifi).
ProviderContainer _cacheContainer(
  CacheStore cache,
  MedicineRepository repo,
) {
  final container = ProviderContainer(
    overrides: [
      medicineRepositoryProvider.overrideWithValue(repo),
      cacheStoreProvider.overrideWithValue(cache),
    ],
  );
  addTearDown(container.dispose);
  return container;
}

void main() {
  group('OfflineCache (Day 12)', () {
    test('lần 1 có mạng -> cache; lần 2 tắt wifi -> vẫn có dữ liệu', () async {
      final cache = InMemoryCacheStore();

      // Lượt 1: có mạng.
      final online = _cacheContainer(cache, _FakeMedicineRepository());
      await online.read(historyControllerProvider.notifier).load();
      final fresh = online.read(historyControllerProvider).valueOrNull!;
      expect(fresh.isStale, isFalse);
      expect(fresh.items, hasLength(2));

      // Lượt 2: container mới + repository luôn lỗi = đã tắt wifi.
      final offline = _cacheContainer(cache, _OfflineMedicineRepository());
      await offline.read(historyControllerProvider.notifier).load();
      final stale = offline.read(historyControllerProvider).valueOrNull!;

      // Chính là DoD: không trắng màn hình, vẫn thấy dữ liệu + cờ cache.
      expect(stale.isStale, isTrue);
      expect(stale.items, hasLength(2));
      expect(stale.cachedAt, isNotNull);
      expect(offline.read(historyControllerProvider).hasError, isFalse);
    });

    test('tắt wifi và chưa có cache -> báo lỗi (không bịa dữ liệu)', () async {
      final container =
          _cacheContainer(InMemoryCacheStore(), _OfflineMedicineRepository());

      await container.read(historyControllerProvider.notifier).load();
      expect(container.read(historyControllerProvider).hasError, isTrue);
    });

    test('search: cache theo từ khoá, không lẫn kết quả của từ khác', () async {
      final cache = InMemoryCacheStore();

      final online = _cacheContainer(cache, _FakeMedicineRepository());
      expect(
        await online
            .read(searchControllerProvider.notifier)
            .search('paracetamol'),
        isTrue,
      );

      final offline = _cacheContainer(cache, _OfflineMedicineRepository());

      // Từ đã cache -> vẫn ra kết quả.
      expect(
        await offline
            .read(searchControllerProvider.notifier)
            .search('paracetamol'),
        isTrue,
      );
      expect(
          offline.read(searchControllerProvider).valueOrNull!.isStale, isTrue);

      // Từ **chưa** cache -> lỗi, không trả nhầm kết quả của từ khác.
      expect(
        await offline
            .read(searchControllerProvider.notifier)
            .search('ibuprofen'),
        isFalse,
      );
    });

    test('xoá cache -> offline không còn dữ liệu cũ', () async {
      final cache = InMemoryCacheStore();
      final online = _cacheContainer(cache, _FakeMedicineRepository());
      await online.read(historyControllerProvider.notifier).load();

      await cache.remove(CacheKeys.history);

      final offline = _cacheContainer(cache, _OfflineMedicineRepository());
      await offline.read(historyControllerProvider.notifier).load();
      expect(offline.read(historyControllerProvider).hasError, isTrue);
    });
  });

  group('CacheStore', () {
    test('InMemory: ghi/đọc/xoá + mốc thời gian', () async {
      final store = InMemoryCacheStore();
      expect(await store.read('k'), isNull);

      await store.write('k', {'a': 1});
      expect(await store.read('k'), {'a': 1});
      expect(store.cachedAtOf('k'), isNotNull);

      await store.remove('k');
      expect(await store.read('k'), isNull);
    });

    test('InMemory: clear() xoá hết', () async {
      final store = InMemoryCacheStore();
      await store.write('a', 1);
      await store.write('b', 2);
      await store.clear();
      expect(await store.read('a'), isNull);
      expect(await store.read('b'), isNull);
    });

    test('CacheKeys.search chuẩn hoá chữ thường -> cùng một khoá', () {
      expect(CacheKeys.search('Paracetamol'), CacheKeys.search('paracetamol'));
    });
  });

  group('OfflineBanner', () {
    test('định dạng mốc thời gian dd/MM/yyyy HH:mm', () {
      expect(
        OfflineBanner.formatCachedAt(DateTime(2026, 10, 5, 14, 30)),
        '05/10/2026 14:30',
      );
    });

    test('cachedAt = null -> không bịa thời gian', () {
      expect(OfflineBanner.formatCachedAt(null), '');
    });
  });

  group('validateCredentials (Day 11)', () {
    test('rỗi -> báo đúng ô cần điền', () {
      expect(
        validateCredentials(username: '', password: 'matkhau'),
        'Vui lòng nhập tên đăng nhập.',
      );
      expect(
        validateCredentials(username: 'an', password: ''),
        'Vui lòng nhập mật khẩu.',
      );
    });

    test('mật khẩu < 6 ký tự -> báo độ dài', () {
      expect(
        validateCredentials(username: 'an', password: '12345'),
        'Mật khẩu phải có ít nhất $kMinPasswordLength ký tự.',
      );
      // Đúng 6 là qua — ranh giới phải chính xác.
      expect(validateCredentials(username: 'an', password: '123456'), isNull);
    });

    test('email bắt buộc: rỗng / sai định dạng -> báo lỗi', () {
      expect(
        validateCredentials(
          username: 'an',
          password: 'matkhau',
          requireEmail: true,
        ),
        'Vui lòng nhập email.',
      );
      expect(
        validateCredentials(
          username: 'an',
          password: 'matkhau',
          email: 'khong-phai-email',
          requireEmail: true,
        ),
        'Email không hợp lệ.',
      );
      expect(
        validateCredentials(
          username: 'an',
          password: 'matkhau',
          email: 'an@example.com',
          requireEmail: true,
        ),
        isNull,
      );
    });

    test('không yêu cầu email thì email sai cũng qua (đăng nhập)', () {
      expect(
        validateCredentials(
          username: 'an',
          password: 'matkhau',
          email: 'khong-phai-email',
        ),
        isNull,
      );
    });

    test('khoảng trắng quanh tên đăng nhập vẫn hợp lệ', () {
      expect(
        validateCredentials(username: '   an  ', password: 'matkhau'),
        isNull,
      );
    });
  });

  group('SearchController', () {
    test('tên rỗng -> không gọi repository, trả false', () async {
      final container = _container(medicine: _FakeMedicineRepository());
      final notifier = container.read(searchControllerProvider.notifier);

      expect(await notifier.search('   '), isFalse);
      // Không có lời gọi nào, nên state vẫn là giá trị build() mặc định.
      expect(container.read(searchControllerProvider).valueOrNull, isNull);
    });

    test('tìm thành công -> state chứa kết quả', () async {
      final container = _container(medicine: _FakeMedicineRepository());
      final notifier = container.read(searchControllerProvider.notifier);

      expect(await notifier.search(' paracetamol '), isTrue);

      final state = container.read(searchControllerProvider);
      expect(state.hasError, isFalse);
      expect(state.valueOrNull!.response.query, 'paracetamol');
      expect(state.valueOrNull!.results.single.name, 'A paracetamol');
      expect(state.valueOrNull!.isStale, isFalse);
    });

    test('lỗi mạng -> state.hasError và giữ nguyên AppException', () async {
      final container =
          _container(medicine: _FakeMedicineRepository(fail: true));
      final notifier = container.read(searchControllerProvider.notifier);

      expect(await notifier.search('thuoc'), isFalse);

      final state = container.read(searchControllerProvider);
      expect(state.hasError, isTrue);
      expect(state.error, isA<AppException>());
      expect((state.error! as AppException).message, 'Mất mạng.');
    });

    test('clearError() gỡ lỗi mà không đụng adapter', () async {
      final container =
          _container(medicine: _FakeMedicineRepository(fail: true));
      final notifier = container.read(searchControllerProvider.notifier);
      await notifier.search('thuoc');

      notifier.clearError();
      expect(container.read(searchControllerProvider).hasError, isFalse);
    });
  });

  group('HistoryController', () {
    test('load() -> state chứa danh sách', () async {
      final container = _container(medicine: _FakeMedicineRepository());
      final notifier = container.read(historyControllerProvider.notifier);

      await notifier.load();

      final state = container.read(historyControllerProvider);
      expect(state.valueOrNull!.items, hasLength(2));
      expect(state.valueOrNull!.isStale, isFalse);
    });

    test('delete() gọi repository rồi loại bản ghi khỏi state', () async {
      final repo = _FakeMedicineRepository();
      final container = _container(medicine: repo);
      final notifier = container.read(historyControllerProvider.notifier);

      await notifier.load();
      expect(await notifier.delete(1), isTrue);

      expect(repo.deleted, [1]);
      final remaining = container.read(historyControllerProvider).valueOrNull!;
      expect(remaining.items.map((e) => e.id), [2]);
    });
  });
  group('AuthController', () {
    test('login thành công -> state chứa phiên có token', () async {
      final container = _container(auth: _FakeAuthRepository());
      final notifier = container.read(authControllerProvider.notifier);

      expect(await notifier.login('an', 'matkhau'), isTrue);

      final session = container.read(authControllerProvider).valueOrNull!;
      expect(session.isAuthenticated, isTrue);
      expect(session.user.username, 'an');
    });

    test('login sai -> hasError, message tiếng Việt', () async {
      final container = _container(auth: _FakeAuthRepository(fail: true));
      final notifier = container.read(authControllerProvider.notifier);

      expect(await notifier.login('an', 'sai'), isFalse);
      expect(
        (container.read(authControllerProvider).error! as AppException).message,
        'Sai tài khoản.',
      );
    });

    test('register trùng tên -> conflict', () async {
      final container = _container(auth: _FakeAuthRepository(fail: true));
      final notifier = container.read(authControllerProvider.notifier);

      expect(await notifier.register('an', 'matkhau', 'a@b.co'), isFalse);
      expect(
        container.read(authControllerProvider).error,
        isA<AppException>()
            .having((e) => e.kind, 'kind', AppExceptionKind.conflict),
      );
    });

    test('logout() -> state về null (đã đăng xuất)', () async {
      final container = _container(auth: _FakeAuthRepository());
      final notifier = container.read(authControllerProvider.notifier);
      await notifier.login('an', 'matkhau');

      await notifier.logout();
      expect(container.read(authControllerProvider).valueOrNull, isNull);
    });

    test('khôi phục phiên: có token sẵn -> có phiên đã đăng nhập', () async {
      final container = _container(
        auth: _FakeAuthRepository(), // currentToken() = 'tok'
      );
      await container.read(authControllerProvider.notifier).restoreSession();

      final session = container.read(authControllerProvider).valueOrNull;
      expect(session, isNotNull);
      expect(session!.isAuthenticated, isTrue);
      expect(session.token, 'tok');
    });

    test('khôi phục phiên: lấy được tên người dùng từ /me', () async {
      final container = _container(
        auth: _FakeAuthRepository(me: const User(username: 'annie')),
      );
      await container.read(authControllerProvider.notifier).restoreSession();

      final session = container.read(authControllerProvider).valueOrNull;
      expect(session!.user.username, 'annie');
      expect(session.isAuthenticated, isTrue);
    });

    test('khôi phục phiên: /me lỗi -> vẫn có phiên (chỉ thiếu tên)', () async {
      final container = _container(auth: _FakeAuthRepository());
      await container.read(authControllerProvider.notifier).restoreSession();

      final session = container.read(authControllerProvider).valueOrNull;
      expect(session!.isAuthenticated, isTrue);
      expect(session.user.username, isEmpty);
    });

    test('khôi phục phiên: không có token -> vẫn null', () async {
      final container = _container(auth: _FakeAuthRepository(token: null));
      await container.read(authControllerProvider.notifier).restoreSession();
      expect(container.read(authControllerProvider).valueOrNull, isNull);
    });

    test('handleUnauthorized() xoá phiên (401)', () async {
      final container = _container(auth: _FakeAuthRepository());
      final notifier = container.read(authControllerProvider.notifier);
      await notifier.login('an', 'matkhau');
      expect(container.read(authControllerProvider).hasValue, isTrue);

      await notifier.handleUnauthorized();
      expect(container.read(authControllerProvider).valueOrNull, isNull);
    });

    test('logout() xoá cache ngoại tuyến — không lộ dữ liệu cho người sau',
        () async {
      final cache = InMemoryCacheStore();
      await cache.write(CacheKeys.history, <String, dynamic>{'history': []});
      final container = _container(auth: _FakeAuthRepository(), cache: cache);
      final notifier = container.read(authControllerProvider.notifier);
      await notifier.login('an', 'matkhau');

      await notifier.logout();

      expect(await cache.read(CacheKeys.history), isNull);
    });

    test('deleteAccount thành công -> xoá cache và đưa về chưa đăng nhập',
        () async {
      final cache = InMemoryCacheStore();
      await cache.write(CacheKeys.history, <String, dynamic>{'history': []});
      final container = _container(auth: _FakeAuthRepository(), cache: cache);
      final notifier = container.read(authControllerProvider.notifier);
      await notifier.login('an', 'matkhau');

      await notifier.deleteAccount('matkhau');

      expect(await cache.read(CacheKeys.history), isNull);
      expect(container.read(authControllerProvider).valueOrNull, isNull);
    });

    test('deleteAccount sai mật khẩu -> ném lỗi, GIỮ NGUYÊN phiên và cache',
        () async {
      final cache = InMemoryCacheStore();
      await cache.write(CacheKeys.history, <String, dynamic>{'history': []});
      final container = _container(auth: _FakeAuthRepository(), cache: cache);
      final notifier = container.read(authControllerProvider.notifier);
      await notifier.login('an', 'matkhau');

      await expectLater(
        notifier.deleteAccount('sai'),
        throwsA(isA<AppException>()),
      );

      expect(await cache.read(CacheKeys.history), isNotNull);
      expect(container.read(authControllerProvider).valueOrNull, isNotNull);
    });

    test('deleteAccount mật khẩu rỗng -> validation, không gọi server',
        () async {
      final container = _container(auth: _FakeAuthRepository());
      final notifier = container.read(authControllerProvider.notifier);

      await expectLater(
        notifier.deleteAccount(''),
        throwsA(isA<AppException>()
            .having((e) => e.kind, 'kind', AppExceptionKind.validation)),
      );
    });

    test('handleUnauthorized() cũng xoá cache ngoại tuyến', () async {
      final cache = InMemoryCacheStore();
      await cache.write(CacheKeys.history, <String, dynamic>{'history': []});
      final container = _container(auth: _FakeAuthRepository(), cache: cache);
      final notifier = container.read(authControllerProvider.notifier);
      await notifier.login('an', 'matkhau');

      await notifier.handleUnauthorized();

      expect(await cache.read(CacheKeys.history), isNull);
    });
  });

  group('ApiClient qua providers', () {
    test('tokenRefresher chạy được, KHÔNG ném CircularDependencyError',
        () async {
      // Lỗi thật gặp trên máy ảo: apiClientProvider đọc authRepositoryProvider (vốn phụ
      // thuộc apiClientProvider) -> mọi lần làm mới phiên đều ném lỗi vòng phụ thuộc.
      // Kho token rỗng nên refresher trả false mà không cần gọi mạng.
      final container = ProviderContainer(
        overrides: [
          tokenStoreProvider.overrideWithValue(InMemoryTokenStore()),
          cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
        ],
      );
      addTearDown(container.dispose);

      final client = container.read(apiClientProvider);

      expect(await client.tokenRefresher!(), isFalse);
    });

    test('repository và ApiClient dùng CHUNG một bộ làm mới (single-flight)',
        () {
      final container = ProviderContainer(
        overrides: [
          tokenStoreProvider.overrideWithValue(InMemoryTokenStore()),
          cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
        ],
      );
      addTearDown(container.dispose);

      expect(
        identical(
          container.read(sessionRefresherProvider),
          container.read(sessionRefresherProvider),
        ),
        isTrue,
      );
      // Tạo repository không được ném (kiểm tra đồ thị phụ thuộc không có vòng).
      expect(() => container.read(authRepositoryProvider), returnsNormally);
    });
  });

  group('AnalysisController', () {
    test('analyze thành công -> state chứa kết quả', () async {
      final container = _container(analyzer: _FakeAnalyzer());
      final notifier = container.read(analysisControllerProvider.notifier);

      expect(await notifier.analyze('/tmp/a.jpg'), isTrue);
      final result = container.read(analysisControllerProvider).valueOrNull!;
      expect(result.medicines.single.name, 'Thuốc A');
    });

    test('analyze lỗi -> hasError', () async {
      final container = _container(analyzer: _FakeAnalyzer(fail: true));
      final notifier = container.read(analysisControllerProvider.notifier);

      expect(await notifier.analyze('/tmp/a.jpg'), isFalse);
      expect(container.read(analysisControllerProvider).hasError, isTrue);
    });

    test('reset() xoá kết quả cũ', () async {
      final container = _container(analyzer: _FakeAnalyzer());
      final notifier = container.read(analysisControllerProvider.notifier);
      await notifier.analyze('/tmp/a.jpg');

      notifier.reset();
      expect(container.read(analysisControllerProvider).valueOrNull, isNull);
    });
  });

  group('DI wiring', () {
    test('analyzerCapabilityProvider đọc từ adapter', () {
      final container = _container(analyzer: _FakeAnalyzer());
      expect(container.read(analyzerCapabilityProvider), 'fake-analyzer');
    });

    test('apiClientProvider đóng socket khi container bị dispose', () {
      final container = ProviderContainer();
      expect(container.read(apiClientProvider).dio, isNotNull);
      // `ref.onDispose(client.close)` đã được đăng ký -> không ném lỗi.
      container.dispose();
    });
  });
}
