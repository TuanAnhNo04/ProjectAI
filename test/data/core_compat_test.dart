// Tương thích `medassist-core`: model đọc cả hai kiểu JSON, làm mới phiên khi
// access token hết hạn, thử lại GET khi mất kết nối, xoá tài khoản.
//
// Toàn bộ dữ liệu là GIẢ. Dùng `FakeHttpAdapter` nên không chạm mạng.

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/config/api_paths.dart';
import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/adapters/remote_medicine_catalog.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/data/models/user.dart';

import 'adapters/fake_http.dart';

const Map<String, Object?> _coreSession = {
  'user': {'id': 7, 'username': 'annie', 'email': 'annie@e.co'},
  'token': {
    'access_token': 'access-1',
    'token_type': 'bearer',
    'expires_in': 900,
    'refresh_token': 'refresh-1',
    'refresh_expires_in': 604800,
  },
};

Map<String, Object?> _rotated(int n) => {
      'user': {'id': 7, 'username': 'annie'},
      'token': {
        'access_token': 'access-$n',
        'expires_in': 900,
        'refresh_token': 'refresh-$n',
      },
    };

/// Client có refresher + trả lời theo [responder]; không chờ giữa các lần thử.
({ApiClient client, FakeHttpAdapter adapter, List<int> unauthorized}) _client(
  ResponseBody Function(RequestOptions) responder, {
  TokenProvider? tokenProvider,
  TokenRefresher? refresher,
}) {
  final adapter = FakeHttpAdapter(responder);
  final unauthorized = <int>[];
  final client = ApiClient(
    dio: Dio()..httpClientAdapter = adapter,
    tokenProvider: tokenProvider,
    tokenRefresher: refresher,
    onUnauthorized: () => unauthorized.add(1),
    retryDelays: const [Duration.zero, Duration.zero],
    logTraffic: false,
  );
  return (client: client, adapter: adapter, unauthorized: unauthorized);
}

void main() {
  group('model đọc JSON của medassist-core', () {
    test('Medicine: khoá snake_case của core', () {
      final m = Medicine.fromJson(const {
        'name': 'Thuoc Gia',
        'composition': 'Hoat chat gia 100mg',
        'uses': 'Giam dau gia',
        'side_effects': 'Buon ngu gia',
        'manufacturer': 'Hang Gia',
        'excellent_review_percent': 80.0,
      });
      expect(m.name, 'Thuoc Gia');
      expect(m.composition, 'Hoat chat gia 100mg');
      expect(m.sideEffects, 'Buon ngu gia');
      expect(m.excellentReviewPercent, 80);
    });

    test('Medicine: khoá Flask vẫn đọc được và ưu tiên hơn khoá core', () {
      final m = Medicine.fromJson(const {
        'Medicine_Name': 'Flask',
        'name': 'Core',
      });
      expect(m.name, 'Flask');
    });

    test('AuthSession: token lồng + refresh token', () {
      final s = AuthSession.fromJson(_coreSession);
      expect(s.token, 'access-1');
      expect(s.refreshToken, 'refresh-1');
      expect(s.user.username, 'annie');
      expect(s.isAuthenticated, isTrue);
    });

    test('AuthSession: token chuỗi kiểu cũ vẫn đọc được', () {
      final s = AuthSession.fromJson(const {'token': 'abc'});
      expect(s.token, 'abc');
      expect(s.refreshToken, isNull);
    });

    test('AuthSession.toString không lộ token', () {
      final text = AuthSession.fromJson(_coreSession).toString();
      expect(text, isNot(contains('access-1')));
      expect(text, isNot(contains('refresh-1')));
    });
  });

  group('ApiPaths', () {
    test('core dùng /v1, flask giữ đường dẫn cũ', () {
      expect(ApiPaths.core.login, '/v1/auth/login');
      expect(ApiPaths.core.search, '/v1/search');
      expect(ApiPaths.core.catalog, '/v1/medicines');
      expect(ApiPaths.flask.login, '/login');
      expect(ApiPaths.flask.catalog, isNull);
    });

    test('backend lạ rơi về core', () {
      expect(ApiPaths.forBackend('xyz'), same(ApiPaths.core));
      expect(ApiPaths.forBackend('flask'), same(ApiPaths.flask));
    });
  });

  group('RemoteAuthRepository với core', () {
    test('login lưu cả access lẫn refresh token', () async {
      final f = fakeClient((_) => jsonBody(_coreSession));
      final store = InMemoryTokenStore();
      final auth = RemoteAuthRepository(f.client, store: store);

      await auth.login('annie', 'matkhau123');

      expect(await store.read(), 'access-1');
      expect(await store.readRefresh(), 'refresh-1');
    });

    test('refreshSession xoay token và lưu cặp mới', () async {
      final f = fakeClient((_) => jsonBody(_rotated(2)));
      final store = InMemoryTokenStore();
      await store.write('access-1');
      await store.writeRefresh('refresh-1');
      final auth = RemoteAuthRepository(f.client, store: store);

      expect(await auth.refreshSession(), isTrue);

      expect(f.adapter.last.uri.path, '/v1/auth/refresh');
      expect(f.adapter.last.data, {'refresh_token': 'refresh-1'});
      expect(await store.read(), 'access-2');
      expect(await store.readRefresh(), 'refresh-2');
    });

    test('refreshSession SINGLE-FLIGHT: 5 lần gọi đồng thời chỉ 1 request',
        () async {
      final f = fakeClient((_) => jsonBody(_rotated(2)));
      final store = InMemoryTokenStore();
      await store.writeRefresh('refresh-1');
      final auth = RemoteAuthRepository(f.client, store: store);

      final results = await Future.wait(
        List.generate(5, (_) => auth.refreshSession()),
      );

      expect(results, everyElement(isTrue));
      expect(f.adapter.sent, hasLength(1));
    });

    test('refreshSession: không có refresh token -> false, không gọi mạng',
        () async {
      final f = fakeClient((_) => jsonBody(_rotated(2)));
      final auth = RemoteAuthRepository(f.client);

      expect(await auth.refreshSession(), isFalse);
      expect(f.adapter.sent, isEmpty);
    });

    test('refreshSession: server từ chối (401) -> false, không ném', () async {
      final f = fakeClient((_) => jsonBody({'code': 'unauthorized'}, 401));
      final store = InMemoryTokenStore();
      await store.writeRefresh('het-han');
      final auth = RemoteAuthRepository(f.client, store: store);

      expect(await auth.refreshSession(), isFalse);
    });

    test('currentUser đọc tên từ /v1/auth/me', () async {
      final f = fakeClient(
        (_) => jsonBody({'id': 7, 'username': 'annie', 'email': 'a@e.co'}),
      );
      final auth = RemoteAuthRepository(f.client);

      final user = await auth.currentUser();

      expect(f.adapter.last.uri.path, '/v1/auth/me');
      expect(user?.username, 'annie');
    });

    test('currentUser: lỗi mạng/401 -> null, không ném', () async {
      final f = fakeClient((_) => jsonBody({'code': 'unauthorized'}, 401));
      expect(await RemoteAuthRepository(f.client).currentUser(), isNull);
    });

    test(
        'currentUser: backend không có endpoint (mePath null) -> null, không gọi mạng',
        () async {
      final f = fakeClient((_) => jsonBody({'username': 'x'}));
      final auth = RemoteAuthRepository(f.client, mePath: null);

      expect(await auth.currentUser(), isNull);
      expect(f.adapter.sent, isEmpty);
    });

    test('logout gửi refresh token để server thu hồi rồi xoá cục bộ', () async {
      final f = fakeClient((_) => jsonBody(null, 204));
      final store = InMemoryTokenStore();
      await store.write('access-1');
      await store.writeRefresh('refresh-1');
      final auth = RemoteAuthRepository(f.client, store: store);

      await auth.logout();

      expect(f.adapter.last.data, {'refresh_token': 'refresh-1'});
      expect(await store.read(), isNull);
      expect(await store.readRefresh(), isNull);
    });

    test('deleteAccount gửi mật khẩu và xoá token cục bộ khi thành công',
        () async {
      final f = fakeClient((_) => jsonBody(null, 204));
      final store = InMemoryTokenStore();
      await store.write('access-1');
      await store.writeRefresh('refresh-1');
      final auth = RemoteAuthRepository(f.client, store: store);

      await auth.deleteAccount('matkhau123');

      expect(f.adapter.last.uri.path, '/v1/auth/delete-account');
      expect(f.adapter.last.data, {'password': 'matkhau123'});
      expect(await store.read(), isNull);
      expect(await store.readRefresh(), isNull);
    });

    test('deleteAccount sai mật khẩu KHÔNG làm mới phiên, KHÔNG đăng xuất',
        () async {
      var refreshCalls = 0;
      final c = _client(
        (_) => jsonBody({'code': 'unauthorized'}, 401),
        tokenProvider: () async => 'access-1',
        refresher: () async {
          refreshCalls++;
          return true;
        },
      );
      final auth = RemoteAuthRepository(c.client);

      await expectLater(
        auth.deleteAccount('sai'),
        throwsA(
          isA<AppException>()
              .having((e) => e.message, 'message', 'Mật khẩu không đúng.'),
        ),
      );

      expect(refreshCalls, 0);
      expect(c.unauthorized, isEmpty);
      expect(c.adapter.sent, hasLength(1));
    });

    test('deleteAccount sai mật khẩu -> ném, KHÔNG xoá token', () async {
      final f = fakeClient((_) => jsonBody({'code': 'unauthorized'}, 401));
      final store = InMemoryTokenStore();
      await store.write('access-1');
      final auth = RemoteAuthRepository(f.client, store: store);

      await expectLater(
        auth.deleteAccount('sai'),
        throwsA(isA<AppException>()),
      );
      expect(await store.read(), 'access-1');
    });
  });

  group('ApiClient: làm mới phiên khi 401', () {
    test('401 -> làm mới -> gọi lại 1 lần với token mới, không đăng xuất',
        () async {
      var token = 'cu';
      final c = _client(
        (o) => o.headers['Authorization'] == 'Bearer moi'
            ? jsonBody({'ok': true})
            : jsonBody({'code': 'unauthorized'}, 401),
        tokenProvider: () async => token,
        refresher: () async {
          token = 'moi';
          return true;
        },
      );

      final response = await c.client.get('/v1/history');

      expect(response.statusCode, 200);
      expect(c.adapter.sent, hasLength(2));
      expect(c.unauthorized, isEmpty);
    });

    test('làm mới thất bại -> báo hết phiên đúng 1 lần', () async {
      final c = _client(
        (_) => jsonBody({'code': 'unauthorized'}, 401),
        tokenProvider: () async => 'cu',
        refresher: () async => false,
      );

      await expectLater(
        c.client.get('/v1/history'),
        throwsA(isA<AppException>()),
      );
      expect(c.unauthorized, hasLength(1));
      expect(c.adapter.sent, hasLength(1));
    });

    test('vẫn 401 sau khi đã làm mới -> không lặp vô hạn', () async {
      var refreshCalls = 0;
      final c = _client(
        (_) => jsonBody({'code': 'unauthorized'}, 401),
        tokenProvider: () async => 'x',
        refresher: () async {
          refreshCalls++;
          return true;
        },
      );

      await expectLater(
        c.client.get('/v1/history'),
        throwsA(isA<AppException>()),
      );
      expect(refreshCalls, 1);
      expect(c.adapter.sent, hasLength(2));
      expect(c.unauthorized, hasLength(1));
    });

    test('endpoint công khai (login/refresh) KHÔNG kích hoạt làm mới',
        () async {
      var refreshCalls = 0;
      final c = _client(
        (_) => jsonBody({'code': 'unauthorized'}, 401),
        refresher: () async {
          refreshCalls++;
          return true;
        },
      );

      await expectLater(
        c.client.post('/v1/auth/login', data: {'username': 'a'}),
        throwsA(isA<AppException>()),
      );
      await expectLater(
        c.client.post('/v1/auth/refresh', data: {'refresh_token': 'x'}),
        throwsA(isA<AppException>()),
      );
      expect(refreshCalls, 0);
      expect(c.unauthorized, isEmpty);
    });
  });

  group('ApiClient: thử lại khi lỗi tạm thời', () {
    test('GET gặp 503 hai lần rồi thành công -> trả kết quả', () async {
      var calls = 0;
      final c = _client(
        (_) => ++calls < 3
            ? jsonBody({'code': 'unavailable'}, 503)
            : jsonBody({'ok': true}),
      );

      final response = await c.client.get('/v1/search');

      expect(response.statusCode, 200);
      expect(calls, 3);
    });

    test('GET lỗi mãi -> dừng sau số lần thử tối đa rồi ném', () async {
      final c = _client((_) => jsonBody({'code': 'unavailable'}, 503));

      await expectLater(
        c.client.get('/v1/search'),
        throwsA(isA<AppException>()),
      );
      // 1 lần gốc + 2 lần thử lại.
      expect(c.adapter.sent, hasLength(3));
    });

    test('POST KHÔNG được thử lại (tránh tạo/xoá hai lần)', () async {
      final c = _client((_) => jsonBody({'code': 'unavailable'}, 503));

      await expectLater(
        c.client.post('/v1/history', data: {'a': 1}),
        throwsA(isA<AppException>()),
      );
      expect(c.adapter.sent, hasLength(1));
    });

    test('lỗi 4xx/500 không được thử lại', () async {
      final c400 = _client((_) => jsonBody({'code': 'bad'}, 400));
      await expectLater(
        c400.client.get('/v1/search'),
        throwsA(isA<AppException>()),
      );
      expect(c400.adapter.sent, hasLength(1));

      final c500 = _client((_) => jsonBody({'code': 'boom'}, 500));
      await expectLater(
        c500.client.get('/v1/search'),
        throwsA(isA<AppException>()),
      );
      expect(c500.adapter.sent, hasLength(1));
    });
  });

  group('RemoteMedicineCatalog', () {
    test('tải đủ các trang rồi dừng', () async {
      final all = [
        for (var i = 0; i < 5; i++) {'name': 'Thuoc $i'},
      ];
      final f = fakeClient((o) {
        final offset = int.parse('${o.uri.queryParameters['offset']}');
        final limit = int.parse('${o.uri.queryParameters['limit']}');
        return jsonBody({
          'results': all.skip(offset).take(limit).toList(),
          'total': all.length,
        });
      });

      final list = await RemoteMedicineCatalog(
        f.client,
        path: '/v1/medicines',
        pageSize: 2,
      ).fetchAll();

      expect(list.map((m) => m.name), [for (var i = 0; i < 5; i++) 'Thuoc $i']);
      expect(f.adapter.sent, hasLength(3));
    });

    test('server trả trang rỗng dù total lớn -> không lặp vô hạn', () async {
      final f = fakeClient((_) => jsonBody({'results': [], 'total': 999}));

      final list = await RemoteMedicineCatalog(f.client, path: '/v1/medicines')
          .fetchAll();

      expect(list, isEmpty);
      expect(f.adapter.sent, hasLength(1));
    });
  });
}
