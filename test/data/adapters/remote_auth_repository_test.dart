// Day 8 — RemoteAuthRepository nối `/login`, `/register`, `/logout`.
//
// Bảo mật là trọng tâm: mật khẩu chỉ đi trong body, không bao giờ nằm lại
// trong object sống lâu; token chỉ lưu sau khi server chấp nhận.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';

import 'fake_http.dart';
import '../models/fixtures/server_samples.dart';

/// Fixtures là **chuỗi JSON thật**; `jsonBody` cần object nên decode trước.
Object? _json(String body) => jsonDecode(body);

void main() {
  group('RemoteAuthRepository.login', () {
    test('gửi username/password và lưu token khi server trả về', () async {
      final f = fakeClient((_) => jsonBody(_json(loginWithTokenBody)));
      final auth = RemoteAuthRepository(f.client);

      final session = await auth.login('tuananh', 'Bicam@0110');

      final sent = f.adapter.last;
      expect(sent.method, 'POST');
      expect(sent.uri.path, '/login');
      expect(sent.data, {'username': 'tuananh', 'password': 'Bicam@0110'});

      expect(session.user.username, 'tuananh');
      expect(session.isAuthenticated, isTrue);
      expect(await auth.currentToken(), 'eyJhbGciOiJIUzI1NiJ9.demo');
    });

    test('server bản cũ chỉ trả message -> không token, vẫn chạy được',
        () async {
      final f = fakeClient((_) => jsonBody(_json(loginLegacyBody)));
      final auth = RemoteAuthRepository(f.client);

      final session = await auth.login('tuan', 'x');

      expect(session.isAuthenticated, isFalse);
      expect(await auth.currentToken(), isNull);
    });

    test('sai mật khẩu (401) -> unauthorized, KHÔNG lưu token', () async {
      final f = fakeClient(
        (_) => jsonBody(_json(loginWrongPasswordBody), 401),
      );
      final auth = RemoteAuthRepository(f.client);

      await expectLater(
        auth.login('tuan', 'sai'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.unauthorized),
        ),
      );
      expect(await auth.currentToken(), isNull);
    });

    test('HTTP 200 nhưng success=false -> coi như đăng nhập thất bại',
        () async {
      final f = fakeClient(
        (_) => jsonBody({'success': false, 'message': 'Sai thông tin'}),
      );
      final auth = RemoteAuthRepository(f.client);

      await expectLater(
        auth.login('tuan', 'sai'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.unauthorized)
              .having((e) => e.detail, 'detail', 'Sai thông tin'),
        ),
      );
      expect(await auth.currentToken(), isNull);
    });
  });

  group('RemoteAuthRepository.register', () {
    test('gửi kèm email và trả phiên', () async {
      final f = fakeClient((_) => jsonBody(_json(registerBody)));
      final auth = RemoteAuthRepository(f.client);

      await auth.register('tuananh', 'Bicam@0110', 'tuan@example.com');

      expect(f.adapter.last.uri.path, '/register');
      expect(f.adapter.last.data, {
        'username': 'tuananh',
        'password': 'Bicam@0110',
        'email': 'tuan@example.com',
      });
    });
  });

  group('RemoteAuthRepository.logout', () {
    test('xoá token cục bộ và gọi /logout', () async {
      final f = fakeClient((_) => jsonBody(_json(loginWithTokenBody)));
      final auth = RemoteAuthRepository(f.client);
      await auth.login('tuananh', 'x');

      await auth.logout();

      expect(await auth.currentToken(), isNull);
      expect(f.adapter.last.method, 'POST');
      expect(f.adapter.last.uri.path, '/logout');
    });

    test('server chưa có /logout (404) -> vẫn xoá token, KHÔNG ném', () async {
      final f = fakeClient((options) {
        if (options.uri.path == '/logout') {
          return jsonBody({'message': 'not found'}, 404);
        }
        return jsonBody(_json(loginWithTokenBody));
      });
      final auth = RemoteAuthRepository(f.client);
      await auth.login('tuananh', 'x');

      await expectLater(auth.logout(), completes);
      expect(await auth.currentToken(), isNull);
    });

    test('gọi logout nhiều lần vẫn an toàn', () async {
      final f = fakeClient((_) => jsonBody({}));
      final auth = RemoteAuthRepository(f.client);

      await expectLater(auth.logout(), completes);
      await expectLater(auth.logout(), completes);
      expect(await auth.currentToken(), isNull);
    });
  });

  test('token ban đầu là null', () async {
    final f = fakeClient((_) => jsonBody({}));
    final auth = RemoteAuthRepository(f.client);

    expect(await auth.currentToken(), isNull);
  });
}
