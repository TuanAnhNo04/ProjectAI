// Day 6 — User/AuthSession. Test quan trọng nhất:
// **mật khẩu không bao giờ được giữ lại hay phát lại ra JSON**.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/data/models/user.dart';

import 'fixtures/server_samples.dart';

Map<String, dynamic> _decoded(String body) =>
    jsonDecode(body) as Map<String, dynamic>;

void main() {
  group('User.fromJson', () {
    test('đọc user lồng trong phản hồi login', () {
      final session = AuthSession.parse(loginWithTokenBody);

      expect(session.user.id, 7);
      expect(session.user.username, 'tuananh');
      expect(session.user.email, 'tuan@example.com');
      expect(session.token, 'eyJhbGciOiJIUzI1NiJ9.demo');
      expect(session.isAuthenticated, isTrue);
    });

    test('bản cũ KHÔNG trả token -> isAuthenticated = false, không ném', () {
      final session = AuthSession.parse(loginLegacyBody);

      expect(session.token, isNull);
      expect(session.isAuthenticated, isFalse);
    });

    test('thiếu hẳn user -> vẫn dựng được User (username rỗng)', () {
      final session = AuthSession.fromJson(<String, dynamic>{
        'token': 't',
      });

      expect(session.user.username, isEmpty);
      expect(session.isAuthenticated, isTrue);
    });

    test('token rỗng -> coi như chưa đăng nhập', () {
      final session = AuthSession.fromJson(<String, dynamic>{
        'token': '   ',
        'user': <String, dynamic>{'username': 'a'},
      });

      expect(session.isAuthenticated, isFalse);
    });

    test('đọc được access_token (FastAPI Day 15)', () {
      final session = AuthSession.fromJson(<String, dynamic>{
        'access_token': 'jwt',
      });

      expect(session.token, 'jwt');
    });

    test('token có độ ưu tiên hơn access_token', () {
      expect(
        AuthSession.fromJson(<String, dynamic>{
          'token': 'a',
          'access_token': 'b',
        }).token,
        'a',
      );
    });
  });

  group('validate email (Day 11)', () {
    test('email hợp lệ', () {
      for (final e in ['a@b.co', 'tuan.anh+thuoc@gmail.com', 'x_y@z.io']) {
        expect(User(username: 'a', email: e).hasValidEmail, isTrue, reason: e);
      }
    });

    test('email sai -> false, không ném', () {
      for (final e in ['', 'khongco@', '@khongco', 'khong co dau', 'a@b']) {
        expect(
          User(username: 'a', email: e).hasValidEmail,
          isFalse,
          reason: e,
        );
      }
      expect(User(username: 'a').hasValidEmail, isFalse);
    });
  });

  group('BẢO MẬT — mật khẩu không nằm lại trong model', () {
    const secret = 'Bicam@0110';

    test('toJson() KHÔNG chứa password', () {
      final user = User(username: 'tuananh', email: 'a@b.co');

      expect(user.toJson().toString(), isNot(contains('password')));
    });

    test('toJson() không đổi trên cả vòng đời', () {
      final user = User(username: 'tuananh');

      user.toCredentialsJson(password: secret);

      expect(user.toJson().toString(), isNot(contains(secret)));
    });

    test('chỉ toCredentialsJson mới chứa mật khẩu, và đúng 1 lần', () {
      final body = User(username: 'tuananh').toCredentialsJson(password: secret);

      expect(body['username'], 'tuananh');
      expect(body['password'], secret);
      // Không gửi email rỗng lên /register.
      expect(body.containsKey('email'), isFalse);
    });

    test('AuthSession.toString KHÔNG lộ token', () {
      final text = AuthSession.parse(loginWithTokenBody).toString();

      expect(text, isNot(contains('eyJhbGci')));
      expect(text, contains('có token'));
    });

    test('User.toString chỉ có username', () {
      expect(
        const User(username: 'tuananh').toString(),
        isNot(contains('password')),
      );
    });
  });

  group('copyWith / ==', () {
    test('copyWith đổi email, giữ username', () {
      final u = const User(id: 1, username: 'a', email: 'old@b.co');

      final changed = u.copyWith(email: 'new@b.co');

      expect(changed.email, 'new@b.co');
      expect(changed.username, 'a');
      expect(changed.id, 1);
    });

    test('truyền null xoá được email', () {
      const u = User(username: 'a', email: 'a@b.co');

      expect(u.copyWith(email: null).email, isNull);
      expect(u.copyWith().email, 'a@b.co');
    });

    test('== và hashCode', () {
      const a = User(id: 1, username: 'a', email: 'a@b.co');
      const b = User(id: 1, username: 'a', email: 'a@b.co');

      expect(a, b);
      expect(a.hashCode, b.hashCode);
      expect(a, isNot(a.copyWith(email: 'c@d.co')));
    });
  });

  group('dữ liệu lỗi từ server không làm sập model', () {
    test('body đăng nhập sai vẫn parse được (không phải lỗi parse)', () {
      final body = _decoded(loginWrongPasswordBody);

      expect(body['message'], contains('Sai tai khoan'));
      // Lỗi này do server trả, KHÔNG phải AppExceptionKind.parsing.
      expect(AuthSession.fromJson(body).isAuthenticated, isFalse);
    });

    test('register body không có user vẫn dựng được', () {
      final session = AuthSession.fromJson(_decoded(registerBody));

      expect(session.user.username, isEmpty);
      expect(session.isAuthenticated, isFalse);
    });

    test('id gửi lên dạng chuỗi vẫn đọc được', () {
      expect(
        User.fromJson(<String, dynamic>{'id': '7', 'username': 'a'}).id,
        7,
      );
    });
  });
}