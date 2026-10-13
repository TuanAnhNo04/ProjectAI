// Day 11 — kiểm chứng **bảo mật** ở tầng client.
//
// Mục tiêu: chứng minh bằng test rằng
//   1. endpoint có bảo vệ (`/history`, `/process-image`) **có** header
//      `Authorization: Bearer <token>`;
//   2. endpoint công khai (`/login`, `/register`) **không** gửi token;
//   3. gặp 401 thì `onUnauthorized` được gọi đúng một lần;
//   4. token/mật khẩu **không bao giờ** lọt vào log.
//
// Dùng `HttpClientAdapter` giả nên **không chạm socket thật**.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';

class _RecordingAdapter implements HttpClientAdapter {
  _RecordingAdapter(this.responder);

  final ResponseBody Function(RequestOptions options) responder;

  final List<RequestOptions> sent = <RequestOptions>[];

  @override
  Future<ResponseBody> fetch(
    RequestOptions options,
    Stream<Uint8List>? requestStream,
    Future<void>? cancelFuture,
  ) async {
    sent.add(options);
    return responder(options);
  }

  @override
  void close({bool force = false}) {}
}

ResponseBody _json(Object? body, [int status = 200]) => ResponseBody.fromString(
      jsonEncode(body),
      status,
      headers: {
        Headers.contentTypeHeader: [Headers.jsonContentType]
      },
    );

/// Client có token giả + adapter ghi lại mọi request.
({ApiClient client, _RecordingAdapter adapter}) _build(
  ResponseBody Function(RequestOptions) responder, {
  TokenProvider? tokenProvider,
  UnauthorizedCallback? onUnauthorized,
}) {
  final adapter = _RecordingAdapter(responder);
  final client = ApiClient(
    dio: Dio()..httpClientAdapter = adapter,
    tokenProvider: tokenProvider ?? () async => 'tok-123',
    onUnauthorized: onUnauthorized,
    logTraffic: false,
  );
  return (client: client, adapter: adapter);
}

void main() {
  group('Authorization header', () {
    test('endpoint có bảo vệ -> gắn "Bearer <token>"', () async {
      final env = _build((_) => _json({'ok': true}));

      await env.client.get('/history');
      expect(
          env.adapter.sent.single.headers['Authorization'], 'Bearer tok-123');

      env.client.close();
    });

    test('upload ảnh cũng có header (quyền cho /process-image)', () async {
      final env = _build((_) => _json({'ok': true}));

      // File tạm 1 byte: adapter giả không đọc nội dung.
      final dir = Directory.systemTemp.createTempSync('medassist_d11');
      final file = File('${dir.path}/a.jpg')..writeAsBytesSync([0]);

      await env.client
          .upload('/process-image', files: [UploadFile(path: file.path)]);
      expect(
          env.adapter.sent.single.headers['Authorization'], 'Bearer tok-123');

      env.client.close();
      // Phải đóng client TRƯỚC khi xoá thư mục: Dio còn giữ handle tới file
      // đã upload nên xoá lúc này sẽ ném PathAccessException trên Windows.
      await Future<void>.delayed(Duration.zero);
      if (dir.existsSync()) {
        try {
          dir.deleteSync(recursive: true);
        } on FileSystemException {
          // Handle chưa giải phóng xong: thư mục tạm trong %TEMP% không
          // quan trọng nên bỏ qua thay vì làm test đỏ vì lý do này.
        }
      }
    });

    test('endpoint công khai -> KHÔNG gửi token', () async {
      final env = _build((_) => _json({'ok': true}));

      await env.client.post('/login', data: const {'username': 'an'});
      expect(
        env.adapter.sent.single.headers.containsKey('Authorization'),
        isFalse,
      );

      env.client.close();
    });

    test('không có token -> không gắn header rỗng', () async {
      final env = _build(
        (_) => _json({'ok': true}),
        tokenProvider: () async => null,
      );

      await env.client.get('/history');
      expect(
        env.adapter.sent.single.headers.containsKey('Authorization'),
        isFalse,
      );

      env.client.close();
    });
  });

  group('401', () {
    test('gặp 401 ở endpoint có bảo vệ -> gọi onUnauthorized', () async {
      var called = 0;
      final env = _build(
        (_) => _json({'m': 'expired'}, 401),
        onUnauthorized: () => called++,
      );

      await expectLater(
        env.client.get('/history'),
        throwsA(
          isA<AppException>().having(
            (e) => e.kind,
            'kind',
            AppExceptionKind.unauthorized,
          ),
        ),
      );
      expect(called, 1);

      env.client.close();
    });

    test('gặp 401 ở /login -> KHÔNG gọi (sai mật khẩu ≠ hết phiên)', () async {
      var called = 0;
      final env = _build(
        (_) => _json({'m': 'bad'}, 401),
        onUnauthorized: () => called++,
      );

      await expectLater(
        env.client.post('/login', data: const {'username': 'an'}),
        throwsA(isA<AppException>()),
      );
      expect(called, 0, reason: 'đang đăng nhập sai thì không xoá phiên');

      env.client.close();
    });

    test('lỗi 500 -> không gọi onUnauthorized', () async {
      var called = 0;
      final env = _build(
        (_) => _json({'m': 'boom'}, 500),
        onUnauthorized: () => called++,
      );

      await expectLater(
          env.client.get('/history'), throwsA(isA<AppException>()));
      expect(called, 0);

      env.client.close();
    });
  });

  group('không rò rỉ bí mật', () {
    test('redactHeaders che Authorization', () {
      final redacted = AuthInterceptor.redactHeaders(
        const {'Authorization': 'Bearer tok-123', 'Accept': 'application/json'},
      );
      expect(redacted['Authorization'], '***');
      expect(redacted['Accept'], 'application/json');
    });

    test('redactBody che password/token ở mọi cấp', () {
      final redacted = AuthInterceptor.redactBody(
        const {
          'username': 'an',
          'password': 'matkhau-that',
          'nested': {'token': 'tok-123'},
        },
      ) as Map<String, Object?>;

      final flat = jsonEncode(redacted);
      expect(flat.contains('matkhau-that'), isFalse);
      expect(flat.contains('tok-123'), isFalse);
      // Phần không nhạy cảm vẫn còn để debug được.
      expect(flat.contains('an'), isTrue);
    });

    test('danh sách khoá nhạy cảm đủ rộng', () {
      expect(AuthInterceptor.sensitiveKeys, contains('authorization'));
      expect(AuthInterceptor.sensitiveKeys, contains('password'));
      expect(AuthInterceptor.sensitiveKeys, contains('token'));
    });
  });
}
