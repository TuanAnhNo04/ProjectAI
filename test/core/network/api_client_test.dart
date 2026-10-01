// Day 5 — ApiClient: dựng URL, gắn token, đổi lỗi, và KHÔNG log bí mật.
//
// Thay vì gọi mạng thật (flutter test chặn socket), test cài `HttpClientAdapter`
// giả để kiểm soát từng phản hồi — đúng chỗ Dio sẽ đi tới khi Day 8 nối Flask.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/config/app_config.dart';
import 'package:image_processing_app/core/network/api_client.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/network/result.dart';

class _FakeAdapter implements HttpClientAdapter {
  _FakeAdapter(this.responder);

  final ResponseBody Function(RequestOptions options) responder;

  /// Bắt lại mọi request để kiểm tra URL/header/body mà interceptor tạo ra.
  final List<RequestOptions> sent = [];

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

late _FakeAdapter _adapter;

ApiClient _client(
  ResponseBody Function(RequestOptions options) responder, {
  TokenProvider? tokenProvider,
  UnauthorizedCallback? onUnauthorized,
}) {
  _adapter = _FakeAdapter(responder);
  return ApiClient(
    dio: Dio()..httpClientAdapter = _adapter,
    tokenProvider: tokenProvider,
    onUnauthorized: onUnauthorized,
    logTraffic: false,
  );
}

String? _authOf(RequestOptions o) => o.headers['Authorization'] as String?;

void main() {
  group('AppConfig -> ApiClient.url', () {
    test('nối path tương đối vào baseUrl', () {
      final client = _client((_) => _json({}));

      expect(client.url('/history').toString(), '${AppConfig.baseUrl}/history');
      expect(client.url('history').toString(), '${AppConfig.baseUrl}/history');
    });

    test('URL tuyệt đối của bên thứ 3 được giữ nguyên', () {
      final client = _client((_) => _json({}));
      const external = 'https://cdn.khac.vn/anh/1.jpg';

      expect(client.url(external).toString(), external);
    });

    test('query được encode, không nối chuỗi thủ công', () {
      final client = _client((_) => _json({}));
      final uri = client.url('/search-medicine', {'name': 'Panadol & Co'});

      expect(uri.toString(), contains('name=Panadol+%26+Co'));
    });

    test('timeout mặc định được áp vào Dio', () {
      final client = _client((_) => _json({}));

      expect(client.dio.options.connectTimeout, AppConfig.connectTimeout);
      expect(client.dio.options.receiveTimeout, AppConfig.receiveTimeout);
      expect(client.dio.options.headers['Accept'], 'application/json');
    });
  });

  group('basename', () {
    test('hỗ trợ cả đường dẫn Windows lẫn thiết bị', () {
      expect(ApiClient.basename(r'C:\Users\tuan\Anh.jpg'), 'Anh.jpg');
      expect(ApiClient.basename('/data/user/0/cache/photo.png'), 'photo.png');
      expect(ApiClient.basename('notitle.txt'), 'notitle.txt');
    });
  });

  group('request thành công', () {
    test('GET /history -> dựng đúng URL, decode JSON', () async {
      final client = _client((_) => _json([
            {'id': 1, 'filepath': 'a.jpg'},
          ]));

      final res = await client.get('/history');

      expect(_adapter.sent.single.method, 'GET');
      expect(_adapter.sent.single.uri.path, '/history');
      expect(_adapter.sent.single.headers['Accept'], 'application/json');
      expect(res.data, [
        {'id': 1, 'filepath': 'a.jpg'},
      ]);
    });

    test('POST gửi JSON kèm Content-Type của Dio', () async {
      final client = _client((_) => _json({'success': true}));

      final res = await client.post('/login', data: {'username': 'tuan'});

      final sent = _adapter.sent.single;
      expect(sent.method, 'POST');
      expect(sent.data, {'username': 'tuan'});
      expect(res.data['success'], isTrue);
    });

    test('DELETE /history/{id} dùng đúng method', () async {
      final client = _client((_) => _json({'success': true}));

      await client.delete('/history/42');

      expect(_adapter.sent.single.method, 'DELETE');
      expect(_adapter.sent.single.uri.path, '/history/42');
    });

    test('upload đặt tên field là "image" như Flask mong đợi', () async {
      final dir = await Directory.systemTemp.createTemp('api_client_test');
      final file = File('${dir.path}/toa-thuoc.jpg')..writeAsStringSync('abc');
      final client = _client((_) => _json({'success': true}));

      // Dọn ở `addTearDown` **sau khi** mọi assertion đã chạy: Dio giữ handle
      // file tới khi dựng xong multipart, nên xoá trong `finally` sẽ ném
      // PathAccessException trên Windows (errno 32) dù test vốn đã pass.
      addTearDown(() async {
        try {
          await dir.delete(recursive: true);
        } on FileSystemException {
          // File còn bị khoá -> thư mục nằm trong %TEMP%, không sao để lại.
        }
      });

      await client.upload(
        '/process-image',
        files: [UploadFile(path: file.path)],
        fields: {'note': 'chup ban thuoc'},
      );

      final form = _adapter.sent.single.data as FormData;
      expect(form.files.map((e) => e.key), ['image']);
      expect(form.files.single.value.filename, 'toa-thuoc.jpg');
      // So sánh `key=value` thay vì `MapEntry`: Dio khai báo `fields` là
      // `List<MapEntry<String, dynamic>>` còn `const MapEntry('a','b')` suy ra
      // `MapEntry<String, String>` — `==` của MapEntry so sánh cả runtimeType
      // nên `contains` dù giá trị giống hệt vẫn trượt.
      expect(
        form.fields.map((e) => '${e.key}=${e.value}'),
        contains('note=chup ban thuoc'),
      );
    });

    test('truy vấn được encode trước khi gửi', () async {
      final client = _client((_) => _json([]));

      await client.get('/search-medicine', query: {'name': 'sữa rửa mặt'});

      expect(_adapter.sent.single.uri.queryParameters['name'], 'sữa rửa mặt');
    });
  });

  group('token (chỗ gắn cho Day 11)', () {
    test('endpoint có bảo vệ -> gửi Bearer token', () async {
      final client =
          _client((_) => _json([]), tokenProvider: () async => 'XYZ');

      await client.get('/history');

      expect(_authOf(_adapter.sent.single), 'Bearer XYZ');
    });

    test('login/register là endpoint công khai -> không gửi token cũ',
        () async {
      final client =
          _client((_) => _json({}), tokenProvider: () async => 'XYZ');

      await client.post('/login', data: const {'username': 'tuan'});

      expect(_authOf(_adapter.sent.single), isNull);
    });

    test('không có token -> vẫn gửi request bình thường', () async {
      final client = _client((_) => _json([]), tokenProvider: () async => null);

      await client.get('/history');

      expect(_authOf(_adapter.sent.single), isNull);
    });

    test('header gọi truyền thẳng được ưu tiên, interceptor không ghi đè',
        () async {
      final client =
          _client((_) => _json([]), tokenProvider: () async => 'MOI');

      await client.get('/history', headers: {'Authorization': 'Bearer CU'});

      expect(_authOf(_adapter.sent.single), 'Bearer CU');
    });
  });

  group('lỗi được đổi thành AppException', () {
    test('404 -> notFound + message tiếng Việt', () async {
      final client = _client((_) => _json({'message': 'not found'}, 404));

      await expectLater(
        client.delete('/history/9'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.notFound)
              .having((e) => e.statusCode, 'status', 404)
              .having((e) => e.message, 'message',
                  'Không tìm thấy dữ liệu bạn yêu cầu.'),
        ),
      );
    });

    test('500 -> server, chi tiết của server nằm ở detail', () async {
      final client = _client(
        (_) => _json({'message': 'rollback chết'}, 500),
      );

      AppException? err;
      try {
        await client.get('/history');
      } on AppException catch (e) {
        err = e;
      }

      expect(err, isNotNull);
      expect(err!.kind, AppExceptionKind.server);
      expect(err.detail, 'rollback chết');
      expect(err.isRetryable, isTrue);
    });

    test('server chưa bật -> network, message nói tới host', () async {
      final client = _client((options) {
        throw DioException(
          type: DioExceptionType.connectionError,
          requestOptions: options,
          error: 'Connection refused',
        );
      });

      await expectLater(
        client.get('/history'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.network)
              .having((e) => e.isRetryable, 'retryable', isTrue),
        ),
      );
    });

    test('guard() bọc lại thành Result -> UI không cần try/catch', () async {
      final client = _client((_) => _json({}, 422));

      final result =
          await guard(() async => (await client.post('/login')).data);

      expect(result.isErr, isTrue);
      expect(result.errorOrNull?.kind, AppExceptionKind.validation);
    });
  });

  group('401 -> báo tầng auth (Day 11)', () {
    test('401 ở endpoint có bảo vệ gọi onUnauthorized đúng 1 lần', () async {
      var fired = 0;
      final client = _client(
        (_) => _json({'message': 'hết hạn'}, 401),
        onUnauthorized: () => fired++,
      );

      await expectLater(client.get('/history'), throwsA(isA<AppException>()));
      expect(fired, 1);
    });

    test('401 ở /login (sai mật khẩu) KHÔNG đá người dùng về màn hình cũ',
        () async {
      var fired = 0;
      final client = _client(
        (_) => _json({'message': 'sai mật khẩu'}, 401),
        onUnauthorized: () => fired++,
      );

      await expectLater(
        client.post('/login', data: const {'username': 'a'}),
        throwsA(isA<AppException>()),
      );
      expect(fired, 0);
    });

    test('đường dẫn có prefix vẫn nhận là công khai (/health/live)', () async {
      var fired = 0;
      final client = _client(
        (_) => _json({}, 401),
        onUnauthorized: () => fired++,
      );

      await expectLater(
          client.get('/health/live'), throwsA(isA<AppException>()));
      expect(fired, 0);
    });
  });

  group('log không được rò rỉ bí mật (DoD Day 11)', () {
    test('Authorization và Cookie bị che', () {
      final redacted = AuthInterceptor.redactHeaders({
        'Authorization': 'Bearer ABCDEF',
        'Cookie': 'sid=123',
        'Accept': 'application/json',
      });

      expect(redacted['Authorization'], '***');
      expect(redacted['Cookie'], '***');
      expect(redacted['Accept'], 'application/json');
    });

    test('mật khẩu bị che ở mọi độ sâu', () {
      final body = AuthInterceptor.redactBody({
        'username': 'tuan',
        'password': 'Bicam@0110',
        'profile': {'token': 'T-1', 'ten': 'Tuan Anh'},
      }) as Map<String, Object?>;

      expect(body['username'], 'tuan');
      expect(body['password'], isNot('Bicam@0110'));
      expect((body['profile']! as Map)['token'], '***');
    });

    test('FormData chỉ hiện số lượng, không in nhị phân', () {
      final form = FormData()
        ..fields.add(const MapEntry('username', 'tuan'))
        ..files.add(
          MapEntry('image', MultipartFile.fromString('abc', filename: 'x')),
        );

      expect(AuthInterceptor.redactBody(form), '<multipart 1 fields, 1 files>');
    });

    test('danh sách khoá nhạy cảm phủ các key Flask/FastAPI hay dùng', () {
      for (final key in [
        'authorization',
        'password',
        'confirm_password',
        'old_password',
        'new_password',
        'token',
        'access_token',
        'refresh_token',
      ]) {
        expect(
          AuthInterceptor.sensitiveKeys,
          contains(key),
          reason: 'thiếu key: $key',
        );
      }
    });
  });
}
