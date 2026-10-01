// Day 5 — mọi lỗi mạng phải đổi được thành một câu tiếng Việt đúng nghĩa.
//
// Test này dựng DioException thủ công (không gọi mạng) nên chạy offline hoàn toàn.

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';

const String _host = 'http://10.0.2.2:5000';

DioException _status(int code, {Object? data, String path = '/history'}) {
  final options = RequestOptions(path: '$_host$path');
  return DioException(
    type: DioExceptionType.badResponse,
    requestOptions: options,
    response: Response<dynamic>(
      statusCode: code,
      data: data,
      requestOptions: options,
    ),
  );
}

DioException _failure(DioExceptionType type, {String path = '/history'}) =>
    DioException(
      type: type,
      requestOptions: RequestOptions(path: '$_host$path'),
      error: 'chi tiet ky thuat',
    );

void main() {
  group('fromStatus', () {
    test('map từng mã HTTP sang đúng kind', () {
      const Map<int, AppExceptionKind> expected = {
        400: AppExceptionKind.validation,
        401: AppExceptionKind.unauthorized,
        403: AppExceptionKind.forbidden,
        404: AppExceptionKind.notFound,
        409: AppExceptionKind.conflict,
        422: AppExceptionKind.validation,
        500: AppExceptionKind.server,
        502: AppExceptionKind.server,
        503: AppExceptionKind.server,
        418: AppExceptionKind.unknown,
        302: AppExceptionKind.unknown,
      };
      expected.forEach((status, kind) {
        expect(AppException.fromStatus(status), kind, reason: 'status $status');
      });
    });
  });

  group('thông báo theo mã HTTP', () {
    test('401 -> bắt đăng nhập lại', () {
      final e = AppException.fromDio(_status(401));

      expect(e.kind, AppExceptionKind.unauthorized);
      expect(e.statusCode, 401);
      expect(e.isUnauthorized, isTrue);
      expect(e.message, contains('đăng nhập lại'));
    });

    test('404 -> Không tìm thấy dữ liệu', () {
      expect(
        AppException.fromDio(_status(404)).message,
        'Không tìm thấy dữ liệu bạn yêu cầu.',
      );
    });

    test('500 -> lỗi máy chủ, retry được', () {
      final e = AppException.fromDio(_status(500));

      expect(e.kind, AppExceptionKind.server);
      expect(e.isRetryable, isTrue);
      expect(e.message, contains('Máy chủ'));
    });

    test('mã lạ -> hiện đúng mã để debug', () {
      expect(AppException.messageForStatus(418), contains('418'));
    });

    test('lỗi validation không tính là retry được', () {
      expect(AppException.fromDio(_status(422)).isRetryable, isFalse);
    });
  });

  group('lấy message từ thân phản hồi', () {
    test('Flask: {"message": "..."} -> nằm ở detail để log', () {
      final e = AppException.fromDio(
        _status(401, data: {'message': 'Sai tài khoản hoặc mật khẩu'}),
      );

      expect(e.detail, 'Sai tài khoản hoặc mật khẩu');
      expect(e.message, contains('đăng nhập lại'));
    });

    test('FastAPI (Day 15): {"detail": "..."}', () {
      expect(
        AppException.serverMessage({'detail': 'Thuoc khong ton tai'}),
        'Thuoc khong ton tai',
      );
    });

    test('FastAPI validation: detail là danh sách', () {
      final detail = AppException.serverMessage({
        'detail': [
          {
            'loc': ['body', 'password'],
            'msg': 'Shorter than minimum length'
          },
        ],
      });

      expect(detail, contains('Shorter than minimum length'));
    });

    test('body là chuỗi JSON vẫn đọc được', () {
      expect(
        AppException.serverMessage('{"error":"UNG AUTHORISATION"}'),
        'UNG AUTHORISATION',
      );
    });

    test('HTML lỗi của Flask bị bỏ qua (không bắt người dùng đọc thẻ <>)', () {
      expect(
        AppException.serverMessage('<!doctype html><h1>500 Er</h1>'),
        isNull,
      );
    });

    test('nén khoảng trắng và cắt message quá dài', () {
      expect(
        AppException.serverMessage({'message': '  dòng   nhiều   khoảng '}),
        'dòng nhiều khoảng',
      );

      final cut = AppException.serverMessage({'message': 'a' * 500})!;
      expect(cut.length, 241);
      expect(cut, endsWith('…'));
    });

    test('message rỗng hoặc sai kiểu -> coi như không có', () {
      expect(AppException.serverMessage({'message': '   '}), isNull);
      expect(AppException.serverMessage(null), isNull);
      expect(AppException.serverMessage(123), isNull);
    });
  });

  group('lỗi trước khi có phản hồi', () {
    test('không kết nối được -> nêu rõ tên host', () {
      final e =
          AppException.fromDio(_failure(DioExceptionType.connectionError));

      expect(e.kind, AppExceptionKind.network);
      expect(e.statusCode, isNull);
      expect(e.uri.toString(), '$_host/history');
      expect(e.message, contains('10.0.2.2'));
      expect(e.isRetryable, isTrue);
    });

    test('phân biệt 3 loại timeout', () {
      expect(
        AppException.fromDio(_failure(DioExceptionType.connectionTimeout)).kind,
        AppExceptionKind.connectionTimeout,
      );
      expect(
        AppException.fromDio(_failure(DioExceptionType.sendTimeout)).kind,
        AppExceptionKind.sendTimeout,
      );
      expect(
        AppException.fromDio(_failure(DioExceptionType.receiveTimeout)).kind,
        AppExceptionKind.receiveTimeout,
      );
    });

    test('certificate sai tính là lỗi mạng (proxy tự ký khi debug)', () {
      expect(
        AppException.fromDio(_failure(DioExceptionType.badCertificate)).kind,
        AppExceptionKind.network,
      );
    });

    test('request bị huỷ -> isSilent, không hiện SnackBar', () {
      final e = AppException.fromDio(_failure(DioExceptionType.cancel));

      expect(e.kind, AppExceptionKind.canceled);
      expect(e.isSilent, isTrue);
      expect(e.isRetryable, isFalse);
    });

    test('thiếu host vẫn có câu an toàn', () {
      final message = AppException.messageForKind(AppExceptionKind.network);

      expect(message, contains('máy chủ'));
    });
  });

  group('from()', () {
    test('AppException được giữ nguyên, không bọc lại', () {
      const original = AppException(
        kind: AppExceptionKind.notFound,
        message: 'Không tìm thấy dữ liệu bạn yêu cầu.',
      );

      expect(AppException.from(original), same(original));
    });

    test('lỗi bất kỳ -> unknown + giữ cause', () {
      final cause = StateError('boom');
      final e = AppException.from(cause);

      expect(e.kind, AppExceptionKind.unknown);
      expect(e.cause, same(cause));
      expect(e.message, 'Ứng dụng gặp sự cố, vui lòng thử lại.');
    });
  });

  test('toString có kind + mã + detail cho log', () {
    final e = AppException.fromDio(_status(500, data: {'message': 'db down'}));

    expect(e.toString(), contains('server'));
    expect(e.toString(), contains('500'));
    expect(e.toString(), contains('db down'));
  });
}
