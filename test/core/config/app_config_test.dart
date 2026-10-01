// Day 5 — kiểm tra AppConfig: mọi URL của app đi qua đây.
//
// Chạy bình thường:      flutter test test/core      -> dùng giá trị mặc định
// Chạy với override:     flutter test test/core --dart-define=API_BASE_URL=http://192.168.1.10:5000
// Cả hai lần đều phải xanh — test tự phát hiện --dart-define có được đọc hay không.

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/config/app_config.dart';

/// Đọc đúng biến mà `AppConfig` đang dùng.
const String _rawDefine = String.fromEnvironment('API_BASE_URL');

void main() {
  group('AppConfig.baseUrl', () {
    test('lấy từ --dart-define, nếu không có thì dùng host emulator', () {
      if (_rawDefine.isEmpty) {
        expect(AppConfig.baseUrl, 'http://10.0.2.2:5000');
      } else {
        expect(AppConfig.baseUrl, _rawDefine);
      }
    });

    test('cấu hình mặc định không có vấn đề gì', () {
      expect(AppConfig.validate(), isEmpty);
    });

    test('timeout: nhận dữ liệu phải dài hơn lúc kết nối', () {
      expect(AppConfig.connectTimeout, const Duration(seconds: 15));
      expect(AppConfig.receiveTimeout, const Duration(seconds: 60));
      expect(AppConfig.receiveTimeout, greaterThan(AppConfig.connectTimeout));
    });

    test('describe() nói rõ đang trỏ đâu (để in ra log)', () {
      expect(AppConfig.describe(), contains(AppConfig.baseUrl));
      expect(AppConfig.describe(), contains('http'));
    });
  });

  group('AppConfig.uri', () {
    test('nối đường dẫn tương đối vào baseUrl', () {
      expect(
        AppConfig.uri('/history').toString(),
        '${AppConfig.baseUrl}/history',
      );
    });

    test('tự thêm dấu / nếu quên', () {
      expect(
        AppConfig.uri('process-image').toString(),
        '${AppConfig.baseUrl}/process-image',
      );
    });

    test('encode tham số tìm kiếm (sửa bug nối chuỗi của code cũ)', () {
      final uri = AppConfig.uri(
        '/search-medicine',
        queryParameters: {'name': 'Panadol 500 mg & Co'},
      );

      expect(uri.toString(), contains('name=Panadol+500+mg+%26+Co'));
      expect(uri.queryParameters['name'], 'Panadol 500 mg & Co');
    });

    test('encode tên thuốc có dấu tiếng Việt', () {
      final uri = AppConfig.uri(
        '/search-medicine',
        queryParameters: {'name': 'thuốc cảm cúm'},
      );

      expect(uri.toString(), isNot(contains('thuốc')));
      expect(uri.queryParameters['name'], 'thuốc cảm cúm');
    });

    test('bỏ qua tham số null và không tạo "?" thừa', () {
      final uri = AppConfig.uri(
        '/history',
        queryParameters: {'page': null, 'q': 'x'},
      );

      expect(uri.toString(), '${AppConfig.baseUrl}/history?q=x');
    });

    test('không có query thì không có dấu ?', () {
      expect(AppConfig.uri('/login').query, isEmpty);
      expect(
        AppConfig.uri('/login', queryParameters: const {'a': null}).query,
        isEmpty,
      );
    });
  });
}
