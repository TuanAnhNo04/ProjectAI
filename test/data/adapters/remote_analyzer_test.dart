// Day 8 — RemoteAnalyzer nối `POST /process-image` của Flask hiện tại.
//
// Dùng adapter giả (xem fake_http.dart) thay vì mạng thật, JSON thật lấy từ
// `test/data/models/fixtures/server_samples.dart`.

import 'dart:convert';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/adapters/remote_analyzer.dart';

import 'fake_http.dart';
import '../models/fixtures/server_samples.dart';

/// Fixtures là **chuỗi JSON thật**; `jsonBody` cần object nên decode trước.
Object? _json(String body) => jsonDecode(body);

/// Tạo một file ảnh tạm; trả về đường dẫn. Dọn ở `addTearDown` để Dio kịp
/// nhả handle (Windows ném `PathAccessException errno 32` nếu xoá quá sớm).
Future<String> _tempImage() async {
  final dir = await Directory.systemTemp.createTemp('remote_analyzer_test');
  final file = File('${dir.path}/toa-thuoc.jpg')..writeAsStringSync('abc');
  addTearDown(() async {
    try {
      await dir.delete(recursive: true);
    } on FileSystemException {
      // File còn bị khoá -> thư mục nằm trong %TEMP%, không sao để lại.
    }
  });
  return file.path;
}

void main() {
  group('RemoteAnalyzer', () {
    test('capability nói rõ đang chạy tầng từ xa', () {
      final f = fakeClient((_) => jsonBody(_json(processImageBody)));
      final analyzer = RemoteAnalyzer(f.client);

      expect(analyzer.capability, 'remote-flask');
    });

    test('upload ảnh đúng field "image" và map medicine_info', () async {
      final imagePath = await _tempImage();
      final f = fakeClient((_) => jsonBody(_json(processImageBody)));
      final analyzer = RemoteAnalyzer(f.client);

      final result = await analyzer.analyze(imagePath);

      // --- hợp đồng HTTP ---
      final sent = f.adapter.last;
      expect(sent.method, 'POST');
      expect(sent.uri.path, '/process-image');
      final form = sent.data as FormData;
      expect(form.files.map((e) => e.key), ['image']);
      expect(form.files.single.value.filename, 'toa-thuoc.jpg');

      // --- dữ liệu map đúng ---
      expect(result.isNotEmpty, isTrue);
      expect(result.medicines.single.name, 'Panadol 500mg');
      expect(result.medicines.single.composition, 'Paracetamol 500mg');
      expect(result.medicines.single.excellentReviewPercent, 93);
    });

    test('ảnh không nhận ra thuốc nào -> kết quả rỗng, không lỗi', () async {
      final imagePath = await _tempImage();
      final f = fakeClient((_) => jsonBody(_json(processImageEmptyBody)));
      final analyzer = RemoteAnalyzer(f.client);

      final result = await analyzer.analyze(imagePath);

      expect(result.isEmpty, isTrue);
      expect(result.medicines, isEmpty);
    });

    test('đường dẫn ảnh rỗng -> validation, KHÔNG gửi request', () async {
      final f = fakeClient((_) => jsonBody(_json(processImageBody)));
      final analyzer = RemoteAnalyzer(f.client);

      await expectLater(
        analyzer.analyze('   '),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.validation),
        ),
      );
      expect(f.adapter.sent, isEmpty);
    });

    test('server 500 -> AppException server (UI bọc guard() hiển thị)',
        () async {
      final imagePath = await _tempImage();
      final f = fakeClient(
        (_) => jsonBody({'message': 'OCR chết'}, 500),
      );
      final analyzer = RemoteAnalyzer(f.client);

      await expectLater(
        analyzer.analyze(imagePath),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.server)
              .having((e) => e.detail, 'detail', 'OCR chết'),
        ),
      );
    });

    test('body sai hình dạng (list) -> parsing, không sập UI', () async {
      final imagePath = await _tempImage();
      final f = fakeClient((_) => jsonBody(<int>[1, 2, 3]));
      final analyzer = RemoteAnalyzer(f.client);

      await expectLater(
        analyzer.analyze(imagePath),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.parsing),
        ),
      );
    });
  });
}
