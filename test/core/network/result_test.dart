// Day 5 — Result<T>/guard(): tầng UI không được thấy exception.

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/network/result.dart';

AppException _err([AppExceptionKind kind = AppExceptionKind.network]) =>
    AppException(kind: kind, message: 'lỗi mẫu');

void main() {
  group('Ok', () {
    test('mang giá trị, không có lỗi', () {
      const r = Ok<int>(42);

      expect(r.isOk, isTrue);
      expect(r.isErr, isFalse);
      expect(r.valueOrNull, 42);
      expect(r.errorOrNull, isNull);
      expect(r.toString(), 'Ok(42)');
    });

    test('map() rẽ đúng nhánh ok', () {
      expect(
        const Ok<String>('a').map(
          ok: (v) => 'ok:$v',
          err: (e) => 'err:${e.message}',
        ),
        'ok:a',
      );
    });

    test('mapValue() biến đổi giá trị', () {
      expect(const Ok<int>(2).mapValue((v) => v * 3), isA<Ok<int>>());
      expect(const Ok<int>(2).mapValue((v) => v * 3).valueOrNull, 6);
    });

    test('mapValue() đổi được kiểu Result', () {
      final r = const Ok<int>(2).mapValue<String>((v) => 's$v');

      expect(r, isA<Ok<String>>());
      expect(r.valueOrNull, 's2');
    });
  });

  group('Err', () {
    test('mang lỗi, không có giá trị', () {
      final r = Err<String>(_err());

      expect(r.isOk, isFalse);
      expect(r.isErr, isTrue);
      expect(r.valueOrNull, isNull);
      expect(r.errorOrNull?.kind, AppExceptionKind.network);
      expect(r.toString(), contains('lỗi mẫu'));
    });

    test('giữ dữ liệu một phần cho luồng offline (Day 12)', () {
      final r = Err<List<int>>(_err(), partial: const [1, 2]);

      expect(r.valueOrNull, [1, 2]);
      expect(r.errorOrNull, isNotNull);
    });

    test('map() rẽ đúng nhánh err', () {
      expect(
        Err<int>(_err()).map(ok: (v) => 'ok', err: (e) => e.message),
        'lỗi mẫu',
      );
    });

    test('mapValue() không chạy khi lỗi', () {
      var called = false;
      final r = Err<int>(_err()).mapValue((v) {
        called = true;
        return v;
      });

      expect(called, isFalse);
      expect(r.isErr, isTrue);
    });
  });

  group('guard', () {
    test('trả Ok khi body thành công', () async {
      final r = await guard(() async => 7);

      expect(r, isA<Ok<int>>());
      expect(r.valueOrNull, 7);
    });

    test('bắt AppException thành Err, giữ nguyên kind', () async {
      final r = await guard<int>(() async {
        throw _err(AppExceptionKind.notFound);
      });

      expect(r.errorOrNull?.kind, AppExceptionKind.notFound);
    });

    test('bắt lỗi bất kỳ thành Err/unknown', () async {
      final r = await guard<int>(() async => throw StateError('boom'));

      expect(r.isErr, isTrue);
      expect(r.errorOrNull?.kind, AppExceptionKind.unknown);
      expect(r.errorOrNull?.cause, isA<StateError>());
    });

    test('không nuốt lỗi khi body trả về lỗi khi dựng Result', () async {
      final r =
          await guard<String>(() async => throw const FormatException('x'));

      expect(r.errorOrNull?.kind, AppExceptionKind.unknown);
    });
  });
}
