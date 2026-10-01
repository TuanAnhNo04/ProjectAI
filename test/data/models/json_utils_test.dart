// Day 6 — json_utils: lớp phòng thủ chung cho mọi model.
//
// Nguyên tắc: hàm "đọc" **không bao giờ ném** (mất 1 ô dữ liệu còn hơn sập
// màn hình); hàm "ép hình dạng" **phải ném** AppException(parsing).

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/json_utils.dart';

void main() {
  group('asString / asStringOrNull', () {
    test('cắt khoảng trắng thừa', () {
      expect(asString('  Panadol  '), 'Panadol');
    });

    test('null dùng fallback', () {
      expect(asString(null, fallback: 'N/A'), 'N/A');
      expect(asString(null), isEmpty);
    });

    test('số/bool quy về chuỗi thay vì ném', () {
      expect(asString(123), '123');
      expect(asString(true), 'true');
    });

    test('chuỗi rỗng hoặc toàn khoảng trắng thành null', () {
      expect(asStringOrNull(''), isNull);
      expect(asStringOrNull('   '), isNull);
      expect(asStringOrNull(null), isNull);
      expect(asStringOrNull('  a '), 'a');
    });
  });

  group('asIntOrNull / asDoubleOrNull', () {
    test('đọc được số, chuỗi số, và số thực nguyên', () {
      expect(asIntOrNull(7), 7);
      expect(asIntOrNull('7'), 7);
      expect(asIntOrNull(7.9), 7);
      expect(asDoubleOrNull(7), 7.0);
      expect(asDoubleOrNull('7.5'), 7.5);
    });

    test('sai kiểu -> null, KHÔNG ném', () {
      expect(asIntOrNull('abc'), isNull);
      expect(asIntOrNull(null), isNull);
      expect(asIntOrNull(<String>[]), isNull);
      expect(asDoubleOrNull('abc'), isNull);
    });

    test('NaN / Infinity bị loại (JSON không có nhưng CSV parse có thể sinh)', () {
      expect(asDoubleOrNull(double.nan), isNull);
      expect(asDoubleOrNull(double.infinity), isNull);
    });
  });

  group('asPercent', () {
    test('mọi kiểu server hay gửi về cùng một số', () {
      expect(asPercent('93%'), 93);
      expect(asPercent(' 93 % '), 93);
      expect(asPercent(93), 93);
      expect(asPercent('93.5'), 93.5);
    });

    test('chặn giá trị vô lý', () {
      expect(asPercent(-1), 0);
      expect(asPercent(101), 100);
    });

    test('không đọc được -> null', () {
      expect(asPercent(null), isNull);
      expect(asPercent(''), isNull);
      expect(asPercent('N/A'), isNull);
      expect(asPercent(double.nan), isNull);
    });

    test('formatPercent là nghịch đảo của asPercent', () {
      expect(formatPercent(asPercent('93%')), '93%');
      expect(formatPercent(asPercent(93.4)), '93%');
      expect(formatPercent(null), isNull);
    });
  });

  group('asJsonMap / asJsonList', () {
    test('map khoá không phải String vẫn ép được', () {
      expect(asJsonMap(<int, Object?>{1: 'a'}, 'f'), <String, dynamic>{'1': 'a'});
    });

    test('sai hình dạng -> AppException parsing kèm tên trường', () {
      expect(
        () => asJsonMap('xac', 'ket qua'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.parsing)
              .having((e) => e.detail, 'detail', contains('ket qua')),
        ),
      );
      expect(
        () => asJsonList(5, 'history'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.parsing),
        ),
      );
    });

    test('list null -> rỗng (coi như "không có", không phải lỗi)', () {
      expect(asJsonList(null, 'f'), isEmpty);
      expect(asJsonList(<Object?>[], 'f'), isEmpty);
    });

    test('parsingError có message tiếng Việt cho UI', () {
      expect(parsingError('chi tiet').message, contains('không đúng định dạng'));
    });
  });

  group('decodeJsonObject', () {
    test('JSON hợp lệ -> map', () {
      expect(decodeJsonObject('{"a":1}'), <String, dynamic>{'a': 1});
    });

    test('HTML lỗi của Flask -> parsing, KHÔNG FormatException', () {
      expect(
        () => decodeJsonObject('<!doctype html><h1>500</h1>'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.parsing),
        ),
      );
    });

    test('JSON hợp lệ nhưng là mảng -> parsing', () {
      expect(
        () => decodeJsonObject('[1,2]'),
        throwsA(
          isA<AppException>()
              .having((e) => e.kind, 'kind', AppExceptionKind.parsing),
        ),
      );
    });
  });

  group('asDateTimeOrNull', () {
    test('ISO-8601 -> DateTime local', () {
      final d = asDateTimeOrNull('2026-09-21T08:30:00Z');

      expect(d, isNotNull);
      expect(d!.toUtc().year, 2026);
    });

    test('sai định dạng / rỗng -> null', () {
      expect(asDateTimeOrNull('x'), isNull);
      expect(asDateTimeOrNull(''), isNull);
      expect(asDateTimeOrNull(null), isNull);
    });
  });
}