// Day 6 — Medicine: đọc JSON thật của server, không dùng dữ liệu bịa ra.
//
// Trọng tâm: server là CSV scrape nên **thiếu khoá**, **null**, **chuỗi rỗng**
// và **số thay chuỗi** đều xảy ra. Test phải chứng minh app không sập vì vậy.

import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/medicine.dart';

import 'fixtures/server_samples.dart';

/// Hộp `results` đầu tiên của body thật.
Map<String, dynamic> _firstRawMedicine() =>
    ((jsonDecode(searchMedicineBody) as Map<String, dynamic>)['results']! as List)
        .first as Map<String, dynamic>;

Map<String, dynamic> _secondRawMedicine() =>
    ((jsonDecode(searchMedicineBody) as Map<String, dynamic>)['results']! as List)[1]
        as Map<String, dynamic>;

/// Match: lỗi phải là `AppException` kiểu `parsing` — không phải crash,
/// không phải `FormatException` lọt lên tầng UI.
final Matcher throwsParsing = throwsA(
  isA<AppException>()
      .having((e) => e.kind, 'kind', AppExceptionKind.parsing),
);

void main() {
  group('fromJson — JSON thật', () {
    test('đọc đủ 8 trường của server', () {
      final m = Medicine.fromJson(_firstRawMedicine());

      expect(m.name, 'Panadol 500mg');
      expect(m.composition, 'Paracetamol 500mg');
      expect(m.uses, 'Pain relief, fever reduction');
      expect(m.sideEffects, 'Nausea, allergic reactions');
      expect(m.manufacturer, 'GSK');
      expect(m.excellentReviewPercent, 93);
      expect(m.averageReviewPercent, 5);
      expect(m.poorReviewPercent, 2);
    });

    test('review là số (88) cũng ra 88.0 — không ném', () {
      expect(
        Medicine.fromJson(_secondRawMedicine()).excellentReviewPercent,
        88,
      );
    });

    test('null và chuỗi rỗng thành null, không phải ""', () {
      final m = Medicine.fromJson(_secondRawMedicine());

      expect(m.uses, isNull);
      expect(m.sideEffects, isNull);
      expect(m.averageReviewPercent, isNull);
      expect(m.poorReviewPercent, isNull);
    });

    test('thiếu hẳn khoá cũng không sập (CSV lúc scrape thiếu cột)', () {
      final m = Medicine.fromJson(<String, dynamic>{'Medicine_Name': 'A'});

      expect(m.name, 'A');
      expect(m.composition, isNull);
      expect(m.excellentReviewPercent, isNull);
    });

    test('tên thuốc rỗng ra "" chứ không phải null', () {
      expect(Medicine.fromJson(<String, dynamic>{}).name, isEmpty);
    });

    test('giá trị số bị server gửi thành chuỗi vẫn đọc được', () {
      expect(
        Medicine.fromJson(<String, dynamic>{'Medicine_Name': 12345}).name,
        '12345',
      );
    });
  });

  group('tỉ lệ review', () {
    test('chấp nhận "93%", " 93 %", 93, 93.5 -> cùng một số', () {
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': '93%'})
            .excellentReviewPercent,
        93,
      );
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': ' 93 %'})
            .excellentReviewPercent,
        93,
      );
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': 93})
            .excellentReviewPercent,
        93,
      );
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': 93.5})
            .excellentReviewPercent,
        93.5,
      );
    });

    test('chặn giá trị vô lý (âm, > 100, chữ)', () {
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': -5})
            .excellentReviewPercent,
        0,
      );
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': 250})
            .excellentReviewPercent,
        100,
      );
      expect(
        Medicine.fromJson(<String, dynamic>{'Excellent Review %': 'N/A'})
            .excellentReviewPercent,
        isNull,
      );
    });

    test('round-trip: toJson -> fromJson giữ nguyên', () {
      final original = Medicine.fromJson(_firstRawMedicine());

      expect(Medicine.fromJson(original.toJson()), original);
    });

    test('toJson giữ nguyên TÊN khoá của server (hợp đồng)', () {
      final json = Medicine.fromJson(_firstRawMedicine()).toJson();

      expect(json['Medicine_Name'], 'Panadol 500mg');
      expect(json['Side_effects'], isNotNull);
      expect(json['Excellent Review %'], '93%');
    });
  });

  group('copyWith', () {
    test('đổi được field, giữ nguyên field còn lại', () {
      final m = Medicine.fromJson(_firstRawMedicine());

      final changed = m.copyWith(uses: 'Mới');

      expect(changed.uses, 'Mới');
      expect(changed.name, m.name);
      expect(changed.manufacturer, m.manufacturer);
    });

    test('truyền null XOÁ được field; không truyền thì giữ nguyên', () {
      final m = Medicine.fromJson(_firstRawMedicine());

      expect(m.copyWith(uses: null).uses, isNull);
      expect(m.copyWith().uses, m.uses);
    });

    test('bản gốc không bị mutate', () {
      final m = Medicine.fromJson(_firstRawMedicine());

      m.copyWith(name: 'Khác');

      expect(m.name, 'Panadol 500mg');
    });
  });

  group('== / hashCode', () {
    test('hai bản parse từ cùng JSON bằng nhau và cùng hash', () {
      final a = Medicine.fromJson(_firstRawMedicine());
      final b = Medicine.fromJson(_firstRawMedicine());

      expect(a, b);
      expect(a.hashCode, b.hashCode);
      expect({a, b}, hasLength(1));
    });

    test('khác 1 ô review là khác', () {
      final a = Medicine.fromJson(_firstRawMedicine());

      expect(a, isNot(a.copyWith(poorReviewPercent: 99)));
    });

    test('khác null với chuỗi rỗng là khác', () {
      final a = Medicine.fromJson(_firstRawMedicine());

      expect(a, isNot(a.copyWith(uses: '')));
    });
  });

  group('MedicineSearchResponse', () {
    test('bóc gói {"results": [...]}', () {
      final res = MedicineSearchResponse.parse(searchMedicineBody);

      expect(res.results, hasLength(2));
      expect(res.results.first.name, 'Panadol 500mg');
      expect(res.isEmpty, isFalse);
    });

    test('thiếu gói results -> rỗng, KHÔNG ném', () {
      final res = MedicineSearchResponse.fromJson(<String, dynamic>{});

      expect(res.results, isEmpty);
      expect(res.isEmpty, isTrue);
      expect(res.total, 0);
    });

    test('results là null -> rỗng (server hay trả vậy)', () {
      expect(
        MedicineSearchResponse.fromJson(
          <String, dynamic>{'results': null},
        ).results,
        isEmpty,
      );
    });

    test('results sai kiểu -> AppException parsing', () {
      expect(
        () => MedicineSearchResponse.fromJson(
          <String, dynamic>{'results': 'xac'},
        ),
        throwsParsing,
      );
    });

    test('phần tử không phải object -> AppException parsing', () {
      expect(
        () => MedicineSearchResponse.fromJson(
          <String, dynamic>{
            'results': <Object?>['xac'],
          },
        ),
        throwsParsing,
      );
    });

    test('body không phải JSON -> parsing, KHÔNG phải FormatException', () {
      expect(
        () => MedicineSearchResponse.parse('<html>500</html>'),
        throwsParsing,
      );
    });

    test('total lấy từ server nếu có, không thì = số kết quả', () {
      expect(
        MedicineSearchResponse.fromJson(
          <String, dynamic>{'results': <Object?>[], 'total': 42},
        ).total,
        42,
      );
      expect(MedicineSearchResponse.parse(searchMedicineBody).total, 2);
    });

    test('query do server echo lại; thiếu thì null', () {
      expect(
        MedicineSearchResponse.fromJson(
          <String, dynamic>{'results': <Object?>[], 'query': 'panadol'},
        ).query,
        'panadol',
      );
      expect(
        MedicineSearchResponse.parse(searchMedicineBody).query,
        isNull,
      );
    });
  });
}