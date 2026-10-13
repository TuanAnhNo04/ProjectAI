// Day 6 — AnalysisResult: thay cho `split('\n')` + `split(': ')` của code cũ.
//
// Ca quan trọng nhất: tên thuốc / thành phần chứa `: ` thì code cũ hiển thị
// sai (vì `.split(': ').last` chỉ lấy phần sau cùng). Model này không còn
// cách nào vỡ vì nó ghép thẳng từ field đã tách sẵn.

import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/data/models/analysis_result.dart';
import 'package:image_processing_app/data/models/medicine.dart';

import 'fixtures/server_samples.dart';

final Matcher throwsParsing = throwsA(
  isA<AppException>().having((e) => e.kind, 'kind', AppExceptionKind.parsing),
);

void main() {
  group('fromJson — POST /process-image', () {
    test('bóc gói {"medicine_info": [...]}', () {
      final res = AnalysisResult.parse(processImageBody);

      expect(res.medicines, hasLength(1));
      expect(res.medicines.first.name, 'Panadol 500mg');
      expect(res.isEmpty, isFalse);
      expect(res.isNotEmpty, isTrue);
    });

    test('ảnh không nhận ra thuốc -> rỗng, không ném', () {
      final res = AnalysisResult.parse(processImageEmptyBody);

      expect(res.medicines, isEmpty);
      expect(res.isEmpty, isTrue);
    });

    test('thiếu gói -> rỗng, KHÔNG ném', () {
      expect(
        AnalysisResult.fromJson(<String, dynamic>{}).medicines,
        isEmpty,
      );
    });

    test('medicine_info sai kiểu -> AppException parsing', () {
      expect(
        () => AnalysisResult.fromJson(
          <String, dynamic>{'medicine_info': 'loi'},
        ),
        throwsParsing,
      );
    });

    test('body HTML của Flask -> parsing chứ không crash', () {
      expect(() => AnalysisResult.parse('<!doctype html>'), throwsParsing);
    });
  });

  group('đọc được cả 3 hình dạng body', () {
    test('medicines (hợp đồng mới) được ưu tiên', () {
      final res = AnalysisResult.fromJson(<String, dynamic>{
        'medicines': <Object?>[
          <String, dynamic>{'Medicine_Name': 'Moi'},
        ],
        'medicine_info': <Object?>[
          <String, dynamic>{'Medicine_Name': 'Cu'},
        ],
      });

      expect(res.medicines.single.name, 'Moi');
    });

    test('search_results (dạng history/cache) cũng đọc được', () {
      final res = AnalysisResult.fromJson(<String, dynamic>{
        'search_results': <Object?>[
          <String, dynamic>{'Medicine_Name': 'Lich su'},
        ],
      });

      expect(res.medicines.single.name, 'Lich su');
    });

    test('text + image_path + analyzed_at đi kèm', () {
      final res = AnalysisResult.fromJson(<String, dynamic>{
        'text': 'PANADOL 500',
        'image_path': '/uploads/a.jpg',
        'analyzed_at': '2026-09-21T08:30:00Z',
        'medicines': <Object?>[],
      });

      expect(res.text, 'PANADOL 500');
      expect(res.imagePath, '/uploads/a.jpg');
      expect(res.analyzedAt, isNotNull);
    });
  });

  group('toDisplayLines — thay split(\': \')', () {
    const tricky = Medicine(
      name: 'Panadol: Extra',
      composition: 'Paracetamol 500mg; Caffeine: 65mg',
      uses: 'Giảm đau: đầu, răng',
      sideEffects: null,
    );

    test('giữ nguyên dấu : bên trong dữ liệu (bug của code cũ)', () {
      final lines = const AnalysisResult(medicines: [tricky]).toDisplayLines();

      expect(
        lines.first,
        'Tên: Panadol: Extra',
        reason: 'split(\': \').last sẽ mất chữ "Panadol:"',
      );
      expect(lines[1], 'Thành phần: Paracetamol 500mg; Caffeine: 65mg');
      expect(lines[2], 'Công dụng: Giảm đau: đầu, răng');
    });

    test('bỏ dòng khi field null, không in "null"', () {
      final lines = const AnalysisResult(medicines: [tricky]).toDisplayLines();

      expect(lines.any((l) => l.startsWith('Tác dụng phụ')), isFalse);
      expect(lines.join('\n'), isNot(contains('null')));
    });

    test('thuốc không có field nào -> chỉ còn dòng tên', () {
      expect(
        const AnalysisResult(medicines: [Medicine(name: 'A')]).toDisplayLines(),
        <String>['Tên: A'],
      );
    });

    test('kết quả rỗng -> không dòng nào', () {
      expect(const AnalysisResult.empty().toDisplayLines(), isEmpty);
    });

    test('nhiều thuốc -> dòng nối tiếp đúng thứ tự', () {
      final lines = const AnalysisResult(
        medicines: [
          Medicine(name: 'A', uses: 'u1'),
          Medicine(name: 'B', uses: 'u2'),
        ],
      ).toDisplayLines();

      expect(lines,
          <String>['Tên: A', 'Công dụng: u1', 'Tên: B', 'Công dụng: u2']);
    });
  });

  group('copyWith / == / toJson', () {
    test('copyWith thay danh sách thuốc', () {
      final res = AnalysisResult.parse(processImageBody);

      final cleared = res.copyWith(medicines: const []);

      expect(cleared.medicines, isEmpty);
      expect(res.medicines, hasLength(1), reason: 'bản gốc không mutate');
    });

    test('copyWith null xoá được text', () {
      const res = AnalysisResult(text: 'abc');

      expect(res.copyWith(text: null).text, isNull);
      expect(res.copyWith().text, 'abc');
    });

    test('round-trip toJson -> fromJson giữ nguyên thuốc', () {
      final original = AnalysisResult.parse(processImageBody);

      final again = AnalysisResult.fromJson(original.toJson());

      expect(again.medicines, original.medicines);
    });

    test('toJson dùng khoá "medicines" đúng hợp đồng mới', () {
      final json = AnalysisResult.parse(processImageBody).toJson();

      expect(json['medicines'], isA<List<Object?>>());
      expect(json.containsKey('medicine_info'), isFalse);
    });
  });
}
