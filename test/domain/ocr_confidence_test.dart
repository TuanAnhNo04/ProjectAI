import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/domain/ocr_confidence.dart';

// Tên thuốc trong test đều là tên GIẢ.
void main() {
  group('assessOcrItem', () {
    test('clean item has no issues', () {
      expect(assessOcrItem(name: 'Fakemol 500mg', dose: '1 viên'), isEmpty);
    });

    test('flags missing dose but it does not lock', () {
      final issues = assessOcrItem(name: 'Fakemol', dose: ' ');
      expect(issues, [OcrIssue.missingDose]);
      expect(issues.any(lockingIssues.contains), isFalse);
    });

    test('flags digit inside a word as suspicious (locks)', () {
      final issues = assessOcrItem(name: 'Fakem0l', dose: '1 viên');
      expect(issues, contains(OcrIssue.suspiciousChars));
      expect(issues.any(lockingIssues.contains), isTrue);
    });

    test('strength like 500mg is not suspicious', () {
      expect(
        assessOcrItem(name: 'Fakemol 500mg', dose: '1 viên'),
        isNot(contains(OcrIssue.suspiciousChars)),
      );
    });

    test('odd OCR characters are suspicious', () {
      expect(
        assessOcrItem(name: 'Fake|mol', dose: '1 viên'),
        contains(OcrIssue.suspiciousChars),
      );
    });

    test('short name and unclear frequency', () {
      expect(
        assessOcrItem(name: 'Fa', dose: '1 viên', needsReview: true),
        containsAll([OcrIssue.shortName, OcrIssue.unclearFrequency]),
      );
    });

    test('notInCatalog only after a check returned nothing', () {
      expect(assessOcrItem(name: 'Fakemol', dose: '1 viên'),
          isNot(contains(OcrIssue.notInCatalog)));
      expect(
        assessOcrItem(name: 'Fakemol', dose: '1 viên', match: CatalogMatch.none),
        contains(OcrIssue.notInCatalog),
      );
    });
  });

  group('matchCatalog', () {
    test('exact ignoring accents, case, spaces', () {
      expect(matchCatalog('fakemol 500 MG', ['Fakemol 500mg']),
          CatalogMatch.exact);
      expect(matchCatalog('Thuốc Giả', ['thuoc gia']), CatalogMatch.exact);
    });

    test('prefix is similar', () {
      expect(matchCatalog('Fakemol', ['Fakemol Extra']), CatalogMatch.similar);
    });

    test('different names returned by search are still similar', () {
      expect(matchCatalog('Fakemol', ['Otherfake']), CatalogMatch.similar);
    });

    test('no candidates is none', () {
      expect(matchCatalog('Fakemol', const []), CatalogMatch.none);
      expect(matchCatalog('   ', ['Fakemol']), CatalogMatch.none);
    });
  });
}
