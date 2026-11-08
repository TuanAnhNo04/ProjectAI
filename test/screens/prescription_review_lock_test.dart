import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/health_profile.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';
import 'package:image_processing_app/domain/prescription_parser.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/prescription_review_screen.dart';

class _FakeReminders extends ReminderController {
  @override
  Future<void> load() async {}
}

class _FakeProfile extends HealthProfileController {
  @override
  Future<HealthProfile?> load() async => null;
}

/// Danh mục giả: trả [results] cho mọi truy vấn, hoặc ném lỗi nếu [fail].
class _FakeCatalog implements MedicineRepository {
  _FakeCatalog(this.results, {this.fail = false});
  final List<Medicine> results;
  final bool fail;
  int calls = 0;

  @override
  Future<MedicineSearchResponse> search(String name) async {
    calls++;
    if (fail) throw Exception('offline');
    return MedicineSearchResponse(results: results);
  }

  @override
  Future<List<HistoryItem>> history() async => const [];

  @override
  Future<void> deleteHistory(int id) async {}
}

Widget _wrap(List<ParsedPrescriptionItem> items, {MedicineRepository? repo}) =>
    ProviderScope(
      overrides: [
        reminderControllerProvider.overrideWith(_FakeReminders.new),
        healthProfileControllerProvider.overrideWith(_FakeProfile.new),
        if (repo != null) medicineRepositoryProvider.overrideWithValue(repo),
      ],
      child: MaterialApp(
        locale: const Locale('vi'),
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: PrescriptionReviewScreen(items: items),
      ),
    );

FilledButton _saveBtn(WidgetTester t) => t.widget<FilledButton>(
    find.byWidgetPredicate((w) => w is FilledButton).last);

void main() {
  testWidgets('Save disabled until needsReview item is confirmed',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(800, 2400));
    await tester.pumpWidget(_wrap(const [
      ParsedPrescriptionItem(medicineName: 'FakeMed A', needsReview: true),
    ]));
    await tester.pump();
    expect(_saveBtn(tester).onPressed, isNull);
    await tester.tap(find.byKey(const ValueKey('confirmItem')));
    await tester.pump();
    expect(_saveBtn(tester).onPressed, isNotNull);
  });

  testWidgets('Save enabled when no item needs review', (tester) async {
    await tester.binding.setSurfaceSize(const Size(800, 2400));
    await tester.pumpWidget(_wrap(const [
      ParsedPrescriptionItem(medicineName: 'FakeMed B'),
    ]));
    await tester.pump();
    expect(find.byKey(const ValueKey('confirmItem')), findsNothing);
    expect(_saveBtn(tester).onPressed, isNotNull);
  });

  testWidgets('catalog is not contacted until the user taps check',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(800, 2400));
    final repo = _FakeCatalog(const []);
    await tester.pumpWidget(_wrap(const [
      ParsedPrescriptionItem(medicineName: 'Fakemol', doseText: '1 viên'),
    ], repo: repo));
    await tester.pump();
    expect(repo.calls, 0);
  });

  testWidgets('not in catalog locks save; picking a suggestion is not needed',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(800, 2400));
    await tester.pumpWidget(_wrap(const [
      ParsedPrescriptionItem(medicineName: 'Fakemol', doseText: '1 viên'),
    ], repo: _FakeCatalog(const [])));
    await tester.pump();
    expect(_saveBtn(tester).onPressed, isNotNull);

    await tester.tap(find.byKey(const ValueKey('checkCatalog')));
    await tester.pumpAndSettle();
    expect(find.text('Không tìm thấy trong danh mục — có thể OCR đọc sai, '
        'hãy đối chiếu với đơn'), findsOneWidget);
    expect(_saveBtn(tester).onPressed, isNull);

    await tester.tap(find.byKey(const ValueKey('confirmItem')));
    await tester.pump();
    expect(_saveBtn(tester).onPressed, isNotNull);
  });

  testWidgets('suggestion shows strength and replaces the OCR name',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(800, 2400));
    await tester.pumpWidget(_wrap(const [
      ParsedPrescriptionItem(medicineName: 'Fakem0l', doseText: '1 viên'),
    ], repo: _FakeCatalog(const [
      Medicine(name: 'Fakemol 500', composition: 'Fakeamin 500mg'),
    ])));
    await tester.pump();
    expect(find.byKey(const ValueKey('lowConfidence')), findsOneWidget);
    expect(_saveBtn(tester).onPressed, isNull);

    await tester.tap(find.byKey(const ValueKey('checkCatalog')));
    await tester.pumpAndSettle();
    expect(find.text('Hoạt chất · hàm lượng: Fakeamin 500mg'), findsOneWidget);

    await tester.tap(find.text('Dùng tên này'));
    await tester.pump();
    expect(find.widgetWithText(TextField, 'Fakemol 500'), findsOneWidget);
    expect(find.text('Có trong danh mục thuốc'), findsOneWidget);
    expect(find.byKey(const ValueKey('lowConfidence')), findsNothing);
    expect(_saveBtn(tester).onPressed, isNotNull);
  });

  testWidgets('offline check shows a manual-check message', (tester) async {
    await tester.binding.setSurfaceSize(const Size(800, 2400));
    await tester.pumpWidget(_wrap(const [
      ParsedPrescriptionItem(medicineName: 'Fakemol', doseText: '1 viên'),
    ], repo: _FakeCatalog(const [], fail: true)));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('checkCatalog')));
    await tester.pumpAndSettle();
    expect(find.text('Không đối chiếu được (mất mạng?). Hãy kiểm tra tay.'),
        findsOneWidget);
    // Lỗi mạng không phải "không có trong danh mục" -> không khoá thêm.
    expect(_saveBtn(tester).onPressed, isNotNull);
  });
}
