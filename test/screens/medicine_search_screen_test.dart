// Day 3 — test cho màn Tìm kiếm thuốc.
// Day 9 — màn dùng Riverpod; repository được override bằng fake nên test
// offline hoàn toàn và kiểm tra được cả nhánh "có kết quả".

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';
import 'package:image_processing_app/screens/medicine_search_screen.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';

/// Trả về 1 thuốc cho mọi tên hợp lệ; `null` = ném lỗi mạng.
class _FakeMedicineRepository implements MedicineRepository {
  _FakeMedicineRepository({this.fail = false});

  final bool fail;

  @override
  Future<MedicineSearchResponse> search(String name) async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.network,
        message: 'Không kết nối được tới máy chủ.',
      );
    }
    return MedicineSearchResponse(
      results: [
        const Medicine(
          name: 'Paracetamol 500mg',
          composition: 'Paracetamol',
          uses: 'Giảm đau, hạ sốt',
          excellentReviewPercent: 80,
        ),
      ],
      query: name,
      total: 1,
    );
  }

  @override
  Future<List<HistoryItem>> history() async => const [];

  @override
  Future<void> deleteHistory(int id) async {}
}

Widget _wrap(Widget child, {bool fail = false, Locale? locale}) =>
    ProviderScope(
      overrides: [
        medicineRepositoryProvider
            .overrideWithValue(_FakeMedicineRepository(fail: fail)),
        // Cache offline dùng `shared_preferences` — không có trong test.
        cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
      ],
      // Từ Day 13 mọi chuỗi đến từ `AppLocalizations.of(context)`; thiếu
      // `Localizations` sẽ ném "Null check operator used on a null value".
      child: MaterialApp(
        locale: locale ?? const Locale('vi'),
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: child,
      ),
    );

void main() {
  testWidgets('bỏ trống tên thuốc -> hiện cảnh báo, KHÔNG gọi API', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(_wrap(const MedicineSearchPage()));

    expect(find.text('Tìm kiếm thuốc'), findsOneWidget);
    expect(find.widgetWithText(ElevatedButton, 'Tìm kiếm'), findsOneWidget);

    await tester.tap(find.widgetWithText(ElevatedButton, 'Tìm kiếm'));
    await tester.pump();

    expect(find.text('Vui lòng nhập tên thuốc!'), findsOneWidget);
  });

  testWidgets('có kết quả -> hiện tên thuốc từ model', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(_wrap(const MedicineSearchPage()));

    await tester.enterText(find.byType(TextField), 'paracetamol');
    await tester.tap(find.widgetWithText(ElevatedButton, 'Tìm kiếm'));
    await tester.pumpAndSettle();

    expect(find.text('Paracetamol 500mg'), findsOneWidget);
    expect(find.text('Thành Phần: Paracetamol'), findsOneWidget);
    // Trường server không trả -> "Không có thông tin", không phải "0%".
    expect(find.text('Nhà Sản Xuất: Không có thông tin'), findsOneWidget);
    expect(find.text('Đánh Giá Xuất Sắc: 80%'), findsOneWidget);
  });

  testWidgets('mất mạng -> hiện thông điệp của AppException', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(_wrap(const MedicineSearchPage(), fail: true));

    await tester.enterText(find.byType(TextField), 'paracetamol');
    await tester.tap(find.widgetWithText(ElevatedButton, 'Tìm kiếm'));
    await tester.pumpAndSettle();

    expect(find.text('Không kết nối được tới máy chủ.'), findsOneWidget);
    expect(find.text('Thử lại'), findsOneWidget);
  });
}
