// Day 14 — test cho màn Lịch sử: trạng thái rỗng / lỗi / ngoại tuyến.
//
// `history_screen.dart` trước Day 14 có coverage ~1% vì chưa ai chạm tới nó.
// Đây là màn dễ vỡ nhất: 3 nhánh hiển thị (danh sách / rỗng / lỗi) cùng banner
// ngoại tuyến và hộp thoại xoá.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_processing_app/core/network/api_exception.dart';
import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/models/history_item.dart';
import 'package:image_processing_app/data/models/medicine.dart';
import 'package:image_processing_app/domain/ports/medicine_repository.dart';
import 'package:image_processing_app/l10n/generated/app_localizations.dart';
import 'package:image_processing_app/screens/history_screen.dart';
import 'package:image_processing_app/widgets/medical_disclaimer.dart';

/// History trả về sẵn danh sách, hoặc luôn lỗi mạng.
class _FakeHistoryRepository implements MedicineRepository {
  _FakeHistoryRepository({this.items = const [], this.fail = false});

  final List<HistoryItem> items;
  final bool fail;

  @override
  Future<List<HistoryItem>> history() async {
    if (fail) {
      throw const AppException(
        kind: AppExceptionKind.network,
        message: 'Không kết nối được máy chủ.',
      );
    }
    return items;
  }

  @override
  Future<MedicineSearchResponse> search(String name) async =>
      const MedicineSearchResponse(results: []);

  @override
  Future<void> deleteHistory(int id) async {}
}

Widget _wrap(MedicineRepository repo, CacheStore cache, {Key? key}) =>
    ProviderScope(
      key: key,
      overrides: [
        medicineRepositoryProvider.overrideWithValue(repo),
        cacheStoreProvider.overrideWithValue(cache),
      ],
      child: const MaterialApp(
        locale: Locale('vi'),
        localizationsDelegates: AppLocalizations.localizationsDelegates,
        supportedLocales: AppLocalizations.supportedLocales,
        home: HistoryScreen(),
      ),
    );

void main() {
  testWidgets('có dữ liệu -> hiện danh sách và disclaimer', (tester) async {
    await tester.pumpWidget(_wrap(
      _FakeHistoryRepository(
        items: const [
          HistoryItem(
            id: 1,
            filepath: 'a.jpg',
            searchResults: [Medicine(name: 'Paracetamol')],
          ),
        ],
      ),
      InMemoryCacheStore(),
    ));
    await tester.pumpAndSettle();

    expect(find.text('Tệp: a.jpg'), findsOneWidget);
    // Lịch sử render theo dạng "Tên thuốc: <tên>" nên dùng textContaining.
    expect(find.textContaining('Paracetamol'), findsOneWidget);
    // Bắt buộc: màn ra kết quả phải có disclaimer y khoa.
    expect(find.byType(MedicalDisclaimer), findsOneWidget);
  });

  testWidgets('rỗng -> hiện "Chưa có lịch sử", KHÔNG trắng màn',
      (tester) async {
    await tester
        .pumpWidget(_wrap(_FakeHistoryRepository(), InMemoryCacheStore()));
    await tester.pumpAndSettle();

    expect(find.text('Chưa có lịch sử'), findsOneWidget);
    // Rỗng thì không hiện danh sách.
    expect(find.textContaining('Tệp:'), findsNothing);
  });

  testWidgets('lỗi mạng + không có cache -> báo lỗi kèm nút Thử lại', (
    tester,
  ) async {
    await tester.pumpWidget(_wrap(
      _FakeHistoryRepository(fail: true),
      InMemoryCacheStore(),
    ));
    await tester.pumpAndSettle();

    expect(find.text('Không kết nối được máy chủ.'), findsOneWidget);
    expect(find.text('Thử lại'), findsOneWidget);
  });

  testWidgets('ngoại tuyến -> hiện banner + dữ liệu cache (không trắng màn)', (
    tester,
  ) async {
    final cache = InMemoryCacheStore();

    // Lượt 1: có mạng -> cache được ghi.
    await tester.pumpWidget(_wrap(
      _FakeHistoryRepository(
        items: const [HistoryItem(id: 1, filepath: 'a.jpg')],
      ),
      cache,
      key: const ValueKey('online'),
    ));
    await tester.pumpAndSettle();
    // `load()` ghi cache bằng `unawaited` -> thêm một nhịp để ghi xong.
    await tester.pump();

    // Lượt 2: tắt mạng -> phải ra banner ngoại tuyến, vẫn thấy dữ liệu.
    // ⚠️ **Phải đổi `key`**: nếu không, `pumpWidget` chỉ update tại chỗ, `initState`
    // không chạy lại và `load()` không được gọi -> sẽ còn dữ liệu cũ hiện ra.
    await tester.pumpWidget(_wrap(
      _FakeHistoryRepository(fail: true),
      cache,
      key: const ValueKey('offline'),
    ));
    await tester.pumpAndSettle();

    expect(find.textContaining('Dữ liệu ngoại tuyến'), findsOneWidget);
    expect(find.text('Tệp: a.jpg'), findsOneWidget);
  });

  testWidgets('xoá: hỏi lại trước, bấm Huỷ thì giữ nguyên dữ liệu', (
    tester,
  ) async {
    await tester.pumpWidget(_wrap(
      _FakeHistoryRepository(
        items: const [HistoryItem(id: 1, filepath: 'a.jpg')],
      ),
      InMemoryCacheStore(),
    ));
    await tester.pumpAndSettle();

    await tester.tap(find.byIcon(Icons.delete));
    await tester.pumpAndSettle();

    expect(find.text('Xoá bản ghi này?'), findsOneWidget);

    await tester.tap(find.text('Huỷ'));
    await tester.pumpAndSettle();

    // Chưa xoá -> dữ liệu vẫn còn.
    expect(find.text('Tệp: a.jpg'), findsOneWidget);
  });
}
