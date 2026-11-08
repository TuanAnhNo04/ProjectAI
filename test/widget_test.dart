// Smoke test khởi động app + cổng đồng ý chính sách dữ liệu.
//
// LỊCH SỬ (Day 3): trước đây là test "Counter increments" mặc định của template —
// app không có counter nên luôn FAIL; đã thay bằng smoke test thật.
// Day 9: màn dùng Riverpod nên test bọc `ProviderScope` như `main()`.
// Day 29: app chặn bằng màn đồng ý ở lần mở đầu tiên -> test cả hai nhánh.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/data/local/consent_store.dart';
import 'package:image_processing_app/main.dart';

Widget _app(ConsentStore consent) => ProviderScope(
      // Từ Day 13 màn dùng `AppLocalizations` nên test phải có `Localizations`,
      // nếu không sẽ ném "Null check operator".
      overrides: [
        tokenStoreProvider.overrideWithValue(InMemoryTokenStore()),
        cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
        consentStoreProvider.overrideWithValue(consent),
      ],
      child: const MyApp(),
    );

void main() {
  testWidgets('Đã đồng ý -> app khởi động vào màn Đăng nhập', (tester) async {
    await tester.pumpWidget(_app(InMemoryConsentStore(accepted: true)));
    await tester.pumpAndSettle();

    expect(find.byType(TextField), findsNWidgets(2));
    expect(find.widgetWithText(ElevatedButton, 'Đăng nhập'), findsOneWidget);
    expect(find.text('Đăng ký tài khoản mới'), findsOneWidget);
  });

  testWidgets('Chưa đồng ý -> hiện màn đồng ý, CHƯA có ô đăng nhập',
      (tester) async {
    await tester.pumpWidget(_app(InMemoryConsentStore()));
    await tester.pumpAndSettle();

    expect(find.text('Đồng ý và tiếp tục'), findsOneWidget);
    expect(find.byType(TextField), findsNothing);
  });

  testWidgets('Bấm Đồng ý -> lưu lại và vào màn Đăng nhập', (tester) async {
    final store = InMemoryConsentStore();
    await tester.pumpWidget(_app(store));
    await tester.pumpAndSettle();

    await tester.tap(find.text('Đồng ý và tiếp tục'));
    await tester.pumpAndSettle();

    expect(await store.hasAccepted(), isTrue);
    expect(find.byType(TextField), findsNWidgets(2));
  });
}
