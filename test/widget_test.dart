// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:image_processing_app/core/providers.dart';
import 'package:image_processing_app/data/adapters/remote_auth_repository.dart';
import 'package:image_processing_app/data/local/cache_store.dart';
import 'package:image_processing_app/main.dart';

// LƯU Ý LỊCH SỬ (Day 3): phần dưới đây trước là test "Counter increments" mặc định của
// template Flutter (tìm find.text('0') / Icons.add) nhưng app KHÔNG có counter => luôn FAIL.
// Đã thay bằng smoke test thật của app.
//
// Day 9: màn hình dùng Riverpod nên test phải bọc `ProviderScope` (giống hệt
// `main()`); `MedAssistApp` tách riêng chính là để test làm được điều này mà
// không sửa code chạy thật.

void main() {
  testWidgets('App khởi động và hiển thị màn hình Đăng nhập', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      ProviderScope(
        // Từ Day 13 màn dùng `AppLocalizations` nên test phải có
        // `Localizations`, nếu không sẽ ném "Null check operator".
        overrides: [
          tokenStoreProvider.overrideWithValue(InMemoryTokenStore()),
          cacheStoreProvider.overrideWithValue(InMemoryCacheStore()),
        ],
        child: const MyApp(),
      ),
    );

    expect(find.byType(TextField), findsNWidgets(2));
    expect(find.widgetWithText(ElevatedButton, 'Đăng nhập'), findsOneWidget);
    expect(find.text('Đăng ký tài khoản mới'), findsOneWidget);
  });
}
